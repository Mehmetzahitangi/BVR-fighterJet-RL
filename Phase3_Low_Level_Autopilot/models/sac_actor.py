import math
import torch
import torch.nn as nn
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer

LOG_SIG_MIN = -20.0
LOG_SIG_MAX = 2.0

class SafeActor(nn.Module):
    # num_constraints: Uçağa koyduğumuz sınır sayısı (Hibrit: 4 açı + 4 hız = 8)
    def __init__(self, state_dim, action_dim, num_constraints=8):
        super(SafeActor, self).__init__()

        # 1. MEAN AĞI (deterministik kısım)
        # NOT: Son katmanda Tanh YOK — Tanh forward içinde uygulanır.
        # (state_dict anahtarları değişmez: net.0/2/4 = katmanlar, eski checkpoint'ler uyumlu)
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )

        # 2. GERÇEK SAC GAUSSIAN KAFASI (Entropi Düzenlemesi)
        # Sabit log_std parametresi: keşif gürültüsü öğrenilir, entropy düzenlemesi gerçek olur.
        # (Eski kod dummy log_prob=0 döndürüyordu -> SAC aslında DDPG gibi davranıyordu!)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

        # 3. GERÇEK CBF KALKANI KATMANI
        self.cbf_layer = self._setup_differentiable_cbf(action_dim, num_constraints)

    def _setup_differentiable_cbf(self, action_dim, num_constraints):
        """ Sahte limitler silindi! Yerine Gerçek CBF Matrisleri (CB ve limit) bağlandı. """
        u_rl = cp.Parameter(action_dim)
        CB_param = cp.Parameter((num_constraints, action_dim)) # C_safe @ B Matrisi
        limit_param = cp.Parameter(num_constraints)          # Dinamik Limit (İvme+Mesafe)

        u_safe = cp.Variable(action_dim)
        slack = cp.Variable(num_constraints)  # DMD kestirimi gürültülüyken problem hiç infeasible olmasın

        # Amaç: Ajanın komutundan mümkün olduğunca az sap (slack cezasi 100.0)
        objective = cp.Minimize(cp.sum_squares(u_safe - u_rl) + 100.0 * cp.sum(slack))

        # GERÇEK MATEMATİKSEL KISITLAR (yumuşak CBF: infeasible durumda slack devreye girer,
        # feasible iken slack=0 ve davranış sert kısıtla birebir aynıdır)
        constraints = [
            CB_param @ u_safe <= limit_param + slack, # Gerçek Aerodinamik Zarf Sınırı
            slack >= 0.0,
            u_safe >= -1.0,
            u_safe <= 1.0
        ]

        prob = cp.Problem(objective, constraints)
        # cvxpylayers, PyTorch'tan gelen batched (yığın) tensörleri otomatik işler
        return CvxpyLayer(prob, parameters=[u_rl, CB_param, limit_param], variables=[u_safe, slack])

    @staticmethod
    def _gaussian_log_prob(u, mu, std):
        """ N(mu, std) altında u'nun log-olasılığı (reparametrizasyon ile türevlenebilir) """
        return -0.5 * ((u - mu) / std).pow(2) - torch.log(std) - 0.5 * math.log(2 * math.pi)

    def forward(self, state, A=None, B=None, C_safe=None, d_safe=None, gamma=0.1):
        """
        Gerçek SAC Gaussian politikası:
        - Eğitim modunda (self.training=True): tanh-squash Gaussian'dan örnekleme yapar
          ve gerçek log_prob döndürür (ALPHA*log_prob terimleri artık anlamlı!).
        - Eval modunda (self.training=False): gürültüsüz mean aksiyon (tanh(mu)).
        - Kalkan (CBF) her iki modda da örneklenen aksiyona uygulanır.
        """
        mu = self.net(state)

        if self.training:
            # Reparametrizasyon ile örnekleme + tanh-squash log_prob düzeltmesi
            # log_prob = log N(u) - log(1 - tanh(u)^2)  (SB3 SAC standardı)
            log_std = torch.clamp(self.log_std, LOG_SIG_MIN, LOG_SIG_MAX)
            std = log_std.exp().expand_as(mu)
            u = mu + std * torch.randn_like(mu)
            raw_action = torch.tanh(u)
            log_prob = self._gaussian_log_prob(u, mu, std)
            log_prob = log_prob - torch.log(1.0 - raw_action.pow(2) + 1e-6)
            log_prob = log_prob.sum(dim=-1, keepdim=True)
        else:
            # Test/dağıtım: deterministik mean aksiyon
            raw_action = torch.tanh(mu)
            log_prob = torch.zeros((state.shape[0], 1)).to(state.device)

        # EĞER KALKAN AKTİFSE VE MATRİSLER GELDİYSE:
        if A is not None and B is not None and C_safe is not None and d_safe is not None:
            # matmul kullanılır (bmm değil): C_safe (1,8,14) ile batch'lenmiş state
            # (N,14) veya tek state (1,14) olsun, matmul broadcast ile ikisini de çözer.

            # 1. h_k (Sınıra olan mesafe) = d_safe - (C_safe @ x)
            Cx = torch.matmul(C_safe, state.unsqueeze(-1)).squeeze(-1) # Boyut: (Batch, num_constraints)
            h_k = d_safe - Cx

            # 2. Dinamik İvme Limiti = Cx - (C_safe @ A @ x) + (gamma * h_k)
            CA = torch.matmul(C_safe, A)
            CAx = torch.matmul(CA, state.unsqueeze(-1)).squeeze(-1)
            limit = Cx - CAx + (gamma * h_k) # Boyut: (Batch, num_constraints)

            # 3. C_safe @ B
            CB = torch.matmul(C_safe, B)

            # 4. Kalkan Çözümü (Hata Yakalama - Try/Except ile Zırhlı!)
            # Slack sayesinde problem artık matematiksel olarak her zaman çözülebilir;
            # try/except yalnızca çözücünün numerik uç durumlarında emniyet kemeri gibidir.
            try:
                safe_action, _ = self.cbf_layer(raw_action, CB, limit)

                # NaN (Infeasible) Koruması
                if torch.isnan(safe_action).any():
                    return raw_action, log_prob

                return safe_action, log_prob

            except Exception:
                # EĞER ÇÖZÜCÜ (SOLVER) MATEMATİĞİN İÇİNDEN ÇIKAMAZ VE HATA FIRLATIRSA:
                # Oyunu çökertme! Kalkanı o saniyeliğine devreden çıkar ve ham aksiyonla devam et.
                return raw_action, log_prob

        # Kalkan aktif değilse ham aksiyon
        return raw_action, log_prob
