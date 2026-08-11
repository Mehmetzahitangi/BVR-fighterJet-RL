import torch
import torch.nn as nn
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer

class SafeActor(nn.Module):
    # num_constraints: Uçağa koyduğumuz sınır sayısı (Örn: Pitch ve Roll için 2)
    def __init__(self, state_dim, action_dim, num_constraints=4):
        super(SafeActor, self).__init__()
        
        # 1. Klasik SAC Sinir Ağı
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh() # Ham aksiyonu -1 ile 1 arasına sıkıştır
        )
        
        # 2. GERÇEK CBF KALKANI KATMANI
        self.cbf_layer = self._setup_differentiable_cbf(action_dim, num_constraints)

    def _setup_differentiable_cbf(self, action_dim, num_constraints):
        """ Sahte limitler silindi! Yerine Gerçek CBF Matrisleri (CB ve limit) bağlandı. """
        u_rl = cp.Parameter(action_dim)
        CB_param = cp.Parameter((num_constraints, action_dim)) # C_safe @ B Matrisi
        limit_param = cp.Parameter(num_constraints)          # Dinamik Limit (İvme+Mesafe)
        
        u_safe = cp.Variable(action_dim)
        
        # Amaç: Ajanın komutundan mümkün olduğunca az sap
        objective = cp.Minimize(cp.sum_squares(u_safe - u_rl))
        
        # GERÇEK MATEMATİKSEL KISITLAR
        constraints = [
            CB_param @ u_safe <= limit_param, # Gerçek Aerodinamik Zarf Sınırı
            u_safe >= -1.0, 
            u_safe <= 1.0
        ] 
        
        prob = cp.Problem(objective, constraints)
        # cvxpylayers, PyTorch'tan gelen batched (yığın) tensörleri otomatik işler
        return CvxpyLayer(prob, parameters=[u_rl, CB_param, limit_param], variables=[u_safe])

    def forward(self, state, A=None, B=None, C_safe=None, d_safe=None, gamma=0.1):
        """ 
        Artık sadece A ve B yetmez, sınır matrislerini (C_safe, d_safe) de istiyoruz!
        """
        raw_action = self.net(state)
        dummy_log_prob = torch.zeros((state.shape[0], 1)).to(state.device)
        
        # EĞER KALKAN AKTİFSE VE MATRİSLER GELDİYSE:
        if A is not None and B is not None and C_safe is not None and d_safe is not None:
            # PyTorch Batch (Yığın) Matris Çarpımları (bmm)
            
            # 1. h_k (Sınıra olan mesafe) = d_safe - (C_safe @ x)
            Cx = torch.bmm(C_safe, state.unsqueeze(-1)).squeeze(-1) # Boyut: (Batch, num_constraints)
            h_k = d_safe - Cx 
            
            # 2. Dinamik İvme Limiti = Cx - (C_safe @ A @ x) + (gamma * h_k)
            # DMD'den gelen A ve B 2D'dir; matmul C_safe'i (1,4,14) otomatik broadcast eder
            # (bmm yerine matmul: bmm iki tarafta da 3D ister ve RuntimeError fırlatırdı)
            CA = torch.matmul(C_safe, A)
            CAx = torch.matmul(CA, state.unsqueeze(-1)).squeeze(-1)
            limit = Cx - CAx + (gamma * h_k) # Boyut: (Batch, num_constraints)
            
            # 3. C_safe @ B
            CB = torch.matmul(C_safe, B)
            
            # 4. Kalkan Çözümü (Hata Yakalama - Try/Except ile Zırhlı!)
            try:
                safe_action, = self.cbf_layer(raw_action, CB, limit)
                
                # NaN (Infeasible) Koruması
                if torch.isnan(safe_action).any():
                    return raw_action, dummy_log_prob
                    
                return safe_action, dummy_log_prob
                
            except Exception:
                # EĞER ÇÖZÜCÜ (SOLVER) MATEMATİĞİN İÇİNDEN ÇIKAMAZ VE HATA FIRLATIRSA:
                # Oyunu çökertme! Kalkanı o saniyeliğine devreden çıkar ve ham aksiyonla devam et.
                return raw_action, dummy_log_prob
        
        # Kalkan aktif değilse ham aksiyon
        return raw_action, dummy_log_prob