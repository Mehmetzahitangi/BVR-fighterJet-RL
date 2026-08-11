import torch
import torch.nn as nn
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer

class SafeActor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(SafeActor, self).__init__()
        
        # 1. Klasik SAC Sinir Ağı (Ameliyat Kodumuzla %100 Uyumlu Katmanlar)
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh() # Ham aksiyonu -1 ile 1 arasına sıkıştır
        )
        
        # 2. CBF PyTorch Katmanı
        self.cbf_layer = self._setup_differentiable_cbf(action_dim)

    def _setup_differentiable_cbf(self, action_dim):
        """ cvxpy problemini Türevlenebilir (Differentiable) PyTorch katmanına çevirir """
        u_rl = cp.Parameter(action_dim)
        limit = cp.Parameter(action_dim)
        
        u_safe = cp.Variable(action_dim)
        
        # Amaç: Ajanın komutundan mümkün olduğunca az sap
        objective = cp.Minimize(cp.sum_squares(u_safe - u_rl))
        
        # Kısıtlar: Hem JSBSim fiziksel limitleri hem de CBF dinamik limiti
        constraints = [u_safe <= limit, u_safe >= -1.0, u_safe <= 1.0] 
        
        prob = cp.Problem(objective, constraints)
        return CvxpyLayer(prob, parameters=[u_rl, limit], variables=[u_safe])

    def forward(self, state, A=None, B=None):
        """ 
        state: 14 Boyutlu Gözlem Uzayı
        A, B: DMD'den gelen aerodinamik matrisler
        """
        # 1. Ağ ham komutu üretir (u_rl)
        raw_action = self.net(state)
        
        # Standart SAC'da log_prob hesaplanır. Uyum sorunu çıkmasın diye
        # (Tuple Unpacking hatasını engellemek için) boş bir tensör döndürüyoruz.
        dummy_log_prob = torch.zeros((state.shape[0], 1)).to(state.device)
        
        # 2. CBF Kalkanı (Eğer A ve B matrisleri geldiyse)
        if A is not None and B is not None:
            # NOT: İleride A ve B matrislerini (C_safe @ A @ x) formülü ile 
            # gerçek aerodinamik limite dönüştüreceğiz. Şimdilik sistemin 
            # hata vermeden türev alabilmesi için dummy bir limit (1.0) veriyoruz.
            dummy_limit = torch.ones_like(raw_action)
            
            # CBF Katmanı ham komutu ezer/filtreler (Gradient buradan süzülür)
            safe_action, = self.cbf_layer(raw_action, dummy_limit)
            return safe_action, dummy_log_prob
        
        # İlk saniyeler kalkan kapalıysa ham aksiyonu dön
        return raw_action, dummy_log_prob