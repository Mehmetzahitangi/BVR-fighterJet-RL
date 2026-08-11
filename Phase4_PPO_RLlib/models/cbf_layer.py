import cvxpy as cp
import numpy as np

class CBFFilter:
    def __init__(self, action_dim, gamma=0.1):
        """
        Kontrol Bariyer Fonksiyonu (CBF) Kalkanı
        action_dim: Eylem uzayı boyutu (örn: 3 -> Pitch, Roll, Throttle)
        gamma: Bariyerin ne kadar 'esnek' veya 'sert' olduğunu belirler (0 ile 1 arası).
        """
        self.action_dim = action_dim
        self.gamma = gamma

    def get_safe_action(self, u_rl, x_current, A, B, C_safe, d_safe):
        """
        u_rl: Ajanın (SAC) ham komutu
        x_current: Uçağın o anki durumu (11 boyutlu)
        A, B: DMD'den yeni gelen doğrusal dinamik matrisleri
        C_safe, d_safe: Zarf sınırları (örn: Pitch <= 30, Roll <= 60)
        """
        # u: Bulmaya çalıştığımız 'Güvenli' eylem değişkeni
        u = cp.Variable(self.action_dim)

        # 1. Amaç: Ajanın komutundan (u_rl) mümkün olduğunca az sap!
        objective = cp.Minimize(cp.sum_squares(u - u_rl))

        # 2. CBF Kalkan Formülü (Az önce çıkardığımız matematik)
        # h_k: Sınıra ne kadar uzağız? (Mesafe)
        h_k = d_safe - (C_safe @ x_current)
        
        # limit: Uçağın ivme/dinamik hesabı
        limit = (C_safe @ x_current) - (C_safe @ A @ x_current) + (self.gamma * h_k)
        
        # 3. Kısıtlar (Constraints)
        constraints = [
            (C_safe @ B) @ u <= limit, # CBF Aerodinamik Kısıtı
            u >= -1.0,                 # JSBSim Eylem Alt Sınırı
            u <= 1.0                   # JSBSim Eylem Üst Sınırı
        ]

        # 4. Optimizasyonu Çöz
        prob = cp.Problem(objective, constraints)
        
        try:
            prob.solve(solver=cp.OSQP) # OSQP solver, mikrosaniyeler içinde çözer
            return u.value
        except:
            # Nadir bir durum (Infeasible): Çözüm bulunamazsa nötr komut ver
            return np.zeros(self.action_dim)