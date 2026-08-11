import numpy as np
from scipy.linalg import svd

class RealTimeDMDc:
    def __init__(self, state_dim, action_dim, window_size=300):
        """
        DMDc (Kontrollü Dinamik Mod Ayrıştırması) Sınıfı
        state_dim: Gözlem uzayı boyutu (örn: 14)
        action_dim: Aksiyon uzayı boyutu (örn: 3)
        window_size: Hafızada tutulacak adım sayısı (60 Hz ise 300 adım = 5 saniye).
                     60 örnek 14 boyutlu dinamiği tanımlamak için yetersizdi (rank düşüktü);
                     300 örnek ile SVD kestirimi güvenilir hale gelir.
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.window_size = window_size
        
        # Veri Tamponları (Sliding Window Buffers)
        self.X = np.zeros((state_dim, window_size))
        self.U = np.zeros((action_dim, window_size))
        self.X_prime = np.zeros((state_dim, window_size))
        
        self.ptr = 0
        self.is_ready = False

    def add_data(self, current_state, action, next_state):
        """
        JSBSim'den gelen her adımdaki veriyi hafızaya ekler.
        Eski verilerin üzerine yazar (FIFO mantığı).
        """
        idx = self.ptr % self.window_size
        self.X[:, idx] = current_state
        self.U[:, idx] = action
        self.X_prime[:, idx] = next_state
        self.ptr += 1
        
        # Tampon tamamen dolduğunda matris hesabı yapılabilir duruma gelir
        if self.ptr >= self.window_size:
            self.is_ready = True

    def compute_matrices(self, truncation_threshold=1e-4, ridge=1e-4, return_stats=False):
        """
        Hafızadaki verileri kullanarak A ve B matrislerini hesaplar.

        İyileştirmeler (B kestirimi için kritik):
        1. Satır bazlı ölçekleme: irtifa (~30.000) ile açılar (~radyan) aynı SVD'de
           yarışmasın diye her satır kendi standart sapmasına bölünür. Aksi halde
           SVD'yi büyük büyüklüklü satırlar domine eder, küçük açı dinamikleri
           (pitch/roll → B satırları) kestirimde ezilir ve |C_safe@B| ~ 0.01'e
           düşer (kalkan işlevsiz kalır).
        2. Merkezleme: düz uçuş DC bileşeni (ortalama) regresyondan çıkarılır,
           SVD modlarının ölçeği iyileşir.
        3. Ridge (Tikhonov) düzenlemesi: gürültülü küçük tekil değerler
           1/σ yerine σ/(σ²+ridge) ile yumuşatılır; B'deki gürültü azalır.
        """
        if not self.is_ready:
            if return_stats:
                return None, None, {}
            return None, None # Henüz pencere dolmadı
        
        X = self.X
        U = self.U
        Xp = self.X_prime
        
        # --- 1. Satır Bazlı Ölçekleme ---
        # Sıfır varyanslı satırlar (örn: sabit hedef pitch/roll) ölçekten etkilenmez
        x_std = X.std(axis=1)
        u_std = U.std(axis=1)
        x_std[x_std < 1e-6] = 1.0
        u_std[u_std < 1e-6] = 1.0
        D_x = np.diag(1.0 / x_std)
        D_u = np.diag(1.0 / u_std)
        
        Xn = D_x @ X
        Un = D_u @ U
        Omega = np.vstack((Xn, Un))
        
        # --- 2. Merkezleme (DC bileşeni çıkar) ---
        Omega_c = Omega - Omega.mean(axis=1, keepdims=True)
        Xp_c = Xp - Xp.mean(axis=1, keepdims=True)
        
        # --- 3. SVD (Gürültü Filtreleme) ---
        U_svd, Sigma, Vh = svd(Omega_c, full_matrices=False)
        
        # --- 4. Truncation + Ridge ---
        rank = np.sum(Sigma > truncation_threshold)
        rank = max(1, min(rank, len(Sigma)))
        sigma_eff = Sigma[:rank] / (Sigma[:rank] ** 2 + ridge) # Tikhonov: σ/(σ²+λ)
        Sigma_inv = np.diag(sigma_eff)
        U_trunc = U_svd[:, :rank]
        Vh_trunc = Vh[:rank, :]
        
        # --- 5. Regresyon (ölçekli uzayda) ---
        AB_norm = Xp_c @ Vh_trunc.T @ Sigma_inv @ U_trunc.T
        
        # --- 6. Ölçeği Geri Kat ---
        # AB_norm = [A_scaled | B_scaled]  (14 x 17)
        A_scaled = AB_norm[:, :self.state_dim]
        B_scaled = AB_norm[:, self.state_dim:]
        A = A_scaled @ D_x   # A = A_scaled @ D_x: ölçekli durumdan ham duruma
        B = B_scaled @ D_u   # B = B_scaled @ D_u
        
        if return_stats:
            stats = {
                'rank': rank,
                'sigma_min': float(Sigma[rank - 1]),
                'sigma_max': float(Sigma[0]),
                'cond': float(Sigma[0] / (Sigma[rank - 1] + 1e-12)),
                'B_row_norm': np.linalg.norm(B, axis=1).tolist(),
                'CB_rowsum': float(np.abs(B[[3, 3, 2, 2]]).sum(axis=1).mean()) if self.state_dim > 3 else 0.0,
            }
            return A, B, stats
        return A, B