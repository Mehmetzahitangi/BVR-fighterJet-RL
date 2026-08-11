import numpy as np
from scipy.linalg import svd

class RealTimeDMDc:
    def __init__(self, state_dim, action_dim, window_size=60):
        """
        DMDc (Kontrollü Dinamik Mod Ayrıştırması) Sınıfı
        state_dim: Gözlem uzayı boyutu (örn: 14)
        action_dim: Aksiyon uzayı boyutu (örn: 3)
        window_size: Hafızada tutulacak adım sayısı (60 Hz ise 60 adım = 1 saniye)
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

    def compute_matrices(self, truncation_threshold=1e-4):
        """
        Hafızadaki verileri kullanarak A ve B matrislerini hesaplar.
        """
        if not self.is_ready:
            return None, None # Henüz 1 saniyelik uçuş verisi birikmedi
        
        # Omega matrisini oluştur: X ve U'yu alt alta birleştir
        Omega = np.vstack((self.X, self.U))
        
        # SVD (Singular Value Decomposition) - Gürültü Filtreleme
        U_svd, Sigma, Vh = svd(Omega, full_matrices=False)
        
        # Önemsiz (çok küçük) dinamikleri kesip atma (Truncation)
        rank = np.sum(Sigma > truncation_threshold)
        U_trunc = U_svd[:, :rank]
        Sigma_inv = np.diag(1.0 / Sigma[:rank])
        Vh_trunc = Vh[:rank, :]
        
        # Pseudo-inverse matris çarpımı ile [A, B] bloğunu bul
        # Formül: [A, B] = X' * V * Sigma^-1 * U^T
        AB = self.X_prime @ Vh_trunc.T @ Sigma_inv @ U_trunc.T
        
        # Bloğu A ve B matrisleri olarak ikiye böl
        A = AB[:, :self.state_dim]
        B = AB[:, self.state_dim:]
        
        return A, B