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
        NaN/Inf sanatması: bozuk veri tamponu kirletin ve SVD'yi bozmasın diye temizlenir.
        """
        current_state = np.nan_to_num(np.asarray(current_state, dtype=np.float64),
                                      nan=0.0, posinf=10.0, neginf=-10.0)
        action = np.nan_to_num(np.asarray(action, dtype=np.float64),
                               nan=0.0, posinf=1.0, neginf=-1.0)
        next_state = np.nan_to_num(np.asarray(next_state, dtype=np.float64),
                                   nan=0.0, posinf=10.0, neginf=-10.0)

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
        Her arıza durumunda (None, None) döner -> env tarafı kalkanı pasifleştirir.
        Böylece solver asla çökmez ve kalkan "yanlış değil, aktifse güvenilir" çalışır.
        """
        if not self.is_ready:
            return None, None # Henüz 1 saniyelik uçuş verisi birikmedi

        try:
            # Omega matrisini oluştur: X ve U'yu alt alta birleştir
            Omega = np.vstack((self.X, self.U))

            # Girdi sağlık kontrolü: herhangi bir yerinde NaN/Inf varsa kalkan pasif
            if not (np.isfinite(Omega).all() and np.isfinite(self.X_prime).all()):
                return None, None

            # SVD (Singular Value Decomposition) - Gürültü Filtreleme
            U_svd, Sigma, Vh = svd(Omega, full_matrices=False)

            # Göreceli eşik: küçük ama önemli dinamikleri koru ama gürültü tabanını kes.
            # (Mutlak 1e-4 yerine en büyük tekil değerin 0.001'i de değerlendirilir.)
            relative_threshold = max(float(Sigma[0]) * 1e-3, truncation_threshold)
            rank = int(np.sum(Sigma > relative_threshold))

            # Hiç anlamlı dinamik kalmamışsa kalkanı pasifleştir (sistem çökmez)
            if rank == 0:
                return None, None

            U_trunc = U_svd[:, :rank]
            # Ters alırken 0/patlama koruması (epsilon tabanı)
            Sigma_floor = np.maximum(Sigma[:rank], 1e-8)
            Sigma_inv = np.diag(1.0 / Sigma_floor)
            Vh_trunc = Vh[:rank, :]

            # Pseudo-inverse matris çarpımı ile [A, B] bloğunu bul
            # Formül: [A, B] = X' * V * Sigma^-1 * U^T
            AB = self.X_prime @ Vh_trunc.T @ Sigma_inv @ U_trunc.T

            # Bloğu A ve B matrisleri olarak ikiye böl
            A = AB[:, :self.state_dim]
            B = AB[:, self.state_dim:]

            # Çıktı örnekleme: boyut ve sonluluk kontrolü yanlışsa kalkan pasif
            if A.shape != (self.state_dim, self.state_dim) or B.shape != (self.state_dim, self.action_dim):
                return None, None
            if not (np.isfinite(A).all() and np.isfinite(B).all()):
                return None, None

            return A, B

        except Exception as e:
            # Hesaplama hatası (LinAlgError vb.) -> kalkan bu step'te pasif, sistem çökmez
            print(f"[DMD] Matris hesabı başarısız, kalkan atlandı: {e}")
            return None, None