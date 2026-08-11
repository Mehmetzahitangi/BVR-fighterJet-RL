import math
import numpy as np

class AMRAAM:
    """
    Taktiksel BVR (Görüş Ötesi) eğitimleri için basitleştirilmiş AIM-120 Füze Matematiği.
    YENİ: Radar Kilidi (Doppler Notch) ve Kaçışsız Bölge (No-Escape Zone) mantığı eklendi!
    """
    def __init__(self):
        self.active = False
        self.pos = np.array([0.0, 0.0, 0.0]) # Füzenin x,y,z konumu (Metre)
        self.speed_m_s = 1360.0 # Mach 4 (Füzenin hızı)
        self.max_flight_time = 60.0 # Füzenin pili/yakıtı 60 saniye dayanır
        self.current_flight_time = 0.0
        self.hit_radius = 150.0 # Patlama yarıçapı (Metre)

    def fire(self, launcher_pos_m):
        """ Füzeyi uçaktan ateşle """
        self.active = True
        self.pos = np.array(launcher_pos_m, dtype=np.float64)
        self.current_flight_time = 0.0
        print("🚀 FOX-3! AMRAAM Ateşlendi!")

    def step(self, target_pos_m, target_aa_deg, dt=1.0/60.0):
        """ 
        target_aa_deg: Hedefin füzeye olan bakış açısı (Aspect Angle). 
        """
        if not self.active:
            return "INACTIVE"

        self.current_flight_time += dt

        # 1. Füzenin pili bitti mi?
        if self.current_flight_time >= self.max_flight_time:
            self.active = False
            print("💨 Füze Enerjisi Tükendi (MISS)!")
            return "MISS_ENERGY"

        target_pos = np.array(target_pos_m, dtype=np.float64)
        direction_vec = target_pos - self.pos
        distance_to_target = np.linalg.norm(direction_vec)

        # =================================================================
        # 2. YENİ KURAL: DOPPLER NOTCH (DİK AÇI KAÇIŞI)
        # Eğer düşman füzeye 75° ile 105° arası bir dik açıyla uçuyorsa
        # ve füze 5 km'den (5000m) uzaktaysa radar kilidi anında kırılır!
        # =================================================================
        if 75.0 <= target_aa_deg <= 105.0 and distance_to_target > 5000.0:
            self.active = False
            print(f"📡 Düşman Notch Yaptı! Radar Kilidi Kırıldı! (Mesafe: {distance_to_target:.1f}m, Açı: {target_aa_deg:.1f}°)")
            return "MISS_NOTCH"

        # 3. Vuruş Kontrolü
        if distance_to_target <= self.hit_radius:
            self.active = False
            print(f"💥 Hedef Vuruldu! (Mesafe: {distance_to_target:.1f}m)")
            return "HIT"

        # 4. Füzeyi hedefe doğru uçur
        direction_unit = direction_vec / (distance_to_target + 1e-8)
        self.pos += direction_unit * (self.speed_m_s * dt)

        return "TRACKING"