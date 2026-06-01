import math
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import jsbsim

class FighterEnv(gym.Env):
    """Faz 2 Nihai: Tabula Rasa (Temiz Sayfa) Seyrüsefer Çevresi"""
    
    def __init__(self):
        super(FighterEnv, self).__init__()
        
        print("JSBSim Faz 2 (Tabula Rasa) Başlatılıyor...")
        self.fdm = jsbsim.FGFDMExec(None)
        self.fdm.load_model('f16')

        # 60 FPS, daha hızlı eğitim ve gerçekçilik için zaman adımını sabitliyoruz
        self.fdm.set_dt(1.0 / 60.0)
        self.fdm['propulsion/set-running'] = -1

        # --- AKSİYON UZAYI ---
        # [Pitch Komutu, Roll Komutu, Gaz Komutu]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        # --- GÖZLEM UZAYI (11 SENSÖR) ---
        obs_low = np.array([
            0.0,        # 1. Mevcut İrtifa 
            0.0,        # 2. Mevcut Hız 
            0.0,        # 3. Hedef İrtifa 
            0.0,        # 4. Hedef Hız 
            -50000.0,   # 5. Delta İrtifa 
            -2.0,       # 6. Delta Hız 
            -math.pi,   # 7. Roll 
            -math.pi/2, # 8. Pitch 
            -2000.0,    # 9. VSI (Dikey Hız)
            -math.pi/2, # 10. Alpha (Hücum Açısı - AoA)
            -math.pi/2  # 11. Beta (Yana Kayma Açısı)
        ], dtype=np.float32)

        obs_high = np.array([
            50000.0,    # 1. Max İrtifa
            2.0,        # 2. Max Hız
            50000.0,    # 3. Max Hedef İrtifa
            2.0,        # 4. Max Hedef Hız
            50000.0,    # 5. Max Delta İrtifa
            2.0,        # 6. Max Delta Hız
            math.pi,    # 7. Max Roll
            math.pi/2,  # 8. Max Pitch
            2000.0,     # 9. Max VSI
            math.pi/2,  # 10. Max Alpha
            math.pi/2   # 11. Max Beta
        ], dtype=np.float32)

        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
        
        self.target_altitude = 0.0
        self.target_mach = 0.0
        self.current_step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.fdm.reset_to_initial_conditions(0)

        # RASTGELE BAŞLANGIÇ
        self.initial_altitude = self.np_random.uniform(2000.0, 40000.0)
        min_initial_mach, max_initial_mach = self.get_mach_limits(self.initial_altitude)
        self.initial_mach = self.np_random.uniform(min_initial_mach, max_initial_mach)

        # HEDEF İRTİFA (Tam Zarf: +/- 8000 ft)
        target_delta_altitude = self.np_random.uniform(-8000.0, 8000.0)
        self.target_altitude = self.initial_altitude + target_delta_altitude
        self.target_altitude = max(2000.0, min(40000.0, self.target_altitude))

        # DİNAMİK GÖREV SÜRESİ
        ekstra_saniye = (abs(target_delta_altitude) / 1000.0) * 20.0
        toplam_saniye = 60.0 + ekstra_saniye
        self.max_steps = int(toplam_saniye * 60) 

        # HEDEF HIZ (Güvenli Sınırlar İçinde)
        min_target_mach, max_target_mach = self.get_mach_limits(self.target_altitude)
        safe_min_target_mach = max(min_target_mach, self.initial_mach - 0.1)
        safe_max_target_mach = min(max_target_mach, self.initial_mach + 0.1)
        safe_max_target_mach = max(safe_min_target_mach, safe_max_target_mach)
        self.target_mach = self.np_random.uniform(safe_min_target_mach, safe_max_target_mach)

        # JSBSim Yüklemesi
        self.fdm['ic/h-sl-ft'] = self.initial_altitude 
        self.fdm['ic/mach'] = self.initial_mach   
        self.fdm['ic/gamma-deg'] = 0.0     
        self.fdm.run_ic()  

        self.fdm['propulsion/set-running'] = -1 
        self.fdm['gear/gear-cmd-norm'] = 0.0
        self.fdm.run()

        self.current_step = 0 
                
        info = {
            "mevcut_irtifa": self.fdm['position/h-sl-ft'],
            "hedef_irtifa": self.target_altitude,
            "mevcut_mach": self.fdm['velocities/mach'],
            "hedef_mach": self.target_mach
        }
        return self._get_obs(), info
    
    def step(self, action):
        # 1. FBW VE KOORDİNELİ DÖNÜŞ
        pitch_cmd = action[0]
        roll_cmd = action[1]
        throttle_cmd = action[2]

        self.fdm['fcs/elevator-cmd-norm'] = pitch_cmd
        self.fdm['fcs/aileron-cmd-norm'] = roll_cmd
        self.fdm['fcs/rudder-cmd-norm'] = roll_cmd * 0.3 # Adverse Yaw Koruması
        self.fdm['fcs/throttle-cmd-norm'] = (throttle_cmd + 1.0) / 2.0

        self.fdm.run()
        obs = self._get_obs()

        mevcut_irtifa = obs[0]
        mevcut_hiz = obs[1]
        delta_irtifa = obs[4]
        delta_hiz = obs[5]

        reward = 0.0
        terminated = False 
        truncated = False  

        self.current_step += 1 

        # DİNAMİK STALL KONTROLÜ
        anlik_min_hiz, _ = self.get_mach_limits(mevcut_irtifa)
        is_stall = (mevcut_hiz < anlik_min_hiz)


        # ÖDÜL/CEZA MİMARİSİ (Sıfır Korku)

        # 1. ÖLÜM SINIRLARI (Yerçekimi ve Fiziksel Limitler)
        if mevcut_irtifa < 1500.0 or mevcut_irtifa > 45000.0 or mevcut_hiz < 0.25:
            reward = -7000.0
            terminated = True
        else:
            # Yumuşak Stall Cezası (Rahatsız edici ama öldürücü değil)
            if is_stall:
                reward -= 2.0

            # 2. HEDEF ÇEKİM KUVVETİ (Dinamik Huni ve İğne Deliği Sistemi)
            reward -= (abs(delta_irtifa) / 5000.0)
            reward -= (abs(delta_hiz) * 0.8)

            # 3. TEMBELLİK CEZASI (Yaşama Vergisi)
            reward -= 0.05

            # 4. KADEMELİ ÖDÜL ZARFI VE İĞNE DELİĞİ
            if abs(delta_irtifa) < 2000.0:
                reward += 10.0 
                
            if abs(delta_irtifa) < 1000.0 and abs(delta_hiz) < 0.2:
                reward += 50.0 
        
            if abs(delta_irtifa) < 500.0 and abs(delta_hiz) < 0.1:
                # İğne Deliği Hassasiyet Çarpanı (Gravity Well)
                alt_precision = 1.0 - (abs(delta_irtifa) / 500.0) 
                spd_precision = 1.0 - (abs(delta_hiz) / 0.1)      
                precision_multiplier = (alt_precision * 0.7) + (spd_precision * 0.3)
                
                # Merkeze indikçe tam 150 puan!
                reward += 150.0 * precision_multiplier


        # GÖREV SÜRESİ DOLUMU
        if self.current_step >= self.max_steps:
            truncated = True
            reward -= 300.0

        info = {
            "mevcut_irtifa": mevcut_irtifa,
            "hedef_irtifa": self.target_altitude,
            "mevcut_mach": mevcut_hiz,
            "hedef_mach": self.target_mach,
            "kalan_zaman_adimi": self.max_steps - self.current_step 
        }

        return obs, reward, terminated, truncated, info
    
    def _get_obs(self):
        mevcut_irtifa = self.fdm['position/h-sl-ft']
        mevcut_hiz = self.fdm['velocities/mach']
        roll = self.fdm['attitude/roll-rad']
        pitch = self.fdm['attitude/pitch-rad']
        dikey_hiz = self.fdm['velocities/h-dot-fps']
        alpha = self.fdm['aero/alpha-rad'] 
        beta = self.fdm['aero/beta-rad']   

        delta_irtifa = self.target_altitude - mevcut_irtifa
        delta_hiz = self.target_mach - mevcut_hiz

        return np.array([
            mevcut_irtifa, mevcut_hiz, self.target_altitude, self.target_mach,
            delta_irtifa, delta_hiz, roll, pitch, dikey_hiz, alpha, beta 
        ], dtype=np.float32)

    def get_mach_limits(self, irtifa):
        """İrtifaya göre F-16 Dinamik Uçuş Zarfı Sınırları"""
        min_mach = 0.3 + (irtifa / 40000.0) * 0.35 
        max_mach = 1.15 + (irtifa / 40000.0) * 0.65 
        return min_mach, max_mach