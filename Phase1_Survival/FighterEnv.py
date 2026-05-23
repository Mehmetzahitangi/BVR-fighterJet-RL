import gymnasium as gym
import jsbsim
from gymnasium import spaces
import numpy as np

class FighterEnv(gym.Env):
    """Savaş Pilotu AI - Faz 1: Aerodinamik Hakimiyet ve JSBSim Ortamı"""

    def __init__(self):
        super(FighterEnv, self).__init__()

        self.fdm = jsbsim.FGFDMExec(None)
        self.fdm.load_model('f16')
        self.fdm.set_dt(1.0 / 60.0) # Saniyede 60 kare (60Hz)

        # 1GÖZLEM UZAYI (10 Değişken)
        # Sınırları -Sonsuz ile +Sonsuz arasında bırakıyoruz, sinir ağına girdi olarak vermeden önce normalize edeceğiz.
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)

        
        # EYLEM UZAYI (3 Değişken)
        # [İleri Hız, Dönüş Hızı (Yaw), Tırmanma Hızı (Z)]. Bunlarla ilgili verilecek kararlar JSBSim'in mekanik parçalarına (gaz kolu, elevator, aileron) iletilecek
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # JSBSim Başlangıç Şartları (Initial Conditions - IC)
        self.fdm['ic/h-sl-ft'] = 15000.0   # 15.000 feet irtifa
        self.fdm['ic/vc-kts'] = 400.0      # 400 knot hız
        self.fdm['ic/gamma-deg'] = 0.0     # Düz uçuş
        self.fdm.run_ic()                  # Ayarları uygula
        
        # Uçak havada başladığı için, motoru direkt çalıştırırız
        self.fdm['propulsion/set-running'] = -1 
        self.fdm.run() 

        obs = self._get_obs()

        # İlk bilgileri info içine koyuyoruz
        info = {
            "altitude": obs[0],
            "mach": obs[1],
            "pitch_deg": np.degrees(obs[3])
        }

        return obs, info

    def step(self, action):
            # Akisyonları JSBSim'e iletiriz (Mapping)
            forward_speed_cmd = action[0] 
            yaw_cmd = action[1]
            climb_rate_cmd = action[2]

            # Ajanın -1 ile 1 arasındaki komutlarını uçağın mekanik kanatçıklarına aktarıyoruz
            self.fdm['fcs/throttle-cmd-norm'] = (forward_speed_cmd + 1.0) / 2.0 # Değerler -1 ile +1 arasında geliyordu. Bununla [0, 1] arasına çekildi. 
            self.fdm['fcs/rudder-cmd-norm'] = yaw_cmd                           # Kuyruk dümeni
            self.fdm['fcs/aileron-cmd-norm'] = yaw_cmd                          # F-16 dönüşleri yatışla (roll) yaptığı için aileron beslenir
            self.fdm['fcs/elevator-cmd-norm'] = climb_rate_cmd                  # Burnun aşağı/yukarı hareketi 

            self.fdm.run()

            # Yeni Gözlemi Al
            obs = self._get_obs()
            
            # *** Ödül ve Bitiş Kontrolü (Faz 1.3: Seyrüsefer ve Stabilite) ***
            # Hedeflerimiz:
            TARGET_ALTITUDE = 15000.0  # 15.000 feet
            TARGET_MACH = 0.8          # 0.8 Mach

            altitude = obs[0]
            mach = obs[1]
            roll = obs[2]   # Radyan cinsinden "yatış" açısı
            pitch = obs[3]  # Radyan cinsinden "yunuslama" açısı
    

            # ----- ÖDÜL MÜHENDİSLİĞİ / REWARD SHAPING ----- 

            # 1. Hayatta Kalma (Ufak teşvik)
            r_alive = 0.1 

            # 2. İrtifa Cezası (Hedef irtifadan ne kadar uzaksa o kadar eksi puan, belirli bir irtifada (başlangıç irtifası) gitmeyi öğrenmesini istiyoruz şimdilik)
            # 5000 feet'lik bir sapmayı -1.0 ceza olarak normalize ediyoruz
            alt_error = abs(TARGET_ALTITUDE - altitude)
            r_alt = - (alt_error / 5000.0)

            # 3. Hız Cezası (Hedef hızdan ne kadar uzaksa, belirli bir hızda gitmesini istiyoruz (başlangıç hızı))
            mach_error = abs(TARGET_MACH - mach)
            r_mach = - (mach_error / 0.5)

            # 4. Duruş (Attitude) Cezası: Uçağın düz uçabilmesini istiyoruz Roll ve Pitch 0'a yakın olmalı.
            # Açılar radyan olduğu için doğrudan eksi olarak ekliyoruz (Örn: 0.5 radyan sapma = -0.5 ceza puanı)
            r_roll = - abs(roll)
            r_pitch = - abs(pitch)
            
            # Toplam Ödül
            reward = r_alive + r_alt + r_mach + r_roll + r_pitch

            # ----- ÖDÜL MÜHENDİSLİĞİ / REWARD SHAPING ----- 


            terminated = False
            truncated = False

            # Eğer 2.000 feet'in altına inerse veya uzaya (50.000 feet üstü) çıkarsa bölüm biter
            if altitude < 2000.0 or altitude > 50000.0:
                reward = -100.0 # Ölüm cezası
                terminated = True

            info = {
                "altitude": altitude,
                "mach": obs[1],
                "pitch_deg": np.degrees(obs[3])
            }

            return obs, reward, terminated, truncated, info

    def _get_obs(self):
            # JSBSim Property Tree'den 10 kritik veriyi çekiyoruz
            obs = np.array([
                self.fdm['position/h-sl-ft'],       # 0: İrtifa
                self.fdm['velocities/mach'],        # 1: Mach Hızı
                self.fdm['attitude/phi-rad'],       # 2: Roll (Yatış)
                self.fdm['attitude/theta-rad'],     # 3: Pitch (Yunuslama)
                self.fdm['attitude/psi-rad'],       # 4: Yaw (Sapma)
                self.fdm['velocities/p-rad_sec'],   # 5: Roll Rate
                self.fdm['velocities/q-rad_sec'],   # 6: Pitch Rate
                self.fdm['velocities/r-rad_sec'],   # 7: Yaw Rate
                self.fdm['aero/alpha-rad'],         # 8: AoA (Hücum Açısı - Çok Kritik!)
                self.fdm['aero/beta-rad']           # 9: Sideslip (Kayma Açısı)
            ], dtype=np.float32)
            
            return obs
    


# ORTAM ÇALIŞABİLİRLİK TESTİ
if __name__ == "__main__":
    print("JSBSim Ortamı Başlatılıyor...")
    
    env = FighterEnv()
    
    # Ortamı sıfırla (Uçağı 15.000 feet'e koy)
    obs, info = env.reset()
    print(f"Başlangıç Durumu -> İrtifa: {info['altitude']:.1f} ft, Mach: {info['mach']:.2f}")
    
    # Uçağa deneme için 100 adım boyunca rastgele komutlar verelim
    print("\n--- RASTGELE UÇUŞ TESTİ BAŞLIYOR ---")
    for step_idx in range(1, 101):
        # Ajanımız [İleri Hız, Dönüş Hızı, Tırmanma Hızı] için -1 ile +1 arası rastgele tuşlara basıyor
        random_action = env.action_space.sample() 
        
        # Fizik motorunu 1 adım ilerlet
        obs, reward, terminated, truncated, info = env.step(random_action)
        
        # Saniyede 1 kere (Her 60 adımda bir) ekrana rapor bas
        if step_idx % 20 == 0:
            print(f"Adım {step_idx} | "
                  f"İrtifa: {info['altitude']:.1f} ft | "
                  f"Mach: {info['mach']:.2f} | "
                  f"Pitch Açısı: {info['pitch_deg']:.1f}°")
            
        if terminated or truncated:
            print(f"UÇAK DÜŞTÜ VEYA SINIR AŞILDI! Adım: {step_idx}")
            break
            
    print("Sağlamlık Testi (Sanity Check) Başarıyla Tamamlandı!")