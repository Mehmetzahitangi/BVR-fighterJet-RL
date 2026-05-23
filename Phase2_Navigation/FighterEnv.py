import math
import random

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import jsbsim


class FighterEnv(gym.Env):
    """Faz 2: Seyrüsefer ve Komut Takibi Çevresi"""
    
    def __init__(self):
        super(FighterEnv, self).__init__
        
        print("JSBSim Faz 2 Komut Takibi Başlatılıyor...")
        self.fdm = jsbsim.FGFDMExec(None)
        self.fdm.load_model('f16')

        # 60 FPS, daha hızlı eğitim ve görsel ortamda senkron sağlar
        self.fdm.set_dt(1.0 / 60.0)

        # Ön ayarlar (yakıtın bitmesini vs engelliyoruz şimdilik)
        self.fdm['propulsion/set-running'] = -1

        # --- AKSİYON UZAYI --- FBW Komuta Modu, Faz 1 ile AYNI
        # Ajan 3 taktiksel karar verir: [Pitch Komutu, Roll Komutu, Gaz Komutu]
        # Bu değerler step() fonksiyonunda alt seviye kontrolcülerle (Aileron, Elevator, vb.) birleştirilecektir.
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        # --- GÖZLEM UZAYI --- YENİLENDİ
        # Ajanın göreceği 8 değişken için minimum ve maksimum sınırları belirliyoruz.
        # Sinir ağları (Neural Network) sonsuzlukla başa çıkamaz. Bu yüzden sınır (low/high) koyuyoruz.
        obs_low = np.array([
            0.0,        # 1. Mevcut İrtifa (Min 0 ft - Yer)
            0.0,        # 2. Mevcut Hız (Min 0 Mach)
            0.0,        # 3. Hedef İrtifa (Min 0 ft)
            0.0,        # 4. Hedef Hız (Min 0 Mach)
            -50000.0,   # 5. Delta İrtifa (Max aşağı sapma)
            -2.0,       # 6. Delta Hız (Max yavaşlama sapması)
            -math.pi,   # 7. Roll (Min -180 derece)
            -math.pi/2  # 8. Pitch (Min -90 derece, tam dalış)
        ], dtype=np.float32)

        obs_high = np.array([
            50000.0,    # 1. Mevcut İrtifa (Max 50.000 ft - Uzay sınırı)
            2.0,        # 2. Mevcut Hız (Max Mach 2.0 - F16 son hızı)
            50000.0,    # 3. Hedef İrtifa (Max 50.000 ft)
            2.0,        # 4. Hedef Hız (Max Mach 2.0)
            50000.0,    # 5. Delta İrtifa (Max yukarı sapma)
            2.0,        # 6. Delta Hız (Max hızlanma sapması)
            math.pi,    # 7. Roll (Max +180 derece)
            math.pi/2   # 8. Pitch (Max +90 derece, tam tırmanış)
        ], dtype=np.float32)

        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
        
        # Bölüm içindeki dinamik hedef değişkenlerimiz (Başlangıçta sıfır)
        self.target_altitude = 0.0
        self.target_mach = 0.0
        self.current_step = 0  # Adım sayacı

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        #Fizik Motorunu Sıfırla ve Uçağı Havada Başlat (Airborne Initialization)
        self.fdm.reset_to_initial_conditions(0)
        self.fdm['ic/h-sl-ft'] = 15000.0   # 15.000 feet'te başla
        self.fdm['ic/vc-kts'] = 350.0      # Yaklaşık Mach 0.6 ile başla
        self.fdm['ic/gamma-deg'] = 0.0     # Burun düz
        self.fdm.run_ic()

        # Curriculum Learning: Ajana Rastgele Yeni Bir Görev Ver
        # Ajan her bölümde farklı bir irtifa ve hıza gitmeyi öğrenecek.
        # İrtifa ve hız limitlerinin tüm aralığını vermedik. 
        # Yüksek ve düşük feet, düşük ve yüksek hız arasındaki fiziksel farklılıkları tek bir seferde öğrenmeye çalışıp bozulmasın diye. Başlangıçta sadece bu aralıklarda eğitim yaptırıp sonradan limitleri yer ve uzaya göre genişleteceğiz.
        self.target_altitude = self.np_random.uniform(10000.0, 20000.0) # random.uniform(10000.0, 20000.0) # 10k ile 20k feet arası rastgele hedef
        self.target_mach = self.np_random.uniform(0.5, 0.9) # random.uniform(0.5, 0.9)             # Mach 0.5 ile 0.9 arası rastgele hız
        # random.uniform YERİNE self.np_random.uniform kullanıyoruz ortamımız Deterministik (tekrar edilebilir) olsun diye. random.uniform GYM'in seed sistemi ile uyumlu çalışmaz 
        
        # Simülasyonu 1 adım ilerletip ilk sensör verilerini alalım
        self.fdm.run()

        self.current_step = 0  # Her bölümde sıfırla
                
        # Gymnasium standardı gereği state ve info döner
        info = {"mevcut_irtifa": self.fdm['position/h-sl-ft'],
            "hedef_irtifa": self.target_altitude,
            "mevcut_mach": self.fdm['velocities/mach'],
            "hedef_mach": self.target_mach}

        return self._get_obs(), info
    
    def step(self, action):


            # FLY-BY-WIRE (FBW) ve KOORDİNELİ DÖNÜŞ (Aksiyon Çevirisi)
            # Ajanın -1 ile 1 arasındaki 3 kararını alıyoruz
            pitch_cmd = action[0]
            roll_cmd = action[1]
            throttle_cmd = action[2]

            # Elevatör (İrtifa) ve Aileron (Yatış) komutlarını doğrudan iletiyoruz
            self.fdm['fcs/elevator-cmd-norm'] = pitch_cmd
            self.fdm['fcs/aileron-cmd-norm'] = roll_cmd
            
            # KOORDİNELİ DÖNÜŞ MATEMATİĞİ (Adverse Yaw Engelleme)
            # Gerçek bir F-16'nın uçuş bilgisayarı gibi davranıyoruz.
            # Uçak yattığında (roll), havada kaymaması için rudder'ı (kuyruk dümeni) 
            # yatış yönüne orantılı olarak (%30) otomatik eziyoruz. 
            self.fdm['fcs/rudder-cmd-norm'] = roll_cmd * 0.3
            
            # Gaz kolunu -1/1 aralığından 0/1 (Rölanti/Tam Gaz) aralığına çekiyoruz
            self.fdm['fcs/throttle-cmd-norm'] = (throttle_cmd + 1.0) / 2.0

            # FİZİK MOTORUNU ÇALIŞTIR
            self.fdm.run()

            obs = self._get_obs()

            mevcut_irtifa = obs[0]
            mevcut_hiz = obs[1]
            delta_irtifa = obs[4]
            delta_hiz = obs[5]
            roll = obs[6]
            pitch = obs[7]


            # YENİ ÖDÜL MİMARİSİ (Faz 2: Hedef Takibi)
            reward = 0.0
            terminated = False # Oyun fiziksel ölümle mi bitti?
            truncated = False  # Zaman/Adım sınırı mı doldu?

            self.current_step += 1 # Sayacı artır

            # ÖLÜM SINIRLARI (Exploit Kırıcı)
            # İlk eğitimden sonra ajanın "intihar sendromunu" engellemek için İNTİHAR CEZASI artırıldı, hedef sınırları içerisine girdiğinde aldığı ödül artırıldı, yaşama cezası hafifletildi
            # Hız 0.4 Mach altına düşerse stall (tutunma kaybı) yaşanır ve ajan ölür, aynı ceza irtifa için de geçerli
            if mevcut_irtifa < 2000.0 or mevcut_irtifa > 50000.0 or (mevcut_hiz < 0.4 and self.current_step > 60): # Bölüm ilk başladığında uçağın hızı sınırımız altında kalabilir
                # İNTİHAR CEZASI ARTIRILDI (-100'den -5000'e)
                reward = -5000.0
                terminated = True
            else:
                # DURUŞ CEZASI (Stabilite için ufak bir ceza)
                # DURUŞ CEZASI (Ilımlı Disiplin - Karesel iptal edildi)
                reward -= (abs(roll) * 5.0) + (abs(pitch) * 0.1)

                # HEDEF ZARFI (Local Optima Sorununu Kırmak için Devasa Ödül)
                # İrtifada 2000 feet, hızda 0.1 Mach hata payı ile hedefe yaklaşırsa
                #if abs(delta_irtifa) < 2000.0 and abs(delta_hiz) < 0.3:
                if abs(delta_irtifa) < 1000.0 and abs(delta_hiz) < 0.2:
                    reward += 100.0  # Her saniye ödül yağmuru (Global Optima)
                else:
                    # Zarfın dışındaysa, hedefe uzaklığıyla orantılı yumuşak bir ceza
                    #reward -= (abs(delta_irtifa) / 25000.0) + (abs(delta_hiz) / 2)

                    # Cezaları çok güçlendirdik. 
                    # Artık hedeften uzak kalmak, fırıldaklık yapmak kadar canını yakacak
                    # 25000.0'ı 5000.0'a böldük (İrtifa cezası 5 kat arttı)
                    # Hız cezasını da daha çok ciddiye alması için çarpanı artırdık.
                    reward -= (abs(delta_irtifa) / 5000.0) + (abs(delta_hiz) * 5.0)

            info = {
                "mevcut_irtifa": mevcut_irtifa,
                "hedef_irtifa": self.target_altitude,
                "mevcut_mach": mevcut_hiz,
                "hedef_mach": self.target_mach
            }

            return obs, reward, terminated, truncated, info
    
    def _get_obs(self):
        """Sensörleri okuyup Gözlem Dizisini (Observation Array) standart bir şekilde döndürür."""
        mevcut_irtifa = self.fdm['position/h-sl-ft']
        mevcut_hiz = self.fdm['velocities/mach']
        roll = self.fdm['attitude/roll-rad']
        pitch = self.fdm['attitude/pitch-rad']

        delta_irtifa = self.target_altitude - mevcut_irtifa
        delta_hiz = self.target_mach - mevcut_hiz

        return np.array([
            mevcut_irtifa,
            mevcut_hiz,
            self.target_altitude,
            self.target_mach,
            delta_irtifa,
            delta_hiz,
            roll,
            pitch
        ], dtype=np.float32)
