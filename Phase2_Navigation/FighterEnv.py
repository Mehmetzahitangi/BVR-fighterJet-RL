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
        # Ajanın göreceği 11 değişken için minimum ve maksimum sınırları belirliyoruz.
        # Sinir ağları (Neural Network) sonsuzlukla başa çıkamaz. Bu yüzden sınır (low/high) koyuyoruz.
        obs_low = np.array([
            0.0,        # 1. Mevcut İrtifa (Min 0 ft - Yer)
            0.0,        # 2. Mevcut Hız (Min 0 Mach)
            0.0,        # 3. Hedef İrtifa (Min 0 ft)
            0.0,        # 4. Hedef Hız (Min 0 Mach)
            -50000.0,   # 5. Delta İrtifa (Max aşağı sapma)
            -2.0,       # 6. Delta Hız (Max yavaşlama sapması)
            -math.pi,   # 7. Roll (Min -180 derece)
            -math.pi/2,  # 8. Pitch (Min -90 derece, tam dalış)
            -2000.0,    # 9. VSI (Dikey Hız - Dalış)
            -math.pi/2, # 10. YENİ: Alpha (Hücum Açısı - AoA)
            -math.pi/2  # 11. YENİ: Beta (Yana Kayma Açısı)
        ], dtype=np.float32)

        obs_high = np.array([
            50000.0,    # 1. Mevcut İrtifa (Max 50.000 ft - Uzay sınırı)
            2.0,        # 2. Mevcut Hız (Max Mach 2.0 - F16 son hızı)
            50000.0,    # 3. Hedef İrtifa (Max 50.000 ft)
            2.0,        # 4. Hedef Hız (Max Mach 2.0)
            50000.0,    # 5. Delta İrtifa (Max yukarı sapma)
            2.0,        # 6. Delta Hız (Max hızlanma sapması)
            math.pi,    # 7. Roll (Max +180 derece)
            math.pi/2,   # 8. Pitch (Max +90 derece, tam tırmanış)
            2000.0,    # 9. VSI (Dikey Hız - Dalış)
            math.pi/2, # 10. YENİ: Alpha (Hücum Açısı - AoA)
            math.pi/2  # 11. YENİ: Beta (Yana Kayma Açısı)
        ], dtype=np.float32)

        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
        
        # Bölüm içindeki dinamik hedef değişkenlerimiz (Başlangıçta sıfır)
        self.target_altitude = 0.0
        self.target_mach = 0.0
        self.current_step = 0  # Adım sayacı

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Fizik Motorunu Sıfırla ve Uçağı Havada Başlat (Initialization)
        self.fdm.reset_to_initial_conditions(0)

        # RASTGELE İRTİFA SEÇİMİ 
        self.initial_altitude = self.np_random.uniform(2000.0, 40000.0)

        #  İRTİFAYA BAĞLI HIZ  
        min_initial_mach, max_initial_mach = self.get_mach_limits(self.initial_altitude)
        self.initial_mach = self.np_random.uniform(min_initial_mach, max_initial_mach)

        # HEDEF İRTİFA (Maks +/- 8000 ft Tırmanış/Dalış) 
        # Hedefi ilk olarak +/- 2000 feet arasında seçerek eğitim yapıyoruz. Ajan bunu öğrendikten sonra aynı süreci +/- 5000 feet arasında yaparak daha zorlu hedeflere geçiyoruz. 
        # Son olarak da tam +/- 8000 feet'e kadar hedef seçerek ajanı gerçek sınırlarına kadar zorluyoruz.
        target_delta_altitude = self.np_random.uniform(-8000.0, 8000.0)
        self.target_altitude = self.initial_altitude + target_delta_altitude
        self.target_altitude = max(2000.0, min(40000.0, self.target_altitude))



        # DİNAMİK GÖREV SÜRESİ HESAPLAMASI
        # Uçağa toparlanması ve hızı ayarlaması için banko 60 saniye avans veriyoruz.
        # İrtifa değiştirmesi gereken her 1000 feet için de ekstra 20 saniye ekliyoruz.
        ekstra_saniye = (abs(target_delta_altitude) / 1000.0) * 20.0
        toplam_saniye = 60.0 + ekstra_saniye
        
        # Saniyede 60 adım (FPS) işliyoruz
        self.max_steps = int(toplam_saniye * 60) 
        # Örneğin: 2000 ft gidecekse 90 saniye (5400 adım) süresi olur.
        # İleride 8000 ft hedefine geçtiğimizde 180 saniye (10800 adım) süresi olur.

        #  HEDEF İRTİFAYA BAĞLI HEDEF HIZ (Kritik) 
        # Ajanı 40 bin feet'e gönderip, orada Mach 0.4'te uç (Stall ol) diyemeyiz
        min_target_mach, max_target_mach = self.get_mach_limits(self.target_altitude)

        #safe_max_target = max_target_mach - 0.2

        # Ajanı çok zorlamamak için hedefi mevcut hızına yakın (+/- 0.1) tutalım 
        # Ama aerodinamik sınırların (min/max_target_mach) dışına asla çıkmasına izin vermiyoruz
        safe_min_target_mach = max(min_target_mach, self.initial_mach - 0.1)
        safe_max_target_mach = min(max_target_mach, self.initial_mach + 0.1)

        # KONTROL (NumPy çökmesin diye):
        # Çok ekstrem irtifa değişimlerinde alt sınır üst sınırı geçerse, eşitleriz.
        safe_max_target_mach = max(safe_min_target_mach, safe_max_target_mach)

        self.target_mach = self.np_random.uniform(safe_min_target_mach, safe_max_target_mach)

        self.fdm['ic/h-sl-ft'] = self.initial_altitude 
        self.fdm['ic/mach'] = self.initial_mach   
        self.fdm['ic/gamma-deg'] = 0.0     # Burun düz
        self.fdm.run_ic()  

        # MOTORLARI ATEŞLE VE İNİŞ TAKIMLARINI TOPLA (Kritik)
        self.fdm['propulsion/set-running'] = -1 
        self.fdm['gear/gear-cmd-norm'] = 0.0

        # Simülasyonu 1 adım ilerletip ilk sensör verilerini alalım
        self.fdm.run()

        self.current_step = 0  # Her bölümde sıfırla
        self.prev_dist = None  # Ödül şekillendirme için mesafe takibi
                
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
            
            # KOORDİNELİ DÖNÜŞ MATEMATİĞİ (Adverse Yaw Engelleme). Gerçek bir F-16'nın uçuş bilgisayarı gibi davranması için.
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


            reward = 0.0
            terminated = False # Oyun fiziksel ölümle mi bitti?
            truncated = False  # Zaman/Adım sınırı mı doldu?

            self.current_step += 1 # Sayacı artır

            """ # ÖLÜM SINIRLARI (Exploit Kırıcı)
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
                    reward -= (abs(delta_irtifa) / 5000.0) + (abs(delta_hiz) * 5.0) """
            
            # 1. DİNAMİK STALL HESAPLAMASI (Ölüm Sınırları İçin)
            # Uçağın anlık irtifasındaki gerçek Stall hızını öğrenelim
            anlik_min_hiz, _ = self.get_mach_limits(mevcut_irtifa)

            # Eğer uçak dinamik stall sınırının altına düşerse (çok ufak bir toleransla)
            is_stall = (mevcut_hiz < anlik_min_hiz)

            # ÖLÜM SINIRLARI (Exploit Kırıcı & Gerçekçi)
            # İrtifa limitleri 1500 (yere çakılma payı) ve 45000 (uzaya çıkma payı) olarak esnetildi.
            # Ajanı dinamik stall yüzünden anında ÖLDÜRMÜYORUZ. 
            # Sadece hız 0.25'in altına düşerse öldürüyoruz
            if mevcut_irtifa < 1500.0 or mevcut_irtifa > 45000.0 or mevcut_hiz < 0.25: #(is_stall and self.current_step > 150):
                # İNTİHAR CEZASI ARTIRILDI (-7000'e)
                reward = -7000.0
                terminated = True
            else:

                # YUMUŞAK STALL CEZASI (Öldürme, Cezalandır)
                if is_stall:
                    # Ajan o irtifanın stall sınırındaysa, her adımda canı yanar.
                    # Bu sayede ölmez ama "Gaz açmam lazım" demeyi öğrenir.
                    reward -= 20.0



                # 2. DİSİPLİN (Ilımlı Duruş ve YENİ VSI - Dikey Hız Cezası)
                # YENİ: VSI (Dikey Hız) Cezası. Uçak roket gibi fırlamasın veya taş gibi dalmasın
                # obs dizisindeki 8. indeks 'dikey_hiz'ın cezasıdır. 
                reward -= abs(obs[8]) * 0.01

                # Fırıldaklık yapmasını ve burnunu gereksiz dikmesini engeller.
                # GERÇEKÇİ ORYANTASYON CEZASI (Split-S'e izin verir, sürekli ters uçmayı engeller)
                # Karesel değil doğrusal ceza. Uçak anlık ters dönebilir ama düz uçuş her zaman daha kârlıdır.
                reward -=  (abs(roll) * 1.5) + (abs(pitch) * 0.5)
                #reward -= (abs(roll) * 5.0) + (abs(pitch) * 0.1)

                # 3. DİNAMİK HUNİ (Esnek Cezalar)
                # İrtifa cezası aynen kalıyor (Hedefe güçlü çekim kuvveti)
                reward -= (abs(delta_irtifa) / 5000.0)

                # ENERJİ TAKASI İZNİ (Kritik): Hız cezası 5.0'dan 1.5'e düşürüldü.
                # Artık tırmanırken hız kaybederse panikleyip dalışa geçmeyecek
                reward -= (abs(delta_hiz) * 1.2)


                #  POTANSİYEL TEMELLİ ÖDÜL (Durgunluk Kırıcı) 
                # Mevcut mesafe hesaplama
                current_dist = abs(delta_irtifa) + (abs(delta_hiz) * 5000) 
                
                # Eğer mesafe azalıyorsa progress > 0 olur, artıyorsa progress < 0
                if self.prev_dist is not None:
                    progress = self.prev_dist - current_dist
                    reward += progress * 0.01  # İlerlemeyi teşvik ediyoruz (Çok büyük değil, küçük bir teşvik)
                
                self.prev_dist = current_dist

                # ZAMAN VE TEMBELLİK CEZASI (Living Penalty)
                # Ajan havada boş boş süzülmesin, görevi bir an önce bitirsin diye
                # her adımda (saniyenin 1/60'ında) ufak bir zaman vergisi kesiyoruz.
                # 60 adım = 1 saniye = -3.0 puan. (Ajan hızlı olmaya mecbur kalacak)
                reward -= 0.05

                # 4. KADEMELİ Curriculum Target - Local Optimal Kırıcı
                # Ajan 8000 ft uzaktan gelirken hevesini kaybetmesin diye 3 aşamalı ödül:
                if abs(delta_irtifa) < 2000.0:
                    # Aşama 1: Yaklaşma (Ufak ödül)
                    reward += 10.0 
                    
                if abs(delta_irtifa) < 1000.0 and abs(delta_hiz) < 0.2:
                    # Aşama 2: Seyrüsefer Zarfına Giriş (Büyük ödül)
                    reward += 50.0 
            

                if abs(delta_irtifa) < 500.0 and abs(delta_hiz) < 0.1:
                    # Aşama 3: İĞNE DELİĞİ (Kusursuz Trim - Devasa Ödül)
                    reward += 150.0

                    """Daha fazla geliştirmek istersek:
                    # Aşama 3: İĞNE DELİĞİ (Kusursuz Trim - Hassasiyet Çarpanı)
                    # Artık 500 feet'in içine girmek yetmiyorsa, merkeze inmeyi ne kadar iyi yaptığına göre ödül verelim.
                    
                    # 1. Merkeze Yakınlık Yüzdesi (0.0 ile 1.0 arası değer üretir)
                    # 500 ft'de %0 (0.0), 0 ft'de %100 (1.0) hassasiyet.
                    alt_precision = 1.0 - (abs(delta_irtifa) / 500.0) 
                    
                    # 0.1 Mach'ta %0 (0.0), 0 Mach'ta %100 (1.0) hassasiyet.
                    spd_precision = 1.0 - (abs(delta_hiz) / 0.1)      
                    
                    # 2. Ağırlıklı Hassasiyet Çarpanı
                    # İrtifayı tutturmak uçağın asıl görevi olduğu için İrtifaya %70, Hıza %30 önem veriyoruz.
                    precision_multiplier = (alt_precision * 0.7) + (spd_precision * 0.3)
                    
                    # 3. Dinamik Ödül Dağıtımı
                    # Merkeze tam oturursa +150 alır, sınırda gezinirse +5, +10 gibi komik rakamlar alır.
                    reward += 150.0 * precision_multiplier
                    """

            if self.current_step >= self.max_steps:
                truncated = True
                # Görevi bitiremediği için ciddi ama ölümcül olmayan bir Görev İptal Cezası yiyor
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
        """Sensörleri okuyup Gözlem Dizisini (Observation Array) standart bir şekilde döndürür."""
        mevcut_irtifa = self.fdm['position/h-sl-ft']
        mevcut_hiz = self.fdm['velocities/mach']
        roll = self.fdm['attitude/roll-rad']
        pitch = self.fdm['attitude/pitch-rad']
        dikey_hiz = self.fdm['velocities/h-dot-fps']

        # YENİ SENSÖRLER (JSBSim Aerodinamik Verileri)
        alpha = self.fdm['aero/alpha-rad'] # Hücum Açısı (Angle of Attack)
        beta = self.fdm['aero/beta-rad']   # Yana Kayma Açısı (Sideslip)

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
            pitch,
            dikey_hiz,
            alpha, 
            beta 
        ], dtype=np.float32)


    def get_mach_limits(self, irtifa):
        """
        Gerçek F-16 Uçuş Zarfı (Flight Envelope) Matematiği:
        İrtifa arttıkça minimum hız (Stall) artar, maksimum hız sınırı genişler.
        """
        # 0 ft -> Min Mach 0.3 | 40.000 ft -> Min Mach 0.65 (Hava inceldikçe stall hızı artar)
        min_mach = 0.3 + (irtifa / 40000.0) * 0.35 
        
        # 0 ft -> Max Mach 1.15 (Yapısal limit) | 40.000 ft -> Max Mach 1.6-2.0 (Aerodinamik limit)
        max_mach = 1.15 + (irtifa / 40000.0) * 0.65 
        
        return min_mach, max_mach