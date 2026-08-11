import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import math
import sys
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR) # insert(0, ...) komutu, bu klasörü en yüksek önceliğe alır!


from envs.f16_env import F16Env
from models.sac_actor import SafeActor
from utils.dmd_solver import RealTimeDMDc
from core.weapons import AMRAAM
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

class BVRCombatEnv(gym.Env):
    def __init__(self, env_config=None):
        super(BVRCombatEnv, self).__init__()
        
        # MÜFREDAT SEVİYESİ (Başlangıçta 1. Kur)
        self.curriculum_phase = 1

        # 1. ALT KATMAN ÇEVRESİ VE NORMALİZASYON (FAZ 3)
        self.base_env = F16Env()
        self.dummy_env = DummyVecEnv([lambda: self.base_env])
        
        vec_path = "./fighter_checkpoints/phase3_cbf/sac_env_phase3_final_vec_normalize.pkl"
        if os.path.exists(vec_path):
            self.norm_env = VecNormalize.load(vec_path, self.dummy_env)
            self.norm_env.training = False     
            self.norm_env.norm_reward = False  
        else:
            raise FileNotFoundError(f"Faz 3 Normalizasyon dosyası bulunamadı!")

        # 2. ALT BEYİN (FAZ 3 OTOPİLOTU) VE KALKAN (DMD)
        self.actor = SafeActor(state_dim=14, action_dim=3, num_constraints=4)
        actor_path = "./fighter_checkpoints/phase3_cbf/sac_actor_phase3_final.pth"
        self.actor.load_state_dict(torch.load(actor_path))
        self.actor.eval() # Alt beyni dondur
        
        self.dmd = RealTimeDMDc(state_dim=14, action_dim=3, window_size=60)
        self.dmd_update_interval = 6

        # CBF kısıtlarını (ham birimler) normalize uzaya dönüştürmek için Faz 3
        # normalizasyon parametreleri (z-score). Çünkü kalkan normalize state üzerinde çalışır:
        # x_raw = std * x_norm + mean -> kısıt C·x_raw <= d  <=>  (C/std)·x_norm <= d - C·mean
        obs_mean_np = np.asarray(self.norm_env.obs_rms.mean, dtype=np.float32)
        obs_var_np = np.asarray(self.norm_env.obs_rms.var, dtype=np.float32)
        self.obs_mean = obs_mean_np
        self.obs_std = np.sqrt(obs_var_np + 1e-8)

        # 3. SİLAH SİSTEMLERİ (SADECE 4 FÜZE!)
        self.max_missiles = 4
        self.missiles_left = self.max_missiles
        self.active_missile = AMRAAM()

        # 4. ÜST BEYİN (PPO) GÖZLEM VE AKSİYON UZAYLARI
        # Obs: [Mesafe, ATA, AA, Mach, İrtifa, RWR_İkazı, Kalan_Füze] -> 7 Veri
        self.observation_space = spaces.Box(
            low=np.array([0.0, -180.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            high=np.array([100000.0, 180.0, 180.0, 2.5, 60000.0, 1.0, 4.0]),
            dtype=np.float32
        )
        # Action: [Hedef_Pitch, Hedef_Roll, Füze_Ateşle]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        
        self.bandit = {"lat": 0.0, "lon": 0.0, "alt_m": 0.0, "heading_rad": 0.0}

    def reset(self, *, seed=None, options=None):
        self.current_step = 0
        self.low_level_state = self.norm_env.reset()
        
        # Düşmanı Yarat
        fdm = self.base_env.sim
        base_lat = fdm.get_property_value('position/lat-gc-deg')
        base_lon = fdm.get_property_value('position/long-gc-deg')
        self.bandit = {
            "lat": base_lat + np.random.uniform(0.15, 0.30), 
            "lon": base_lon + np.random.uniform(-0.15, 0.15),
            "alt_m": np.random.uniform(5000, 10000), 
            "heading_rad": np.random.uniform(0, 2 * math.pi) 
        }
        
        # Cephaneyi Doldur
        self.missiles_left = self.max_missiles
        self.active_missile.active = False
        self.dmd = RealTimeDMDc(state_dim=14, action_dim=3, window_size=60)

        
        for _ in range(60):
                    # 1. DURUM KALKANI: Ekrana gelen state bozuksa sıfırla
                    clean_state = np.nan_to_num(self.low_level_state, nan=0.0, posinf=10.0, neginf=-10.0)
                    state_tensor = torch.FloatTensor(clean_state)
                    
                    with torch.no_grad():
                        out = self.actor.net(state_tensor) 
                        
                        # 2. AKSİYON KALKANI: Ağdan NaN veya Sonsuzluk gelirse 0'a çevir!
                        raw_action = out[0].detach().numpy()
                        clean_action = np.nan_to_num(raw_action, nan=0.0, posinf=1.0, neginf=-1.0)
                        
                        action_numpy = np.clip(clean_action, -1.0, 1.0)
                        
                    batched_action = np.array([action_numpy]) 
                    next_state, _, _, _ = self.norm_env.step(batched_action)
                    self.dmd.add_data(clean_state[0], action_numpy, next_state[0])
                    self.low_level_state = next_state
            
        return self._get_obs(), {}

    def step(self, action):
        self.current_step += 1
        action = np.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0)
        action = np.clip(action, -1.0, 1.0)
        
        # 0.35 radyan yaklaşık 20 derece, 1.05 radyan yaklaşık 60 derecedir.
        komutan_pitch_rad = action[0] * 0.35  
        komutan_roll_rad = action[1] * 1.05

        # =================================================================
        # 1. RADAR VE GEOMETRİ HESABI
        # =================================================================
        mesafe, ata, aa = self._calculate_geometry()
        reward = -0.1 # Standart uçuş zaman cezası

        # =================================================================
        # 2. TAKTİKSEL İZ SÜRME ÖDÜLLERİ (PUAN ÇİFTÇİLİĞİ ENGELLENDİ!)
        # =================================================================
        if ata < 20.0:
            # Hedefe bakıyorsa sadece zaman cezasını nötrler (Net Kazanç: 0 olur).
            # Böylece oyunu uzatarak zengin olamaz! Puan kazanmak için vurmak zorundadır.
            reward += 0.1 
        else:
            reward -= 0.1 # Burnunu hedeften çevirirse ekstra ceza yer (Net Ceza: -0.2)

        # =================================================================
        # 3. AKILLI EMNİYET KİLİDİ VE "TEREDDÜT" CEZASI
        # =================================================================
        fuze_atesle_emri = False
        
        if action[2] > 0.8: # AJAN TETİĞE BASIYORSA:
            if mesafe < 15000.0 and ata < 45.0:
                fuze_atesle_emri = True # Menzil doğru, atış serbest!
            else:
                reward -= 0.5 # Yanlış yerde (uzaktan) tetiğe basma cezası!
                
        else: # AJAN TETİĞE BASMIYORSA (PASİFİST MOD):
            if mesafe < 15000.0 and ata < 20.0:
                # Düşman tam önünde ve menzildeyken tetiğe basmazsa büyük ceza yer!
                reward -= 0.5 # "Fırsat varken neden vurmuyorsun!" cezası.

        # =================================================================
        # 4. SİLAH ATEŞLEME MANTIĞI
        # =================================================================
        if fuze_atesle_emri and self.missiles_left > 0 and not self.active_missile.active and self.current_step > 1:
            fdm = self.base_env.sim
            bizim_pos = [0.0, 0.0, fdm.get_property_value('position/h-sl-ft') * 0.3048]
            self.active_missile.fire(bizim_pos)
            self.missiles_left -= 1
            print(f"[KOMUTAN] FOX-3! Kalan Füze: {self.missiles_left}")

        terminated = False

        # --- MACRO-STEP (1 SANİYELİK ALT BEYİN SİMÜLASYONU) ---
        for i in range(60):
            # 1. Kalkanı Güncelle
            if i % self.dmd_update_interval == 0:
                A_numpy, B_numpy = self.dmd.compute_matrices()
                if A_numpy is None or B_numpy is None:
                    # Tampon daha dolmadı (60 veri yok). Kalkan bu adımda pasif kalır,
                    # ham aksiyon döner; sistem çökmez, tampon dolunca kendiliğinden aktifleşir.
                    dynamic_limit_A = None
                    dynamic_limit_B = None
                else:
                    dynamic_limit_A = torch.FloatTensor(A_numpy)
                    dynamic_limit_B = torch.FloatTensor(B_numpy)


            # 1. DÜŞMANI FİZİKSEL OLARAK İLERLET (Mach 0.75 Hızında)
            hiz_m_s = 250.0 
            dt = 1.0 / 60.0
            ilerleme_m = hiz_m_s * dt

            R_EARTH = 6371000.0
            dy = ilerleme_m * math.cos(self.bandit["heading_rad"])
            dx = ilerleme_m * math.sin(self.bandit["heading_rad"])
            
            self.bandit["lat"] += math.degrees(dy / R_EARTH)
            self.bandit["lon"] += math.degrees(dx / (R_EARTH * math.cos(math.radians(self.bandit["lat"]))))

            # ====================================================
            # 2. AKILLI RAKİP ZEKASI (REAKTİF VE TAKTİKSEL)
            # ====================================================
            # Durumsal Farkındalık: F-16'nın nerede olduğunu ve füze gelip gelmediğini anla!
            mesafe, ata_deg, aa_deg = self._calculate_geometry()
            tehdit_var_mi = self.active_missile.active

            if getattr(self, "curriculum_phase", 1) == 1:
                # LEVEL 1: Hedef Uçuşu (Sadece düz uçar, ajan nişan almayı öğrenir)
                pass 
                
            elif getattr(self, "curriculum_phase", 1) == 2:
                # LEVEL 2: Defansif Kaçış (Crank / Kinetik Yenilgi)
                if tehdit_var_mi or mesafe < 25000.0:
                    # Üzerine füze geliyorsa veya F-16 çok yakınsa füzenin enerjisini bitirmek için makaslama yapar
                    self.bandit["heading_rad"] += 0.04 * np.sign(math.sin(i * 0.15))
                else:
                    self.bandit["heading_rad"] += math.sin(i * 0.05) * 0.01 # Sakin devriye
                    
            elif getattr(self, "curriculum_phase", 1) >= 3:
                # LEVEL 3: Ölümcül İt Dalaşı (Avcı ve Kaçak Modu)
                if tehdit_var_mi:
                    # 1. KAÇIŞ MODU: Füze atıldıysa hayatını kurtarmak için çok sert manevra yapar
                    self.bandit["heading_rad"] += 0.06 * np.sign(math.sin(i * 0.2))
                elif mesafe > 15000.0:
                    # 2. AVCI MODU: Füze yoksa, burnunu tam olarak F-16'ya (Bize) çevirir!
                    fdm = self.base_env.sim
                    bizim_lat = fdm.get_property_value('position/lat-gc-deg')
                    bizim_lon = fdm.get_property_value('position/long-gc-deg')
                    
                    # F-16'nın açısını (kerterizini) bul
                    hedef_aci = math.atan2(bizim_lon - self.bandit["lon"], bizim_lat - self.bandit["lat"])
                    
                    # En kısa yoldan burnunu F-16'ya çevir
                    fark = (hedef_aci - self.bandit["heading_rad"] + math.pi) % (2 * math.pi) - math.pi
                    self.bandit["heading_rad"] += max(-0.02, min(0.02, fark)) # Ölümcül dönüş hızı
                else:
                    # 3. YAKIN MUHAREBE (Dogfight): Kargaşa yaratır
                    self.bandit["heading_rad"] += math.sin(i * 0.3) * 0.05

                # RADAR İKAZ ALICISI (RWR) SİMÜLASYONU
                # Eğer düşman sana dönükse (Aspect Angle düşükse) ve 35km'den yakınsa uçağında alarm öter!
                if mesafe < 35000.0 and aa_deg < 45.0 and not tehdit_var_mi:
                    self.rwr_warning = 1.0 
                else:
                    self.rwr_warning = 0.0
            # ====================================================

            # 2. Komutanın (PPO) hedeflerini Omuriliğe (SAC) şırınga et
            raw_obs = self.norm_env.get_original_obs().copy()
            raw_obs[0][12] = komutan_pitch_rad
            raw_obs[0][13] = komutan_roll_rad
            self.low_level_state = self.norm_env.normalize_obs(raw_obs)

            # 3. Omurilik Kararı (Güvenli Uçuş)
            state_tensor = torch.FloatTensor(self.low_level_state)
            
            with torch.no_grad(): 
                # C_safe ve d_safe matrislerini tanımlıyoruz
                # 1 Batch, 4 Kısıt, 14 State
                C_safe_np = np.zeros((1, 4, 14), dtype=np.float32)
                
                # C_safe matrisi SADECE yönleri (1 ve -1) tutmalı.
                C_safe_np[0, 0, 3] = 1.0  # +Pitch
                C_safe_np[0, 1, 3] = -1.0 # -Pitch
                C_safe_np[0, 2, 2] = 1.0  # +Roll
                C_safe_np[0, 3, 2] = -1.0 # -Roll
                
                # d_safe GERÇEK normalize edilmiş sınırları tutmalı: (d_raw - mean) / std
                d_safe_np = np.array([[
                    (0.52 - self.obs_mean[3]) / self.obs_std[3],
                    (0.52 + self.obs_mean[3]) / self.obs_std[3],
                    (1.05 - self.obs_mean[2]) / self.obs_std[2],
                    (1.05 + self.obs_mean[2]) / self.obs_std[2]
                ]], dtype=np.float32)
                
                C_safe_tensor = torch.FloatTensor(C_safe_np)
                d_safe_tensor = torch.FloatTensor(d_safe_np)

                # Matrisleri gerçek kalkanlı beynin içine yolla!
                out = self.actor(state_tensor, dynamic_limit_A, dynamic_limit_B, C_safe_tensor, d_safe_tensor)
                
                raw_action = out[0].detach().cpu().numpy().flatten()
                action_numpy = np.clip(raw_action, -1.0, 1.0)
                
            # VecEnv'in hata vermemesi için aksiyonu (1, 3) formatına paketliyoruz
            batched_action = np.array([action_numpy])

            # 4. JSBSim Motorunu Akıt
            next_state, _, low_level_done, _ = self.norm_env.step(batched_action)
            self.dmd.add_data(self.low_level_state[0], action_numpy, next_state[0])
            self.low_level_state = next_state

            # Uçak yere çarparsa veya kalkanı kırıp stall olursa (Ölüm)
            if low_level_done[0]:
                reward -= 100.0
                terminated = True
                print("[SİSTEM] Uçak Kırım Geçirdi! Savaş Kaybedildi.")
                break


            # 5. Füze Havadaysa Onu Simüle Et
            if self.active_missile.active:
                
                # Füzeye hedef pozisyonunu VE hedefin Aspect Angle (AA) açısını veriyoruz!
                mesafe, ata, aa = self._calculate_geometry()
                hedef_pos = [mesafe, 0.0, self.bandit["alt_m"]]
                
                missile_status = self.active_missile.step(hedef_pos, target_aa_deg=aa)
                
                # --- TACVIEW İÇİN ANLIK FÜZE KONUMU ---
                try:
                    fdm = self.base_env.sim
                    curr_lat = fdm.get_property_value('position/lat-gc-deg')
                    curr_lon = fdm.get_property_value('position/long-gc-deg')
                    m_x, m_y, m_z = self.active_missile.position
                    m_lat = curr_lat + (m_y / 111320.0)
                    m_lon = curr_lon + (m_x / (111320.0 * math.cos(math.radians(curr_lat))))
                    if hasattr(self, "acmi_file") and self.acmi_file:
                        self.acmi_file.write(f"301,T={m_lon}|{m_lat}|{m_z}|0|0|0,Type=Weapon+Missile,Name=AIM-120_AMRAAM,Color=Blue\n")
                except Exception:
                    pass
                # --------------------------------------

                if missile_status == "HIT":
                    reward += 100.0 
                    terminated = True
                    print("[SİSTEM] Hedef Vuruldu!")
                    break
                elif missile_status == "MISS_ENERGY":
                    reward -= 10.0 # Uzaktan attığı için ceza
                elif missile_status == "MISS_NOTCH":
                    reward -= 20.0 # Taktiksel Hata Cezası (Düşman dik açıyken füze israf etti!)
                    # Ajan savaşa devam eder, füzeleri bittiyse kaybeder.

        # EĞER FÜZELER BİTİRSE VE DÜŞMAN ÖLMEDİYSE, KAYBETTİK!
        # EĞER FÜZELER BİTİRSE VE DÜŞMAN ÖLMEDİYSE, KAYBETTİK!
        if self.missiles_left == 0 and not self.active_missile.active and not terminated:
            reward -= 50.0
            terminated = True
            print("[SİSTEM] Cephane Tükendi! (Winchester). Görev Başarısız.")

        # === YENİ EKLENEN KISIM: ZAMAN AŞIMI (TIMEOUT) ===
        truncated = False
        # Örneğin 400 makro-adım (yaklaşık 5-10 dakikalık uçuş) sınırı koyuyoruz
        if self.current_step >= 400:
            truncated = True
            print(f"[SİSTEM] Görev Süresi Doldu! Uçak üsse dönüyor. (Adım: {self.current_step})")

        # Ray'in bölümü çöpe atmasını engellemek için saf tiplere (float, bool) çeviriyoruz
        # DİKKAT: 4. argüman artık False değil, bool(truncated) oldu!
        return self._get_obs(), float(reward), bool(terminated), bool(truncated), {}
    
    def _calculate_geometry(self):
            """ JSBSim'in Dünya Koordinatlarını (Enlem/Boylam) Taktiksel Vektörlere Çeviren Kalp """
            fdm = self.base_env.sim
            
            # BİZİM UÇAK VERİLERİ (JSBSim mülklerinden çekiliyor)
            lat_a = fdm.get_property_value('position/lat-gc-deg')
            lon_a = fdm.get_property_value('position/long-gc-deg')
            alt_a_m = fdm.get_property_value('position/h-sl-ft') * 0.3048
            yaw_a = fdm.get_property_value('attitude/heading-true-rad')
            pitch_a = fdm.get_property_value('attitude/pitch-rad')

            # DÜŞMAN (BANDIT) VERİLERİ
            lat_b = self.bandit["lat"]
            lon_b = self.bandit["lon"]
            alt_b_m = self.bandit["alt_m"]
            yaw_b = self.bandit["heading_rad"]

            # 1. MESAFE VE GÖRÜŞ HATTI (LOS - Line of Sight) HESABI
            R_EARTH = 6371000.0 # metre cinsinden dünya yarıçapı
            d_lat = math.radians(lat_b - lat_a)
            d_lon = math.radians(lon_b - lon_a)

            # Kuzey(Y) ve Doğu(X) yönündeki mesafeleri metreye çeviriyoruz
            dx = d_lon * math.cos(math.radians(lat_a)) * R_EARTH
            dy = d_lat * R_EARTH
            dz = alt_b_m - alt_a_m

            # 3 Boyutlu Toplam Mesafe (Distance)
            distance = math.sqrt(dx**2 + dy**2 + dz**2)

            # Görüş Hattı Vektörü (Birim Vektör haline getiriliyor)
            los_vec = np.array([dx, dy, dz])
            los_unit = los_vec / (np.linalg.norm(los_vec) + 1e-8)

            # 2. BİZİM YÖN VEKTÖRÜMÜZ VE ATA (Antenna Train Angle) HESABI
            # Uçağımızın burnunun baktığı 3B yön vektörü
            hx_a = math.sin(yaw_a) * math.cos(pitch_a)
            hy_a = math.cos(yaw_a) * math.cos(pitch_a)
            hz_a = math.sin(pitch_a)
            heading_a_unit = np.array([hx_a, hy_a, hz_a])

            # ATA: Bizim burnumuz ile düşman arasındaki açı (Nokta Çarpımı ile)
            dot_ata = np.clip(np.dot(heading_a_unit, los_unit), -1.0, 1.0)
            ata_deg = math.degrees(math.acos(dot_ata))

            # 3. DÜŞMAN YÖN VEKTÖRÜ VE ASPECT ANGLE (AA) HESABI
            hx_b = math.sin(yaw_b)
            hy_b = math.cos(yaw_b)
            hz_b = 0.0 # Düşman şimdilik düz uçuyor
            heading_b_unit = np.array([hx_b, hy_b, hz_b])

            # AA: Düşman ekseni ile Gelen LOS arasındaki açı
            dot_aa = np.clip(np.dot(heading_b_unit, los_unit), -1.0, 1.0)
            aa_deg = math.degrees(math.acos(dot_aa))
            
            return distance, ata_deg, aa_deg

    def _get_obs(self):
        """ Üst Beynin (PPO) Gözüne Gidecek Olan Taktiksel Radar Verisi """
        
        # 1. Geometri motorunu çalıştırıp anlık durumları alıyoruz
        distance, ata_deg, aa_deg = self._calculate_geometry()
        
        # 2. Uçağımızın anlık telemetri/enerji verileri
        fdm = self.base_env.sim
        mach = fdm.get_property_value('velocities/mach')
        alt_ft = fdm.get_property_value('position/h-sl-ft')
        
        # 3. RWR (Füze İkazı) şimdilik 0 (İleri seviyelerde düşman ateş edince 1 olacak)
        rwr_warning = getattr(self, "rwr_warning", 0.0) 

        # 4. KRİTİK EKLEME: Kalan Füze Sayısı (self.missiles_left)
        kalan_fuze = float(self.missiles_left)

        # Ajanın Beynine giden 7 boyutlu nihai Radar Matrisi:
        # [Mesafe(m), ATA(°), AA(derece), Hız(Mach), İrtifa(ft), RWR_Uyarısı, Kalan_Füze_Sayısı]
        obs = np.array([distance, ata_deg, aa_deg, mach, alt_ft, rwr_warning, kalan_fuze], dtype=np.float32)
        
        return obs
    
    def set_phase(self, phase):
        """ Eğitim Gözetmeni (Callback) tarafından seviye atlatıldığında çağrılır """
        self.curriculum_phase = phase
        print(f" Çevre Seviyesi Güncellendi: LEVEL {self.curriculum_phase}")