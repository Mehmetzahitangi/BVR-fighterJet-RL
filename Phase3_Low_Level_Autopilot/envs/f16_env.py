import gymnasium as gym
from gymnasium import spaces
import numpy as np
import jsbsim  # JSBSim'in resmi Python kütüphanesi

class F16Env(gym.Env):
    """
    F-16 JSBSim Pekiştirmeli Öğrenme Çevresi (Phase 3 - BVR Low-Level Autopilot)
    """
    def __init__(self, jerk_coef=0.2):
        super(F16Env, self).__init__()

        # Manivela Hassasiyeti (Jerk) ceza katsayısı (duyarlılık testleri için dışarıdan verilebilir)
        self.jerk_coef = jerk_coef
        
        # --- 1. JSBSim Başlatma ---
        self.sim = jsbsim.FGFDMExec(None)
        self.sim.load_model('f16') # JSBSim içindeki F-16 aerodinamik modeli
        
        # Simülasyon Frekansı: 60 Hz (DMD ve CBF için belirlediğimiz hız)
        self.dt = 1.0 / 60.0
        self.sim.set_dt(self.dt)
        
        # --- 2. Eylem Uzayı (Action Space) ---
        # 3 Boyutlu Sürekli Uzay: [Elevator (Pitch), Aileron (Roll), Throttle (Hız)]
        # Değerler -1.0 ile 1.0 arasındadır. CBF bu değerleri filtreleyecek.
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        
        # BVR Hedef (Üst akıldan gelen referans komutlar, Faz 3'te sabit)
        self.target_pitch = 0.0
        self.target_roll = 0.0
        
        # Bölüm başlangıç irtifası (ödül formülü ilk reset'ten önce çalışmasın diye güvenlik değeri)
        self.start_alt = 30000.0
        # Bölüm başlangıç stall Mach'ı (hız cezası eşiği; reset() içinde güncellenir)
        self.stall_mach = 0.30 + (self.start_alt - 15000.0) / 30000.0 * 0.25
        # Manivela Hassasiyeti (Jerk) cezası için önceki aksiyon (PIO/bang-bang önleyici)
        self.last_action = None
        # Anti-Stall kurtarma baypası durumu (histerezis; hız toparlanana dek aktif kalır)
        self.anti_stall_active = False

        # --- 3. Gözlem Uzayı (Observation Space) ---
        # [İrtifa, Mach, Roll, Pitch, Yaw, p, q, r, Alpha, Beta, Özgül Enerji, YAKIT_ORANI]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(14,), dtype=np.float32)
        
        # F-16'nın yaklaşık maksimum dahili yakıt kapasitesi (lbs)
        self.MAX_FUEL_LBS = 7000.0

    def _apply_domain_randomization(self):
        """ Simülasyondan Gerçeğe (Sim2Real) için kaotik rüzgar ve ağırlık ekler """
        # 1. Rüzgar ve Türbülans (Rastgele)
        wind_dir = np.random.uniform(0, 360)
        wind_speed = np.random.uniform(0, 50) # 0-50 knot arası rüzgar
        self.sim.set_property_value('atmosphere/wind-mag-fps', wind_speed * 1.68)
        self.sim.set_property_value('atmosphere/wind-heading-deg', wind_dir)
        self.sim.set_property_value('atmosphere/turb-type', 4) # Türbülans aktif
        
        # 2. Ağırlık / Yakıt Kayması (CG Kayması)
        # Tanklardaki yakıtı rastgele %30 ile %100 arası doldur
        fuel_percentage = np.random.uniform(0.3, 1.0)
        self.sim.set_property_value('propulsion/tank[0]/contents-lbs', 3000 * fuel_percentage)
        self.sim.set_property_value('propulsion/tank[1]/contents-lbs', 3000 * fuel_percentage)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # BVR Hedefleri (Faz 3'te rastgele: CBF kalkan zarfıyla birebir aynı aralıklar!
        # Böylece alt beyin Faz 4'te üst aklın vereceği komutlara cevap vermeyi öğrenir)
        self.target_pitch = np.random.uniform(-0.52, 0.52)  # ±30 derece
        self.target_roll = np.random.uniform(-1.05, 1.05)   # ±60 derece
        
        # BVR Başlangıç Koşulları (İrtifayla İlişkili Rastgele Zarf)
        # İrtifa: 15.000 - 45.000 ft arası rastgele (Faz 4 bandına yakın)
        self.start_alt = np.random.uniform(15000.0, 45000.0)
        
        # Hız: irtifaya göre ilişkili zarf (her zaman stall üstünde, süpersonik segmentli!)
        #   - Stall Mach'ı irtifa ile yükselir: 15K -> 0.30, 30K -> ~0.43, 45K -> 0.55
        #   - Maksimum hız da irtifayla yükselir (yapısal/q limiti): 15K -> 1.00, 45K -> 1.80
        stall_mach = 0.30 + (self.start_alt - 15000.0) / 30000.0 * 0.25
        max_mach = 1.00 + (self.start_alt - 15000.0) / 30000.0 * 0.80
        self.stall_mach = stall_mach  # Hız cezası eşiği bölüm boyunca bu değere sabitlenir
        self.start_mach = np.random.uniform(stall_mach + 0.10, max_mach)
        
        self.sim.set_property_value('ic/h-sl-ft', self.start_alt)
        self.sim.set_property_value('ic/mach', self.start_mach)
        self.sim.set_property_value('ic/gamma-deg', 0)   # Düz Uçuş
        
        self._apply_domain_randomization()
        
        self.sim.run_ic() # Başlangıç koşullarını uygula
        self.sim.run()    # Motorları ve aerodinamiği uyandır
        
        # Jerk cezası bölüm içi süreklilik için geçerli (ilk adım cezasız)
        self.last_action = None
        self.anti_stall_active = False  # Anti-Stall baypası her bölümde temiz başlar
        
        return self._get_state(), {}

    def _get_state(self):
            """ JSBSim'den o anki uçuş verilerini çeker """
            h = self.sim.get_property_value('position/h-sl-ft')
            v_true_fps = self.sim.get_property_value('velocities/vt-fps')
            
            # 1. Özgül Enerji (Specific Energy)
            g = 32.174 
            specific_energy = h + ((v_true_fps ** 2) / (2 * g))
            
            # 2. YENİ EKLENTİ: Anlık Yakıt Fraksiyonu Hesaplama (Realizm)
            # JSBSim'de motor çalıştıkça bu tanklardaki yakıt otomatik olarak azalır!
            tank_0 = self.sim.get_property_value('propulsion/tank[0]/contents-lbs')
            tank_1 = self.sim.get_property_value('propulsion/tank[1]/contents-lbs')
            current_fuel = tank_0 + tank_1
            fuel_fraction = current_fuel / self.MAX_FUEL_LBS # 0.0 (Boş) ile 1.0 (Dolu) arası değer
            
            state = np.array([
                h,
                self.sim.get_property_value('velocities/mach'),
                self.sim.get_property_value('attitude/phi-rad'),
                self.sim.get_property_value('attitude/theta-rad'),
                self.sim.get_property_value('attitude/psi-rad'),
                self.sim.get_property_value('velocities/p-rad_sec'),
                self.sim.get_property_value('velocities/q-rad_sec'),
                self.sim.get_property_value('velocities/r-rad_sec'),
                self.sim.get_property_value('aero/alpha-rad'),
                self.sim.get_property_value('aero/beta-rad'),
                specific_energy,
                fuel_fraction,
                self.target_pitch, # Pitch 
                self.target_roll   # Roll 
        ], dtype=np.float32)
            
            return state

    def step(self, action):
        """ CBF'den filtrelenerek gelen GÜVENLİ aksiyonu simülatöre uygular """
        
        action = np.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0)
        action = np.clip(action, -1.0, 1.0)

        # Aksiyonları JSBSim değişkenlerine eşle
        elevator, aileron, throttle = action[0], action[1], action[2]
        
        # =================================================================
        # 🛡️ AUTO-GCAS (GROUND COLLISION AVOIDANCE SYSTEM) - HARD DECK
        # =================================================================
        # Komutları motora yollamadan önce mevcut irtifaya bakıyoruz
        current_alt_ft = self.sim.get_property_value('position/h-sl-ft')
        HARD_DECK_FT = 2500.0 # Uçağın altına çekilen güvenli irtifa sınırı
        
        auto_gcas_penalty = 0.0
        
        if current_alt_ft < HARD_DECK_FT:
            # 1. Kanatları ufka paralel hale getir (Roll = 0 kilidi)
            aileron = 0.0  
            
            # 2. Burnu şiddetle havaya dik (Pitch Up - JSBSim'de negatif elevator yukarı çeker)
            elevator = -1.0 
            
            # 3. Motoru kökle (Throttle Max)
            throttle = 1.0 
            
            # Ajana güvenlik sınırını ihlal ettiği için caydırıcı ceza ver
            auto_gcas_penalty = -5.0 
        # =================================================================

        # =================================================================
        # 🛡️ ANTI-STALL KURTARMA BAYPASI (ENERGY RECOVERY)
        # =================================================================
        # F-16 stall kurtarma standardı: burnu AŞAĞI eğ (irtifayı hıza çevir) + TAM GAZ.
        # Yukarı çekmek alpha'yı artırır ve stall'ı derinleştirir (testteki +79.8° şaha kalkış
        # enerjiyi öldüren yanlış kurtarmaydı). Hız kazanan tek şey yerçekimi takasıdır.
        # - GCAS önceliği: yalnızca alt > 3500 ft iken devreye girer (takas payı ~1000 ft);
        #   altında GCAS (burun yukarı) kazanır, yere bilerek dalış yapılmaz.
        # - Histerezis: mach < stall+0.08'de devreye gir, stall+0.15'e dönünce bırak
        #   (eşik kenarında salınımı/chattering'i önler).
        # - Not: JSBSim'de POZİTİF elevator burnu AŞAĞI çeker (GCAS'ın -1.0'ının tersi).
        anti_stall_penalty = 0.0
        if not (current_alt_ft < HARD_DECK_FT):
            mach_now = self.sim.get_property_value('velocities/mach')
            if self.anti_stall_active:
                if mach_now < self.stall_mach + 0.15:
                    self.anti_stall_active = True  # kurtarma sürüyor
                else:
                    self.anti_stall_active = False  # hız toparlandı, kontrolü bırak
            elif mach_now < self.stall_mach + 0.08 and current_alt_ft > 3500.0:
                self.anti_stall_active = True  # stall yaklaşıyor, kurtarma başlat
            if self.anti_stall_active:
                elevator = 1.0   # burun aşağı (dalış: irtifayı hıza çevir)
                aileron = 0.0    # kanatları ufka paralel tut (temiz dalış)
                throttle = 1.0   # motoru kökle
                anti_stall_penalty = -1.0  # enerji yönetimi dersi (GCAS'tan yumuşak)
        # =================================================================
        
        self.sim.set_property_value('fcs/elevator-cmd-norm', elevator)
        self.sim.set_property_value('fcs/aileron-cmd-norm', aileron)
        self.sim.set_property_value('fcs/throttle-cmd-norm', (throttle + 1.0) / 2.0) # 0 ile 1 arasına çek
        
        # Simülasyonu 1 adım (1/60 saniye) ilerlet
        self.sim.run()
        
        state = self._get_state()
        state = np.nan_to_num(state, nan=0.0, posinf=10000.0, neginf=-10000.0)
        
        # --- 4. ÖDÜL VE BİTİŞ DURUMU (Reward & Done) ---
        # Faz 3 alt katman ajanı uçmayı (pitch/roll hedefini tutturmayı) öğrenir
        pitch = state[3]
        roll = state[2]
        
        # Hedef açılara ne kadar yakınsa o kadar az ceza (MSE Loss mantığı)
        reward = -1.0 * ((pitch - self.target_pitch)**2 + (roll - self.target_roll)**2)
        
        # Enerji Koruması: hız stall eşiği + 0.12'nin altına düşerse ceza (planör/reward-hacking önleyici)
        # Eşik irtifayla yükselir (15K -> 0.42, 45K -> 0.67); ağırlık 2.5 (izleme cezasıyla rekabet eder)
        mach = state[1]
        reward -= 2.5 * max(0.0, (self.stall_mach + 0.12) - mach)**2
        
        # İrtifa Bandı: bölümün KENDİ başlangıç irtifasından uzaklaşma hafifçe cezalandırılır
        # (dolaşmayı/planör süzülmeyi önler; mutlak irtifayı değil)
        h = state[0]
        reward -= 0.05 * ((h - self.start_alt) / 10000.0)**2
        
        # Aşırı Pitch Cezası: hedef bandı ±30° (0.52 rad); 40° üstünde anlık (indirimsiz)
        # sinyal — öğrenilmiş ölüm dalışı davranışını kırar (80°'de ~0.73, 45°'de ~0.01/adım)
        reward -= 1.5 * max(0.0, abs(pitch) - 0.7)**2
        
        # Manivela Hassasiyeti (Jerk / Action-Rate) Cezası: levyeyi sertçe sarsmak yerine
        # yumuşak komutlar üretmeye zorlar (PIO/bang-bang salınımını törpüler)
        if self.last_action is not None:
            action_now = np.asarray(action, dtype=np.float32)
            reward -= self.jerk_coef * float(np.sum((action_now - self.last_action) ** 2))
        self.last_action = np.asarray(action, dtype=np.float32).copy()
        
        # Auto-GCAS ve Anti-Stall baypasları devreye girdiyse ceza puanlarını ekle
        reward += auto_gcas_penalty + anti_stall_penalty
        
        # Çakılma Kontrolü (Bariyer varken olmaması lazım ama yinede kontrol)
        done = False
        if state[0] <= 1000: # 1000 feet'in altına düştüyse (Kritik irtifa kaybı)
            reward -= 1000.0
            done = True
            
        return state, reward, done, False, {}