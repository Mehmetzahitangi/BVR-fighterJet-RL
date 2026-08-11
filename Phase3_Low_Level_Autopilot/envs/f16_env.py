import gymnasium as gym
from gymnasium import spaces
import numpy as np
import jsbsim  # JSBSim'in resmi Python kütüphanesi

class F16Env(gym.Env):
    """
    F-16 JSBSim Pekiştirmeli Öğrenme Çevresi (Phase 3 - BVR Low-Level Autopilot)
    """
    def __init__(self):
        super(F16Env, self).__init__()
        
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
        
        # BVR Başlangıç Koşulları (Yüksek İrtifa, Yüksek Hız)
        self.sim.set_property_value('ic/h-sl-ft', 30000) # 30.000 ft İrtifa
        self.sim.set_property_value('ic/mach', 0.8)      # Mach 0.8 Hız
        self.sim.set_property_value('ic/gamma-deg', 0)   # Düz Uçuş
        
        self._apply_domain_randomization()
        
        self.sim.run_ic() # Başlangıç koşullarını uygula
        self.sim.run()    # Motorları ve aerodinamiği uyandır
        
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
        
        # Aksiyonları JSBSim değişkenlerine eşle
        elevator, aileron, throttle = action[0], action[1], action[2]
        
        self.sim.set_property_value('fcs/elevator-cmd-norm', elevator)
        self.sim.set_property_value('fcs/aileron-cmd-norm', aileron)
        self.sim.set_property_value('fcs/throttle-cmd-norm', (throttle + 1.0) / 2.0) # 0 ile 1 arasına çek
        
        # Simülasyonu 1 adım (1/60 saniye) ilerlet
        self.sim.run()
        
        state = self._get_state()
        
        # --- 4. ÖDÜL VE BİTİŞ DURUMU (Reward & Done) ---
        # Faz 3 alt katman ajanı uçmayı (pitch/roll hedefini tutturmayı) öğrenir
        pitch = state[3]
        roll = state[2]
        
        # Hedef açılara ne kadar yakınsa o kadar az ceza (MSE Loss mantığı)
        reward = -1.0 * ((pitch - self.target_pitch)**2 + (roll - self.target_roll)**2)
        
        # Çakılma Kontrolü (Bariyer varken olmaması lazım ama yinede kontrol)
        done = False
        if state[0] <= 1000: # 1000 feet'in altına düştüyse (Kritik irtifa kaybı)
            reward -= 1000.0
            done = True
            
        return state, reward, done, False, {}