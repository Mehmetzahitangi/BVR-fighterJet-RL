import os
import sys
import time
import math
import torch
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from envs.f16_env import F16Env
from models.sac_actor import SafeActor
from utils.dmd_solver import RealTimeDMDc

print("Faz 3 (Dinamik Görevli) Test Ortamı Başlatılıyor...")

USE_FLIGHTGEAR = False  
USE_TACVIEW = True      

MODEL_PATH = "./fighter_checkpoints/phase3_cbf/sac_actor_phase3_final.pth" 
VEC_PATH = "./fighter_checkpoints/phase3_cbf/sac_env_phase3_final_vec_normalize.pkl"
TACVIEW_FILE = "phase3_flight_dynamic_targets.acmi"

# 1. ÇEVREYİ KUR
base_env = F16Env()
monitored_env = Monitor(base_env)
env = DummyVecEnv([lambda: monitored_env])

# 2. NORMALİZASYON
if os.path.exists(VEC_PATH):
    norm_env = VecNormalize.load(VEC_PATH, env)
    norm_env.training = False     
    norm_env.norm_reward = False  
else:
    raise FileNotFoundError(f"Zırh dosyası bulunamadı: {VEC_PATH}")

# 3. AKTÖRÜ YÜKLE
actor = SafeActor(state_dim=14, action_dim=3, num_constraints=8)
if os.path.exists(MODEL_PATH):
    actor.load_state_dict(torch.load(MODEL_PATH))
    actor.eval() 
    print("Mükemmel Otopilot Beyni Yüklendi!")
else:
    raise FileNotFoundError(f"Model dosyası bulunamadı: {MODEL_PATH}")

# 4. KALKAN SİSTEMİ (DMD + CBF SINIRLARI)
dmd = RealTimeDMDc(state_dim=14, action_dim=3, window_size=300)
UPDATE_INTERVAL = 6

# CBF kısıtlarını (ham birimler) normalize uzaya dönüştür (z-score):
obs_mean_np = np.asarray(norm_env.obs_rms.mean, dtype=np.float32)
obs_var_np = np.asarray(norm_env.obs_rms.var, dtype=np.float32)
obs_std = np.sqrt(obs_var_np + 1e-8)

# Hibrit CBF: 4 açı kısıtı (pitch/roll bandı) + 4 açısal hız kısıtı (q/p bandı).
# Açılar tek adımda zayıf kontrol edilir; q/p hızları ise gerçek otoriteye sahiptir.
C_safe_np = np.zeros((1, 8, 14), dtype=np.float32)
C_safe_np[0, 0, 3] = 1.0   # +Pitch açısı
C_safe_np[0, 1, 3] = -1.0  # -Pitch açısı
C_safe_np[0, 2, 2] = 1.0   # +Roll açısı
C_safe_np[0, 3, 2] = -1.0  # -Roll açısı
C_safe_np[0, 4, 6] = 1.0   # +Pitch hızı q
C_safe_np[0, 5, 6] = -1.0  # -Pitch hızı q
C_safe_np[0, 6, 5] = 1.0   # +Roll hızı p
C_safe_np[0, 7, 5] = -1.0  # -Roll hızı p

d_safe_np = np.array([[
    (0.52 - obs_mean_np[3]) / obs_std[3],
    (0.52 + obs_mean_np[3]) / obs_std[3],
    (1.05 - obs_mean_np[2]) / obs_std[2],
    (1.05 + obs_mean_np[2]) / obs_std[2],
    (0.35 - obs_mean_np[6]) / obs_std[6],
    (0.35 + obs_mean_np[6]) / obs_std[6],
    (1.0 - obs_mean_np[5]) / obs_std[5],
    (1.0 + obs_mean_np[5]) / obs_std[5]
]], dtype=np.float32)

C_safe_tensor = torch.FloatTensor(C_safe_np)
d_safe_tensor = torch.FloatTensor(d_safe_np)

if USE_TACVIEW:
    acmi = open(TACVIEW_FILE, "w", encoding="utf-8")
    acmi.write("FileType=text/acmi/tacview\nFileVersion=2.1\n")
    acmi.write("0,ReferenceTime=2026-01-01T00:00:00Z\n")
    print(f"Tacview Kaydı Başladı: {TACVIEW_FILE}")

# ==========================================
# TEST DÖNGÜSÜ BAŞLIYOR
# ==========================================
state = norm_env.reset()

# --- SİHİRLİ DOKUNUŞ: İLK HEDEFİ 0.0 OLARAK ZORLA ---
raw_obs = norm_env.get_original_obs().copy()
raw_obs[0][12] = 0.0 # Target Pitch
raw_obs[0][13] = 0.0 # Target Roll
state = norm_env.normalize_obs(raw_obs)

print("\n[SİSTEM] Uçak Havalandı. Sensörler ve DMD Kalkanı Kalibre Ediliyor (5 Saniye)...")

# --- 5 SANİYELİK KALİBRASYON (300 ADIM) ---
for _ in range(300):
    state_tensor = torch.FloatTensor(state)
    with torch.no_grad():
        out = actor.net(state_tensor) 
        safe_action_tensor = out[0] if isinstance(out, tuple) else out
        action_numpy = safe_action_tensor.detach().numpy()
        action_numpy = np.clip(action_numpy, -1.0, 1.0)
        
    next_state, reward, done, info = norm_env.step(action_numpy)
    
    # Hedefi zorla 0.0 tut
    raw_obs = norm_env.get_original_obs().copy()
    raw_obs[0][12] = 0.0
    raw_obs[0][13] = 0.0
    next_state = norm_env.normalize_obs(raw_obs)

    dmd.add_data(state[0], action_numpy[0], next_state[0])
    state = next_state

# Kalkanı ateşle
A_numpy, B_numpy = dmd.compute_matrices()
if A_numpy is None or B_numpy is None:
    print("[SİSTEM] UYARI: DMD kalibrasyonu tamamlanamadı, kalkan pasif kalacak!")
    dynamic_limit_A = None
    dynamic_limit_B = None
else:
    dynamic_limit_A = torch.FloatTensor(A_numpy)
    dynamic_limit_B = torch.FloatTensor(B_numpy)
    print("[SİSTEM] DMD Kalkanı Aktif! Kontrol Tamamen Otopilota Devrediliyor...\n")


# --- ASIL DİNAMİK UÇUŞ (9000 ADIM / 150 SANİYE) ---
current_target_pitch_deg = 0.0
current_target_roll_deg = 0.0

for step in range(9000):
    t_sec = step / 60.0
    
    # 💥 FAZ 4 (ÜST BEYİN) SİMÜLASYONU: GÖREVLERİ DEĞİŞTİR 💥
    if step == 1200: # 20. Saniye
        print("\n🚨 [KOMUTAN EMRİ] Sağa Geniş Dönüş Yap! (Pitch: 5°, Roll: 35°)\n")
    elif step == 2700: # 45. Saniye
        print("\n🚨 [KOMUTAN EMRİ] Agresif Sola Kaçış! (Pitch: 10°, Roll: -45°)\n")
    elif step == 4200: # 70. Saniye
        print("\n🚨 [KOMUTAN EMRİ] Dalışa Geç! (Pitch: -10°, Roll: 0°)\n")
    elif step == 5700: # 95. Saniye
        print("\n🚨 [KOMUTAN EMRİ] Sert Tırmanış ve Sağa Yatış! (Pitch: 15°, Roll: 60°)\n")
    elif step == 7200: # 120. Saniye
        print("\n🚨 [KOMUTAN EMRİ] Görev İptal, Düz Uçuşa Dön! (Pitch: 0°, Roll: 0°)\n")

    # Zaman aralıklarına göre hedefleri belirle
    if 20.0 <= t_sec < 45.0:
        current_target_pitch_deg = 5.0
        current_target_roll_deg = 35.0
    elif 45.0 <= t_sec < 70.0:
        current_target_pitch_deg = 10.0
        current_target_roll_deg = -45.0
    elif 70.0 <= t_sec < 95.0:
        current_target_pitch_deg = -10.0
        current_target_roll_deg = 0.0
    elif 95.0 <= t_sec < 120.0:
        current_target_pitch_deg = 15.0
        current_target_roll_deg = 60.0
    elif t_sec >= 120.0:
        current_target_pitch_deg = 0.0
        current_target_roll_deg = 0.0

    # Kalkan Güncellemesi
    if step % UPDATE_INTERVAL == 0:
        A_numpy, B_numpy = dmd.compute_matrices()
        if A_numpy is None or B_numpy is None:
            dynamic_limit_A = None
            dynamic_limit_B = None
        else:
            dynamic_limit_A = torch.FloatTensor(A_numpy)
            dynamic_limit_B = torch.FloatTensor(B_numpy)

    state_tensor = torch.FloatTensor(state)
    
    # Ajan Kararı (Kalkanla birlikte)
    with torch.no_grad(): 
        if dynamic_limit_A is not None and dynamic_limit_B is not None:
            out = actor(state_tensor, dynamic_limit_A, dynamic_limit_B,
                        C_safe_tensor, d_safe_tensor)
        else:
            out = actor.net(state_tensor)
        safe_action_tensor = out[0] if isinstance(out, tuple) else out
        action_numpy = safe_action_tensor.detach().numpy()
        action_numpy = np.clip(action_numpy, -1.0, 1.0)

    # Simülasyon Adımı
    next_state, reward, done, info = norm_env.step(action_numpy)
    
    # 💥 YENİ HEDEFLERİ AJANIN BEYNİNE (OBSERVATION) ZORLA ENJEKTE ET 💥
    # Ajanın simülasyondan gelen rastgele hedefi görmesini engelliyoruz,
    # onun yerine kendi komutan hedeflerimizi yerleştirip normalize ediyoruz.
    raw_obs = norm_env.get_original_obs().copy()
    raw_obs[0][12] = math.radians(current_target_pitch_deg)
    raw_obs[0][13] = math.radians(current_target_roll_deg)
    next_state = norm_env.normalize_obs(raw_obs)

    dmd.add_data(state[0], action_numpy[0], next_state[0])

    # GERÇEK ZAMANLI TELEMETRİ OKUMALARI
    fdm = norm_env.venv.envs[0].env.sim 
    
    mevcut_pitch_rad = fdm.get_property_value('attitude/pitch-rad')
    mevcut_roll_rad = fdm.get_property_value('attitude/roll-rad')
    
    mevcut_pitch_deg = math.degrees(mevcut_pitch_rad)
    mevcut_roll_deg = math.degrees(mevcut_roll_rad)

    # TACVIEW YAZIMI
    if USE_TACVIEW:
        lat = fdm.get_property_value('position/lat-gc-deg')
        lon = fdm.get_property_value('position/long-gc-deg')
        alt_m = fdm.get_property_value('position/h-sl-ft') * 0.3048
        yaw = fdm.get_property_value('attitude/heading-true-rad') * (180.0 / math.pi)
        
        acmi.write(f"#{t_sec:.3f}\n")
        acmi.write(f"101,T={lon}|{lat}|{alt_m}|{mevcut_roll_deg}|{mevcut_pitch_deg}|{yaw},Type=Air+FixedWing,Name=F-16_Phase3_Dynamic,Color=Blue\n")
        
    if USE_FLIGHTGEAR:
        time.sleep(1.0 / 60.0)
        
    # EKRANA BİLGİ BAS (Her yarım saniyede bir)
    if step % 30 == 0:
        mach = fdm.get_property_value('velocities/mach')
        print(f"Süre: {t_sec:5.1f}s | Mach: {mach:.2f} | PITCH: {mevcut_pitch_deg:5.1f}° -> Hedef: {current_target_pitch_deg:5.1f}° | ROLL: {mevcut_roll_deg:5.1f}° -> Hedef: {current_target_roll_deg:5.1f}°")

    if done[0]:
        print(f"\nUçuş {t_sec:.1f} saniye sonra sınır ihlali (Stall/Kırım) sebebiyle sonlandı!")
        break
        
    state = next_state

if USE_TACVIEW:
    acmi.close()
    
print("\nDinamik Test Başarıyla Tamamlandı! Tacview (ACMI) dosyanız hazır.")
