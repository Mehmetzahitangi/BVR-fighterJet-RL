import os
import sys
import time
import math
import numpy as np

# 'core' modüllerini bulabilmesi için
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from FighterEnv import FighterEnv
from core.jsbsim_utils import setup_flightgear_xml

print("Test Ortamı Başlatılıyor...")


USE_FLIGHTGEAR = False  # Canlı 3D izlemek için True (FlightGear açık olmalı)
USE_TACVIEW = True     # Uçuş bittiğinde .acmi dosyası oluşturmak için True


TEST_STEP = 1800000  

MODEL_PATH = None #f"./fighter_checkpoints/phase2_advanced/sac_f16_phase2_Advanced_{TEST_STEP}_steps.zip"
VEC_PATH = None #f"./fighter_checkpoints/phase2_advanced/sac_f16_phase2_Advanced_{TEST_STEP}_steps_vec_normalize.pkl"
TACVIEW_FILE = f"phase2_flight_{TEST_STEP}_steps.acmi"


# Çevreyi kur ve eğer FlightGear aktifse XML bağla
env_raw = FighterEnv()
if USE_FLIGHTGEAR:
    xml_path = setup_flightgear_xml(port=5550, rate=60)
    env_raw.fdm.set_output_directive(xml_path)

env = DummyVecEnv([lambda: env_raw])

# Normalize istatistiklerini yükle ve GÜNCELLEMEYİ KAPAT (Test Modu)
if os.path.exists(VEC_PATH):
    norm_env = VecNormalize.load(VEC_PATH, env)
    norm_env.training = False
    norm_env.norm_reward = False
else:
    raise FileNotFoundError(f"Zırh dosyası bulunamadı: {VEC_PATH}")

# Ağırlıkları yükle
if os.path.exists(MODEL_PATH):
    model = SAC.load(MODEL_PATH, env=norm_env, device="cuda")
else:
    raise FileNotFoundError(f"Model dosyası bulunamadı: {MODEL_PATH}")


# TACVIEW (.ACMI) DOSYASININ HAZIRLANMASI
if USE_TACVIEW:
    acmi = open(TACVIEW_FILE, "w", encoding="utf-8")
    acmi.write("FileType=text/acmi/tacview\nFileVersion=2.1\n")
    acmi.write("0,ReferenceTime=2026-01-01T00:00:00Z\n")
    print(f"Tacview Kaydı Başladı: {TACVIEW_FILE}")


# TEST DÖNGÜSÜ

obs = norm_env.reset()
print(f"{TEST_STEP}. Adım Ajanı Kokpitte! Harekât Başlıyor...")

# Yaklaşık 1 dakikalık uçuş (60 FPS * 60 Saniye = 3600 adım)
for step in range(7200):
    # deterministic=True : Ajan zar atmaz, öğrendiği en iyi hamleyi yapar
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = norm_env.step(action)
    
    # Gerçek zamanlı JSBSim erişimi
    fdm = norm_env.venv.envs[0].fdm
    
    # Hedef verisini info sözlüğünden alıyoruz
    hedef_irtifa_ft = info[0]["hedef_irtifa"] 
    hedef_mach = info[0]["hedef_mach"]
    

    # TACVIEW VERİ YAZIMI

    if USE_TACVIEW:
        t = step * (1.0 / 60.0)
        lat = fdm['position/lat-gc-deg']
        lon = fdm['position/long-gc-deg']
        
        # Tacview metrik sistemi kullanıyor (Feet -> Metre)
        alt_m = fdm['position/h-sl-ft'] * 0.3048
        hedef_alt_m = hedef_irtifa_ft * 0.3048
        
        roll = fdm['attitude/roll-rad'] * (180.0 / math.pi)
        pitch = fdm['attitude/pitch-rad'] * (180.0 / math.pi)
        yaw = fdm['attitude/heading-true-rad'] * (180.0 / math.pi)
        
        acmi.write(f"#{t:.3f}\n")
        # Bizim Uçağımız (Mavi Renk)
        acmi.write(f"101,T={lon}|{lat}|{alt_m}|{roll}|{pitch}|{yaw},Type=Air+FixedWing,Name=F-16_AI,Color=Blue\n")
        # Sanal Hedef Noktamız (Kırmızı Renk, tam uçağın üstünde/altında süzülür)
        acmi.write(f"202,T={lon}|{lat}|{hedef_alt_m}|0|0|0,Type=Ground+Static,Name=HEDEF_{int(hedef_irtifa_ft)}ft,Color=Red\n")


    # FLIGHTGEAR CANLI GÖRÜNTÜSÜ İÇİN KISITLAMA
    if USE_FLIGHTGEAR:
        time.sleep(1.0 / 60.0)
        
    # Terminale canlı log bas
    if step % 30 == 0:
        mevcut_irtifa = fdm['position/h-sl-ft']
        mevcut_hiz = fdm['velocities/mach']
        print(f"Süre: {step/60:.1f}s | İrtifa: {mevcut_irtifa:.0f} -> Hedef: {hedef_irtifa_ft:.0f} | Hız: {mevcut_hiz:.2f}M -> Hedef: {hedef_mach:.2f}M")

    if done:
        print(f"Uçuş {step/60:.1f} saniye sonra sonlandı (Fiziksel Limit/Ölüm)")
        break

if USE_TACVIEW:
    acmi.close()
    
print("Uçuş Tamamlandı")