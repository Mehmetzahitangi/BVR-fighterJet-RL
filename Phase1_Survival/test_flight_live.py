import os
import time
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from FighterEnv import FighterEnv

print("FlightGear Kuruluyor...")

# FLIGHTGEAR(XML)
# JSBSim'e "Verilerini UDP ile 5550 portuna fırlat" talimatı veriyoruz
fg_xml = """<?xml version="1.0"?>
<output name="127.0.0.1" type="FLIGHTGEAR" port="5550" protocol="udp" rate="60">
    <position> ON </position>
    <attitude> ON </attitude>
    <velocities> ON </velocities>
</output>
"""
with open("fg_output.xml", "w") as f:
    f.write(fg_xml)

# ÇEVRE VE İSTATİSTİKLERİN YÜKLENMESİ
env = DummyVecEnv([lambda: FighterEnv()])
norm_path = "./fighter_checkpoints/sac_f16_model_500000_steps_vec_normalize.pkl"
norm_env = VecNormalize.load(norm_path, env)

# Ajan test modunda, yeni şeyler öğrenmesini engelliyoruz.
norm_env.training = False 
norm_env.norm_reward = False 


model_path = "sac_fighter_phase1_rewardShaping_normalized.zip"
model = SAC.load(model_path, env=norm_env, device="cuda")

obs = norm_env.reset()

# JSBSim'in içinde UDP yayınını aktif ediyoruz
fdm = norm_env.venv.envs[0].fdm
xml_absolute_path = os.path.abspath("fg_output.xml")
print(f"🔗 JSBSim şu XML köprüsüne bağlanıyor: {xml_absolute_path}")

# JSBSim'e tam yolu veriyoruz
fdm.set_output_directive(xml_absolute_path)

print("Uçuş başladı ekranı kontrol edebilirsiniz...")

# 2000 adımlık uçuş (Gerçek zamanda yaklaşık 30-35 saniye)
for step in range(2000):
    # Zar atmayı (rastgeleliği) yasaklıyoruz
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = norm_env.step(action)

    # Anlık irtifa ve yatış açılarını terminale yazdır
    #print(f"Adım: {step} | İrtifa: {alt:.1f} ft | Hız: {obs[1]:.2f} Mach | Yatış: {roll:.2f} derece", end="\r")
    
    # KRİTİK: ZAMAN BÜKÜLMESİ
    # Eğer bu satırı koymazsak uçuş 0.5 saniyede biter
    # Döngüyü 60 FPS (1/60 = 0.016 saniye) hızına sabitliyoruz.
    time.sleep(0.016)
    
    if done:
        print(f"Uçuş {step}. adımda sonlandı!")
        break

print("Uçuş başarıyla tamamlandı.")