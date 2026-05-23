import os
import math
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from FighterEnv import FighterEnv

print("Uçuş Öncesi Kontroller (Pre-flight Checks) Yapılıyor...")

# 1. ÇEVRE VE ZIRHIN (İstatistiklerin) YÜKLENMESİ
# Ortamı kuruyoruz
env = DummyVecEnv([lambda: FighterEnv()])

# Eğitimin bittiği andaki Zırhı (.pkl) yüklüyoruz
norm_path = "./fighter_checkpoints/sac_f16_model_500000_steps_vec_normalize.pkl"
print(f"Normalizasyon verileri yükleniyor: {norm_path}")
norm_env = VecNormalize.load(norm_path, env)

# ÇOK KRİTİK: Ajan artık eğitimde değil, testte! 
# İstatistiklerin bozulmaması için güncellemeyi kilitliyoruz.
norm_env.training = False 
norm_env.norm_reward = False 

# 2. BEYNİN (Modelin) YÜKLENMESİ
model_path = "sac_fighter_phase1_rewardShaping_normalized.zip"
print(f"Şampiyon SAC Beyni yükleniyor: {model_path}")
model = SAC.load(model_path, env=norm_env, device="cuda")

# 3. TACVIEW (ACMI) KAYIT SİSTEMİNİN HAZIRLANMASI
acmi_filename = "f16_test_ucus.acmi"
acmi_file = open(acmi_filename, "w", encoding="utf-8")
# ACMI Başlık Bilgileri
acmi_file.write("FileType=text/acmi/tacview\n")
acmi_file.write("FileVersion=2.1\n")
acmi_file.write("0,ReferenceTime=2026-05-14T12:00:00Z\n") # Sanal uçuş saati

print("Motorlar Ateşlendi! F-16 Havalanıyor...")

obs = norm_env.reset()

# Uçuşu 2000 adım (JSBSim hızına göre yaklaşık 15-20 saniye) boyunca izleyelim
for step in range(2000):
    
    # RASTGELELİK YASAK: deterministic=True ile ajanın en iyi bildiği manevrayı yapmasını istiyoruz
    action, _states = model.predict(obs, deterministic=True)
    
    # Hareketi simülasyona uygula
    obs, reward, done, info = norm_env.step(action)
    
    # JSBSim'in fizik motoruna (fdm) direkt bağlantı kurarak ham verileri çekiyoruz
    fdm = norm_env.venv.envs[0].fdm
    
    lon = fdm['position/long-gc-deg']
    lat = fdm['position/lat-gc-deg']
    alt = fdm['position/h-sl-ft']
    roll = math.degrees(fdm['attitude/roll-rad'])
    pitch = math.degrees(fdm['attitude/pitch-rad'])
    yaw = math.degrees(fdm['attitude/heading-true-rad'])
    time_sec = fdm['simulation/sim-time-sec']

    # Tacview "Flight Level"'ı metre olarak alır, bu yüzden feet'i metreye çevirmemiz gerekiyor 
    alt_meters = alt * 0.3048
    
    # Uçağı Tacview'da F-16 olarak ve Mavi renkle (Dost) tanımlıyoruz (Sadece 1. adımda)
    if step == 0:
        acmi_file.write(f"101,T={lon}|{lat}|{alt_meters}|{roll}|{pitch}|{yaw},Type=Air+FixedWing,Name=F-16,Color=Blue\n")
    
    # Her adımın saniyesini ve 3 boyutlu pozisyonunu/açılarını kaydediyoruz
    acmi_file.write(f"#{time_sec:.3f}\n")
    acmi_file.write(f"101,T={lon}|{lat}|{alt_meters}|{roll}|{pitch}|{yaw}\n")
    
    if done:
        print(f"Uçuş {step}. adımda sonlandı! (Ya yere çarptı ya da uzaya çıktı)")
        break

acmi_file.close()
print(f"Uçuş başarıyla tamamlandı! Tacview kaydı oluşturuldu: {acmi_filename}")