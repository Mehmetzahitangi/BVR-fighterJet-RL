import os
import time
import math
import gymnasium as gym
import numpy as np
import torch 

from ray.rllib.algorithms.ppo import PPO
from ray.tune.registry import register_env

# 1. Kendi Çevrenizi Ray'e Tanıtın 
from phase4_envs.bvr_env_rllib import BVRCombatEnv
register_env("BVRCombatEnv_v0", lambda config: BVRCombatEnv(config))

# =========================================================================
# DİKKAT: V2 EĞİTİMİNDEKİ EN YENİ CHECKPOINT YOLUNU BURAYA YAPIŞTIRIN!
CHECKPOINT_YOLU = "C:\\Zahit\\BVR_fighterJet_AI\\Phase4_PPO_RLlib\\ray_results\\Phase4_BVR_Dogfight_V2\\PPO_BVRCombatEnv_v0_f00a7_00000_0_2026-08-08_19-12-22\\checkpoint_000196" 
TACVIEW_FILE = "phase4_dogfight_test_500K.acmi"
# =========================================================================

def main():
    print(f"\n[SİSTEM] Ajanın beyni yükleniyor: {CHECKPOINT_YOLU}")
    agent = PPO.from_checkpoint(CHECKPOINT_YOLU)
    rl_module = agent.get_module("default_policy")

    env = BVRCombatEnv() 
    
    if hasattr(env, "set_phase"):
        env.set_phase(3) 
        print("[SİSTEM] Çevre Seviyesi 3 (Ölümcül İt Dalaşı) Olarak Ayarlandı.")

    obs, _ = env.reset()
    done = False
    truncated = False
    toplam_puan = 0.0
    adim = 0

    # --- TACVIEW DOSYASINI BAŞLAT ---
    acmi = open(TACVIEW_FILE, "w", encoding="utf-8")
    acmi.write("FileType=text/acmi/tacview\nFileVersion=2.1\n")
    acmi.write("0,ReferenceTime=2026-01-01T00:00:00Z\n")
    print(f"\n[SİSTEM] Tacview Kaydı Başladı: {TACVIEW_FILE}")
    print("[SİSTEM] Ajan Kokpitte! Harekât Başlıyor...\n")

    while not (done or truncated):
        # 1. Ajan Kararı (Inference - Yeni RLModule API)
        obs_tensor = torch.tensor([obs], dtype=torch.float32)
        with torch.no_grad():
            fwd_out = rl_module.forward_inference({"obs": obs_tensor})
            if "action" in fwd_out:
                action = fwd_out["action"][0].cpu().numpy()
            elif "action_dist_inputs" in fwd_out:
                # Kesin (Deterministic) Karar: Ajan zar atmaz, bildiği en iyi hamleyi (mean) seçer
                action_dim = env.action_space.shape[0] 
                action = fwd_out["action_dist_inputs"][0, :action_dim].cpu().numpy()
            else:
                action = list(fwd_out.values())[0][0].cpu().numpy()
                
        # 2. Simülasyon Adımı
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated
        toplam_puan += reward
        adim += 1
        
        # Saniyeyi temsil eden makro-adım
        t_sec = adim 

        # 3. TACVIEW (ACMI) SİNEMATİK KAYIT
        try:
            fdm = env.base_env.sim
            lat = fdm.get_property_value('position/lat-gc-deg')
            lon = fdm.get_property_value('position/long-gc-deg')
            alt_m = fdm.get_property_value('position/h-sl-ft') * 0.3048
            yaw = fdm.get_property_value('attitude/heading-true-rad') * (180.0 / math.pi)
            pitch = fdm.get_property_value('attitude/pitch-rad') * (180.0 / math.pi)
            roll = fdm.get_property_value('attitude/roll-rad') * (180.0 / math.pi)

            b_lat = env.bandit["lat"]
            b_lon = env.bandit["lon"]
            b_alt = env.bandit["alt_m"]
            b_yaw = env.bandit["heading_rad"] * (180.0 / math.pi)

            acmi.write(f"#{t_sec:.3f}\n")
            # Mavi F-16 (Biz)
            acmi.write(f"101,T={lon}|{lat}|{alt_m}|{roll}|{pitch}|{yaw},Type=Air+FixedWing,Name=F-16_AI,Color=Blue\n")
            # Kırmızı MiG-29 (Düşman)
            acmi.write(f"201,T={b_lon}|{b_lat}|{b_alt}|0|0|{b_yaw},Type=Air+FixedWing,Name=MiG-29_Bandit,Color=Red\n")

            # 🚀 SARI AMRAAM FÜZESİ (Ateşlendiyse Çiz!)
            if hasattr(env, "active_missile") and env.active_missile.active:
                m_x, m_y, m_z = env.active_missile.position
                # Füzenin bağıl koordinatlarını Dünya koordinatlarına çevir
                m_lat = lat + (m_y / 111320.0)
                m_lon = lon + (m_x / (111320.0 * math.cos(math.radians(lat))))
                acmi.write(f"301,T={m_lon}|{m_lat}|{m_z}|0|0|0,Type=Weapon+Missile,Name=AIM-120_AMRAAM,Color=Yellow\n")

        except Exception as e:
            pass

    print("\n" + "="*50)
    print(f"GÖREV TAMAMLANDI! (Hayatta Kalınan Adım: {adim})")
    print(f"Nihai Skor: {toplam_puan:.2f}")
    print("="*50)
    
    # Tacview dosyasını kapat ve kaydet
    acmi.close()
    
    if hasattr(env, "close"):
        env.close()
        
    print(f"Tacview (.acmi) dosyanız proje ana klasöründe başarıyla oluşturuldu: \n{os.path.abspath(TACVIEW_FILE)}")

if __name__ == "__main__":
    main()