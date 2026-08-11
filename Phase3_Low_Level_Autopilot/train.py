import os
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.tensorboard import SummaryWriter # TensorBoard için
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from envs.f16_env import F16Env
from models.sac_actor import SafeActor
from models.sac_critic import SACCritic
from utils.dmd_solver import RealTimeDMDc
from utils.replay_buffer import ReplayBuffer

if __name__ == "__main__":
    # ==========================================
    # 1. KLASÖR VE TENSORBOARD KURULUMU
    # ==========================================
    models_dir = "./fighter_checkpoints/phase3_cbf/"
    logs_dir = "./fighter_tensorboard/phase3_cbf/"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=logs_dir) # TensorBoard yazıcımız

    # ==========================================
    # 2. ÇEVRE VE NORMALİZASYON KURULUMU
    # ==========================================
    base_env = F16Env()
    monitored_env = Monitor(base_env) # Ödül ve bölüm uzunluğunu izler
    vec_env = DummyVecEnv([lambda: monitored_env])

    LOAD_MODEL_PATH_ACTOR = "./fighter_checkpoints/phase3_cbf/sac_actor_transferred_14_sensors.pth"
    LOAD_MODEL_PATH_CRITIC = None
    LOAD_VEC_PATH = None 

    if LOAD_VEC_PATH and os.path.exists(LOAD_VEC_PATH):
        print(f"Önceki normalizasyon kayıtları yükleniyor: {LOAD_VEC_PATH}")
        norm_env = VecNormalize.load(LOAD_VEC_PATH, vec_env)
        norm_env.training = True
        norm_env.norm_obs = True
    else:
        print("Normalizasyon kaydı bulunamadı, sıfırdan başlıyor.")
        norm_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # ==========================================
    # 3. YAPAY ZEKA VE MATEMATİK KURULUMU
    # ==========================================
    BATCH_SIZE = 256
    GAMMA = 0.99
    TAU = 0.005
    ALPHA = 0.2
    MAX_STEPS_TOTAL = 3000000 # Toplam adım sayısı (SB3'teki total_timesteps gibi)
    SAVE_FREQ = 50000         # Checkpoint kayıt aralığı

    # train.py içinde:
    actor = SafeActor(state_dim=14, action_dim=3)
    critic = SACCritic(state_dim=14, action_dim=3)
    critic_target = SACCritic(state_dim=14, action_dim=3)

    # DMD matrisleri de artık 14x14 boyutunda bir sistem çözecek!
    dmd = RealTimeDMDc(state_dim=14, action_dim=3, window_size=60)
    replay_buffer = ReplayBuffer(state_dim=14, action_dim=3, capacity=1000000)

    # 1. Aktör (Pilot Beyni) Yüklemesi
    if LOAD_MODEL_PATH_ACTOR and os.path.exists(LOAD_MODEL_PATH_ACTOR):
        print(f"Eski Aktör (Beyin) yükleniyor: {LOAD_MODEL_PATH_ACTOR}")
        actor.load_state_dict(torch.load(LOAD_MODEL_PATH_ACTOR))
    else:
        print("Aktör bulunamadı, sıfırdan başlıyor.")

    # 2. Eleştirmen (Puanlayıcı) Yüklemesi
    if LOAD_MODEL_PATH_CRITIC and os.path.exists(LOAD_MODEL_PATH_CRITIC):
        print(f"Eski Eleştirmen yükleniyor: {LOAD_MODEL_PATH_CRITIC}")
        critic.load_state_dict(torch.load(LOAD_MODEL_PATH_CRITIC))
    else:
        print("Eleştirmen (Critic) kaydı bulunamadı, yeni kalkanlı dünyaya göre SIFIRDAN öğreniyor.")
    
    critic_target.load_state_dict(critic.state_dict()) # Target ağı eşitle

    actor_optimizer = optim.Adam(actor.parameters(), lr=3e-4)
    critic_optimizer = optim.Adam(critic.parameters(), lr=3e-4)


    UPDATE_INTERVAL = 6
    dynamic_limit_A = None
    dynamic_limit_B = None

    # ==========================================
    # 4. EĞİTİM DÖNGÜSÜ (CUSTOM LOOP)
    # ==========================================
    print("Faz 3: CBF Korumalı Eğitim Başlıyor...")
    
    state = norm_env.reset() # Dikkat: VecEnv olduğu için state shape = (1, 14)
    episode_reward = 0
    episode_length = 0
    episodes_completed = 0

    try:
        for total_steps in range(1, MAX_STEPS_TOTAL + 1):
            
            # --- A) DMD GÜNCELLEMESİ ---
            if total_steps % UPDATE_INTERVAL == 0 and dmd.is_ready:
                A_numpy, B_numpy = dmd.compute_matrices()
                dynamic_limit_A = torch.FloatTensor(A_numpy)
                dynamic_limit_B = torch.FloatTensor(B_numpy)

            # --- B) AJAN KARARI (İcra) ---
            state_tensor = torch.FloatTensor(state)
            with torch.no_grad():
                if dynamic_limit_A is not None:
                    # Kalkan (CBF) devrede
                    out = actor(state_tensor, dynamic_limit_A, dynamic_limit_B)
                else:
                    # İlk saniyeler, kalkan yok, ham sinir ağı devrede
                    out = actor.net(state_tensor) 
                
                # Modelin çıktısı tuple (action, log_prob) ise 0. indexi al, 
                # tek bir tensör ise direkt tensörün kendisini al. (Kurşun Geçirmez Mantık)
                safe_action_tensor = out[0] if isinstance(out, tuple) else out

            action_numpy = safe_action_tensor.detach().numpy() # Shape = (1, 3)
            
            # --- C) SİMÜLASYON ADIMI ---
            next_state, reward, done, info = norm_env.step(action_numpy)
            episode_reward += reward[0] # VecEnv array döndürdüğü için index 0
            episode_length += 1
            
            # DMD ve Hafızaya veri ekleme (Index 0 ile batch boyutunu kırıyoruz)
            dmd.add_data(state[0], action_numpy[0], next_state[0])
            
            if dynamic_limit_A is not None:
                # CBF Matrisleri ile birlikte özel hafızaya yaz
                replay_buffer.push(state[0], action_numpy[0], reward[0], next_state[0], done[0], A_numpy, B_numpy)
            
            # --- D) YAPAY ZEKA ÖĞRENMESİ (Gradient Flow) ---
            if len(replay_buffer) > BATCH_SIZE:
                states, actions, rewards, next_states, dones, As, Bs = replay_buffer.sample(BATCH_SIZE)
                
                # Critic Loss
                with torch.no_grad():
                    next_actions, next_log_probs = actor(next_states, As, Bs)
                    target_Q1, target_Q2 = critic_target(next_states, next_actions)
                    target_Q = torch.min(target_Q1, target_Q2) - ALPHA * next_log_probs
                    target_value = rewards + GAMMA * (1 - dones) * target_Q

                current_Q1, current_Q2 = critic(states, actions)
                critic_loss = F.mse_loss(current_Q1, target_value) + F.mse_loss(current_Q2, target_value)

                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()

                # Actor Loss (CBF İçinden Türev Akışı)
                safe_actions_pred, log_probs = actor(states, As, Bs) 
                Q1_pred, Q2_pred = critic(states, safe_actions_pred)
                Q_pred = torch.min(Q1_pred, Q2_pred)
                actor_loss = (ALPHA * log_probs - Q_pred).mean()

                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                # Polyak Güncellemesi
                for target_param, param in zip(critic_target.parameters(), critic.parameters()):
                    target_param.data.copy_(target_param.data * (1.0 - TAU) + param.data * TAU)

                # TensorBoard Log (Sürekli)
                if total_steps % 100 == 0:
                    writer.add_scalar("Loss/1_Actor", actor_loss.item(), total_steps)
                    writer.add_scalar("Loss/2_Critic", critic_loss.item(), total_steps)
                    
                    # Q-Değerlerinin sağlığı (Critic ne kadar ceza bekliyor?)
                    writer.add_scalar("Q_Values/Q_Pred_Mean", Q_pred.mean().item(), total_steps)
                    
                    # Ajanın hareketlilik seviyesi (Log Prob)
                    writer.add_scalar("Action_Stats/Log_Prob", log_probs.mean().item(), total_steps)
                    
            # --- E) BÖLÜM (EPISODE) SONU İŞLEMLERİ ---
            # Ya uçak çakılırsa, ya da 2500 adımı devirirse bölüm bitsin
            if done[0] or episode_length >= 2500:
                episodes_completed += 1
                raw_reward = info[0].get('episode', {}).get('r', episode_reward)
                
                # BVR Hedef Sapma Oranlarını Görelim (O anki hedeften ne kadar saparak çakıldı/bitti?)
                final_pitch_error = abs(state[0][3] - state[0][12]) # Mevcut Pitch - Hedef Pitch
                final_roll_error = abs(state[0][2] - state[0][13])  # Mevcut Roll - Hedef Roll
                final_mach = state[0][1]
                
                # --- TENSORBOARD: UÇUŞ PERFORMANSI ---
                writer.add_scalar("Rollout/1_Episode_Reward", raw_reward, total_steps)
                writer.add_scalar("Rollout/2_Episode_Length", episode_length, total_steps)
                
                writer.add_scalar("Aerodynamics/Final_Mach", final_mach, total_steps)
                writer.add_scalar("Aerodynamics/Pitch_Error_Rad", final_pitch_error, total_steps)
                writer.add_scalar("Aerodynamics/Roll_Error_Rad", final_roll_error, total_steps)
                
                print(f"Bölüm: {episodes_completed} | Adım: {total_steps} | Ödül: {raw_reward:.1f} | Pitch Hata: {final_pitch_error:.2f} | Roll Hata: {final_roll_error:.2f}")
                
                state = norm_env.reset()
                episode_reward = 0
                episode_length = 0
            else:
                state = next_state

            # --- F) CHECKPOINT KAYIT SİSTEMİ ---
            if total_steps % SAVE_FREQ == 0:
                print(f"\n[CHECKPOINT] Adım {total_steps} kaydediliyor...")
                torch.save(actor.state_dict(), f"{models_dir}/sac_actor_phase3_{total_steps}_steps.pth")
                torch.save(critic.state_dict(), f"{models_dir}/sac_critic_phase3_{total_steps}_steps.pth")
                norm_env.save(f"{models_dir}/sac_env_phase3_{total_steps}_steps_vec_normalize.pkl")

    except KeyboardInterrupt:
        print("\nEğitim Manuel Olarak Durduruldu!")

    # ==========================================
    # 5. FİNAL KAYIT İŞLEMLERİ
    # ==========================================
    print("Son veriler kaydediliyor...")
    torch.save(actor.state_dict(), f"{models_dir}/sac_actor_phase3_final.pth")
    torch.save(critic.state_dict(), f"{models_dir}/sac_critic_phase3_final.pth")
    norm_env.save(f"{models_dir}/sac_env_phase3_final_vec_normalize.pkl")
    writer.close()
    
    print("Faz 3 İşlemi Başarıyla Tamamlandı!")