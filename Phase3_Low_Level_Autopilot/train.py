import os
import glob
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

    # ==========================================
    # 3. OTOMATİK RESUME (Kesintiye Dayanıklı Kayıt)
    # ==========================================
    # En son periyodik checkpoint'i bulur. Ctrl+C ile kesildiğinde bile
    # kaldığı adımdan devam edebilir (kayıt asla kaybolmaz).
    def find_latest_ckpt(pattern):
        matches = glob.glob(os.path.join(models_dir, pattern))
        if not matches:
            return None
        return max(matches, key=os.path.getmtime)

    def extract_steps(path):
        """ Dosya adındaki adım sayısını çıkarır: sac_actor_phase3_50000_steps.pth -> 50000 """
        try:
            return int(os.path.basename(path).split("_")[-3])
        except Exception:
            return 0

    actor_ckpt = find_latest_ckpt("sac_actor_phase3_*_steps.pth")
    critic_ckpt = find_latest_ckpt("sac_critic_phase3_*_steps.pth")
    vec_ckpt = find_latest_ckpt("sac_env_phase3_*_steps_vec_normalize.pkl")
    start_step = extract_steps(actor_ckpt) if actor_ckpt else 0

    if actor_ckpt:
        print(f"[RESUME] Kaldığı yerden devam: Adım {start_step} (Aktör: {os.path.basename(actor_ckpt)})")
    else:
        print("[RESUME] Periyodik checkpoint yok, sıfırdan başlanıyor.")

    # 2. Alt Beyin Ve Matematik Kurulumu
    BATCH_SIZE = 256
    GAMMA = 0.99
    TAU = 0.005
    ALPHA = 0.2
    MAX_STEPS_TOTAL = 3000000 # Toplam adım sayısı (SB3'teki total_timesteps gibi)
    SAVE_FREQ = 50000         # Checkpoint kayıt aralığı

    actor = SafeActor(state_dim=14, action_dim=3, num_constraints=8)
    critic = SACCritic(state_dim=14, action_dim=3)
    critic_target = SACCritic(state_dim=14, action_dim=3)

    # DMD matrisleri de artık 14x14 boyutunda bir sistem çözecek!
    dmd = RealTimeDMDc(state_dim=14, action_dim=3, window_size=300)
    replay_buffer = ReplayBuffer(state_dim=14, action_dim=3, capacity=1000000)

    # 1. Aktör (Pilot Beyni) Yüklemesi
    if actor_ckpt:
        print(f"[RESUME] Eski Aktör yükleniyor: {os.path.basename(actor_ckpt)}")
        actor.load_state_dict(torch.load(actor_ckpt, map_location="cpu"))
    else:
        transfer_path = os.path.join(models_dir, "sac_actor_transferred_14_sensors.pth")
        if os.path.exists(transfer_path):
            print(f"[İNİT] Aktarım modeli yükleniyor: {os.path.basename(transfer_path)}")
            actor.load_state_dict(torch.load(transfer_path, map_location="cpu"))
        else:
            print("[İNİT] Aktarım modeli bulunamadı, sıfırdan başlıyor.")

    # 2. Eleştirmen (Puanlayıcı) Yüklemesi
    if critic_ckpt:
        print(f"[RESUME] Eski Eleştirmen yükleniyor: {os.path.basename(critic_ckpt)}")
        critic.load_state_dict(torch.load(critic_ckpt, map_location="cpu"))
    else:
        print("[İNİT] Eleştirmen (Critic) kaydı bulunamadı, sıfırdan öğreniyor.")

    critic_target.load_state_dict(critic.state_dict()) # Target ağı eşitle

    actor_optimizer = optim.Adam(actor.parameters(), lr=3e-4)
    critic_optimizer = optim.Adam(critic.parameters(), lr=3e-4)

    # 3. Normalizasyon Yüklemesi
    if vec_ckpt:
        print(f"[RESUME] Normalizasyon kayıtları yükleniyor: {os.path.basename(vec_ckpt)}")
        norm_env = VecNormalize.load(vec_ckpt, vec_env)
        norm_env.training = True
        norm_env.norm_obs = True
    else:
        print("[İNİT] Normalizasyon kaydı bulunamadı, sıfırdan başlıyor.")
        norm_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # ==========================================
    # 4. CBF KALKAN SINIRLARI (C_safe, d_safe)
    # ==========================================
    # Obs_rms (canlı normalize istatistikleri) ile her güncellemede üretilir.
    # Çünkü eğitim sırasında norm_env.obs_rms güncellenir; kısıtlar da o
    # güncel uzayla uyumlu olmalıdır. (Faz 4 formülünün birebir aynısı)
    def build_cbf_bounds(norm_env):
        """ Hibrit CBF: 4 açı kısıtı (pitch/roll bandı) + 4 açısal hız kısıtı (q/p bandı).
        Açı satırları tek adımda fiziksel olarak zayıf kontrol edilir (|B|~0.01),
        hız satırları ise gerçek otoriteye sahiptir (|B|~0.04-0.06). """
        obs_mean_np = np.asarray(norm_env.obs_rms.mean, dtype=np.float32)
        obs_var_np = np.asarray(norm_env.obs_rms.var, dtype=np.float32)
        obs_std = np.sqrt(obs_var_np + 1e-8)

        C_safe_np = np.zeros((1, 8, 14), dtype=np.float32)
        C_safe_np[0, 0, 3] = 1.0   # +Pitch açısı (idx 3)
        C_safe_np[0, 1, 3] = -1.0  # -Pitch açısı
        C_safe_np[0, 2, 2] = 1.0   # +Roll açısı (idx 2)
        C_safe_np[0, 3, 2] = -1.0  # -Roll açısı
        C_safe_np[0, 4, 6] = 1.0   # +Pitch hızı q (idx 6)
        C_safe_np[0, 5, 6] = -1.0  # -Pitch hızı q
        C_safe_np[0, 6, 5] = 1.0   # +Roll hızı p (idx 5)
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

        return torch.FloatTensor(C_safe_np), torch.FloatTensor(d_safe_np)

    C_safe_tensor, d_safe_tensor = build_cbf_bounds(norm_env)

    UPDATE_INTERVAL = 6
    dynamic_limit_A = None
    dynamic_limit_B = None

    # ==========================================
    # 5. EĞİTİM DÖNGÜSÜ (CUSTOM LOOP)
    # ==========================================
    print("Faz 3: CBF Korumalı Eğitim Başlıyor...")

    state = norm_env.reset() # Dikkat: VecEnv olduğu için state shape = (1, 14)
    episode_reward = 0
    episode_length = 0
    episodes_completed = 0
    crash_count = 0

    def save_checkpoint(step, tag):
        """ Kesintiye dayanıklı kayıt: Aktör + Eleştirmen + Normalizasyon """
        if tag == "periodic":
            suffix = f"{step}_steps"
        else:
            suffix = tag
        torch.save(actor.state_dict(), f"{models_dir}/sac_actor_phase3_{suffix}.pth")
        torch.save(critic.state_dict(), f"{models_dir}/sac_critic_phase3_{suffix}.pth")
        norm_env.save(f"{models_dir}/sac_env_phase3_{suffix}_vec_normalize.pkl")
        print(f"[CHECKPOINT] Adım {step} kaydedildi ({suffix}).")

    try:
        for total_steps in range(start_step + 1, MAX_STEPS_TOTAL + 1):

            # --- A) DMD GÜNCELLEMESİ ---
            if total_steps % UPDATE_INTERVAL == 0 and dmd.is_ready:
                A_numpy, B_numpy = dmd.compute_matrices()
                if A_numpy is None or B_numpy is None:
                    # Tampon bozuksa kalkan bu adımda pasif kalır (sistem çökmez)
                    dynamic_limit_A = None
                    dynamic_limit_B = None
                else:
                    dynamic_limit_A = torch.FloatTensor(A_numpy)
                    dynamic_limit_B = torch.FloatTensor(B_numpy)
                    C_safe_tensor, d_safe_tensor = build_cbf_bounds(norm_env)

            # --- B) AJAN KARARI (İcra) ---
            state_tensor = torch.FloatTensor(state)
            shield_ok = 0.0
            with torch.no_grad():
                if dynamic_limit_A is not None and dynamic_limit_B is not None:
                    # Kalkan (CBF) devrede
                    out = actor(state_tensor, dynamic_limit_A, dynamic_limit_B,
                                C_safe_tensor, d_safe_tensor)
                    shield_ok = 1.0
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
            # NOT: Kalkan kapalıyken bile hafızaya yazıyoruz (veri kaybı yok).
            # A/B yoksa sıfır matris + shield_ok=0 bayrağı kaydedilir.
            dmd.add_data(state[0], action_numpy[0], next_state[0])

            if shield_ok:
                A_store = A_numpy.astype(np.float32)
                B_store = B_numpy.astype(np.float32)
            else:
                A_store = np.zeros((14, 14), dtype=np.float32)
                B_store = np.zeros((14, 3), dtype=np.float32)

            replay_buffer.push(state[0], action_numpy[0], reward[0], next_state[0],
                               done[0], A_store, B_store, shield_ok)

            # --- D) YAPAY ZEKA ÖĞRENMESİ (Gradient Flow) ---
            if len(replay_buffer) > BATCH_SIZE:
                states, actions, rewards, next_states, dones, As, Bs, shield_oks = \
                    replay_buffer.sample(BATCH_SIZE)

                valid_mask = (shield_oks > 0.5).flatten()

                # Critic Loss (Kalkanlı ve kalkansız geçişler ayrı işlenir)
                with torch.no_grad():
                    target_value = torch.zeros_like(rewards)

                    if valid_mask.any():
                        next_actions_v, next_log_probs_v = actor(
                            next_states[valid_mask], As[valid_mask], Bs[valid_mask],
                            C_safe_tensor, d_safe_tensor)
                        target_Q1_v, target_Q2_v = critic_target(next_states[valid_mask], next_actions_v)
                        target_Q_v = torch.min(target_Q1_v, target_Q2_v) - ALPHA * next_log_probs_v
                        target_value[valid_mask] = rewards[valid_mask] + GAMMA * (1 - dones[valid_mask]) * target_Q_v

                    if (~valid_mask).any():
                        next_actions_i = actor.net(next_states[~valid_mask])
                        dummy_log_prob_i = torch.zeros((next_actions_i.shape[0], 1))
                        target_Q1_i, target_Q2_i = critic_target(next_states[~valid_mask], next_actions_i)
                        target_Q_i = torch.min(target_Q1_i, target_Q2_i) - ALPHA * dummy_log_prob_i
                        target_value[~valid_mask] = rewards[~valid_mask] + GAMMA * (1 - dones[~valid_mask]) * target_Q_i

                current_Q1, current_Q2 = critic(states, actions)
                critic_loss = F.mse_loss(current_Q1, target_value) + F.mse_loss(current_Q2, target_value)

                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()

                # Actor Loss (CBF İçinden Türev Akışı)
                actor_loss = torch.tensor(0.0)

                if valid_mask.any():
                    safe_actions_pred_v, log_probs_v = actor(
                        states[valid_mask], As[valid_mask], Bs[valid_mask],
                        C_safe_tensor, d_safe_tensor)
                    Q1_pred_v, Q2_pred_v = critic(states[valid_mask], safe_actions_pred_v)
                    Q_pred_v = torch.min(Q1_pred_v, Q2_pred_v)
                    actor_loss = actor_loss + (ALPHA * log_probs_v - Q_pred_v).mean() * (valid_mask.sum().float() / BATCH_SIZE)

                if (~valid_mask).any():
                    raw_actions_i = actor.net(states[~valid_mask])
                    Q1_pred_i, Q2_pred_i = critic(states[~valid_mask], raw_actions_i)
                    Q_pred_i = torch.min(Q1_pred_i, Q2_pred_i)
                    actor_loss = actor_loss + (-Q_pred_i).mean() * ((~valid_mask).sum().float() / BATCH_SIZE)

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
                    writer.add_scalar("Q_Values/Q_Pred_Mean", Q_pred_v.mean().item() if valid_mask.any() else 0.0, total_steps)

                    # Ajanın hareketlilik seviyesi (Log Prob)
                    if valid_mask.any():
                        writer.add_scalar("Action_Stats/Log_Prob", log_probs_v.mean().item(), total_steps)

            # --- E) BÖLÜM (EPISODE) SONU İŞLEMLERİ ---
            # Ya uçak çakılırsa, ya da 2500 adımı devirirse bölüm bitsin
            if done[0] or episode_length >= 2500:
                episodes_completed += 1
                raw_reward = info[0].get('episode', {}).get('r', episode_reward)

                if done[0]:
                    crash_count += 1

                # BVR Hedef Sapma Oranlarını Görelim (O anki hedeften ne kadar saparak çakıldı/bitti?)
                final_pitch_error = abs(state[0][3] - state[0][12]) # Mevcut Pitch - Hedef Pitch
                final_roll_error = abs(state[0][2] - state[0][13])  # Mevcut Roll - Hedef Roll
                final_mach = state[0][1]

                # --- TENSORBOARD: UÇUŞ PERFORMANSI ---
                writer.add_scalar("Rollout/1_Episode_Reward", raw_reward, total_steps)
                writer.add_scalar("Rollout/2_Episode_Length", episode_length, total_steps)
                writer.add_scalar("Safety/Crash_Count", crash_count, total_steps)

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
                save_checkpoint(total_steps, "periodic")

    except KeyboardInterrupt:
        print("\nEğitim Manuel Olarak Durduruldu! Son durum kaydediliyor...")
        save_checkpoint(total_steps, "periodic")
    except Exception as e:
        print(f"\nBeklenmeyen Hata: {e}")
        print("Son durum kurtarılıyor...")
        save_checkpoint(total_steps, "periodic")
        raise

    # ==========================================
    # 5. FİNAL KAYIT İŞLEMLERİ
    # ==========================================
    print("Son veriler kaydediliyor...")
    torch.save(actor.state_dict(), f"{models_dir}/sac_actor_phase3_final.pth")
    torch.save(critic.state_dict(), f"{models_dir}/sac_critic_phase3_final.pth")
    norm_env.save(f"{models_dir}/sac_env_phase3_final_vec_normalize.pkl")
    writer.close()

    print("Faz 3 İşlemi Başarıyla Tamamlandı!")
