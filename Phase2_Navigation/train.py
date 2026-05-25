import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from FighterEnv import FighterEnv
from core.callbacks import SaveVecNormalizeCallback

if __name__ == "__main__":
    models_dir = "./fighter_checkpoints/phase2_completed/"
    logs_dir = "./fighter_tensorboard/phase2_completed/"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    env = FighterEnv()

    check_env(env)
    print("Çevre Gymnasium standartlarına %100 uygun.")

    monitored_env = Monitor(env)
    vec_env = DummyVecEnv([lambda: monitored_env])

    CHECKPOINT_STEP = 450000
    LOAD_MODEL_PATH = f"./fighter_checkpoints/phase2_completed/sac_f16_phase2_completed_450000_steps.zip"
    LOAD_VEC_PATH = f"./fighter_checkpoints/phase2_completed/sac_f16_phase2_completed_450000_steps_vec_normalize.pkl"


    if LOAD_VEC_PATH and os.path.exists(LOAD_VEC_PATH):
        print(f"Önceki kayıtlar yükleniyor: {LOAD_VEC_PATH}")
        norm_env = VecNormalize.load(LOAD_VEC_PATH, vec_env)
        norm_env.training = True
        norm_env.norm_obs = True
    else:
        print("Kayıt bulunamadı, sıfırdan Otomatik Normalizasyon başlıyor.")
        norm_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # MODEL YÜKLEMESİ
    if LOAD_MODEL_PATH and os.path.exists(LOAD_MODEL_PATH):
            print(f"Önceki model yükleniyor: {LOAD_MODEL_PATH}")
            
            # Zip dosyasının içi zaten kusursuz 'auto' ayarlarına sahip olduğu için
            # doğrudan normal bir şekilde yüklüyoruz. Başka hiçbir kopyalamaya gerek yok.
            model = SAC.load(LOAD_MODEL_PATH, env=norm_env, device="cuda", tensorboard_log=logs_dir)
            print("Eski tecrübeler, yeni 'Auto' ajana aktarıldı.")
    else:
        print("Model bulunamadı, ajan sıfırdan öğreniyor (Yeni 8 Sensörlü/Girdili Mimari).")
        model = SAC(
            policy="MlpPolicy",
            env=norm_env, 
            verbose=1, 
            learning_rate=0.0003,
            batch_size=256,
            ent_coef= "auto",
            tensorboard_log=logs_dir,
            device="cuda"
        )


    # Ağırlıkları (.zip) kaydeder
    checkpoint_callback = CheckpointCallback(
        save_freq=50000, 
        save_path=models_dir,
        name_prefix='sac_f16_phase2_completed'        
    )

    # İstatistikleri (.pkl) kaydeder
    vec_normalize_callback = SaveVecNormalizeCallback(
        save_freq=50000,
        save_path=models_dir,
        name_prefix='sac_f16_phase2_completed'
    )

    callbacks = [checkpoint_callback, vec_normalize_callback]

    print("Eğitim başlıyor...")
    try:
        model.learn(
            total_timesteps=2000000,
            log_interval=4,
            callback=callbacks,
            reset_num_timesteps=False,
            tb_log_name="Phase2_completed_450Kversion"
        )
    except KeyboardInterrupt:
        print("\n Eğitim  durduruldu. Son veriler kaydediliyor...")

    print("Eğitim tamamlandı/durduruldu. Son veriler kaydediliyor...")
    model.save(f"{models_dir}/sac_f16_fighter_phase2_completed_final.zip")
    norm_env.save(f"{models_dir}/sac_f16_fighter_phase2_completed_final_vec_normalize.pkl")

    print("Faz 2 İşlemi Başarıyla Tamamlandı!")