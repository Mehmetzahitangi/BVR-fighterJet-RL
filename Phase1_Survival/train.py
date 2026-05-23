import os
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor # DummyVecEnv, VecNormalize eklediğimiz için ep_len_mean ve ep_rew_mean'i Monitor kullanarak kaydedip görüntüleyebiliriz
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from FighterEnv import FighterEnv


class SaveVecNormalizeCallback(BaseCallback):
    def __init__(self, save_freq: int, save_path: str, name_prefix: str, verbose=1):
        super(SaveVecNormalizeCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix

    def _init_callback(self) -> None:
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            # Sadece VecNormalize kullanılıyorsa kaydet
            if isinstance(self.training_env, VecNormalize):
                path = os.path.join(self.save_path, f"{self.name_prefix}_{self.num_timesteps}_steps_vec_normalize.pkl")
                self.training_env.save(path)
                if self.verbose > 0:
                    print(f"İstatistikler kaydedildi: {path}")
        return True
    

if __name__ == "__main__":
    print("JSBSim Fighter Ortamı Başlatılıyor...")
    env = FighterEnv()

    # SB3 UYUM TESTİ
    # SB3 ortamın Gym standartlarına %100 uygun olup olmadığını denetler
    check_env(env)
    print("Çevre Gymnasium standartlarına %100 uygun")

    # Ham çevreyi Monitor ile sarıyoruz ki skorları kaydetsin
    monitored_env = Monitor(env)

    # Çevreyi Vektörel formata sarıyoruz (SB3 Standardı kullanarak)
    vec_env = DummyVecEnv([lambda: monitored_env])


    LOAD_MODEL_PATH = None # Örn: "./fighter_checkpoints/sac_f16_model_200000_steps.zip" veya sıfırdan eğitim istiyorsak none kalmalı
    LOAD_VEC_PATH = None   # Örn: "./fighter_checkpoints/sac_f16_model_200000_steps_vec_normalize.pkl" veya sıfırdan eğitim istiyorsak none kalmalı

    # Normalizasyon İstatistikleri Yükleme
    # VecNormalize katmanını önceki kayıttan yüklüyoruz
    # Eğer istatistikleri ayrı kaydetmezsek, SB3 model zip'i içinde saklamaz.
    # Daha önceki eğitim sırasında env.save("vec_normalize.pkl") yapılmış olmalı.
    if LOAD_VEC_PATH and os.path.exists(LOAD_VEC_PATH):
        print(f"Önceki istatistikler yükleniyor: {LOAD_VEC_PATH}")
        norm_env = VecNormalize.load(LOAD_VEC_PATH, vec_env)
        # Eğitim devam edeceği için istatistiklerin güncellenmesine izin veriyoruz
        norm_env.training = True
        norm_env.norm_obs = True
    else:
        print("İstatistik bulunamadı, sıfırdan Otomatik Normalizasyon başlıyor.")
        # Eğer ilk kez kuruluyorsa
        # Stable Baselines'ın Otomatik Normalizasyonunu Kullanıyoruz
        # Bu katman 15000 (feet) gibi devasa sayıları otomatik olarak -1 ile 1 arasına çeker. Gradient Exploding'i engelleriz.
        norm_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # SAC Algoritması (Maksimum entropi, exploring teşviği ve smooth öğrenme )
    print("SAC Yükleniyor...")


    # Sinir Ağı Ağırlıklarını Yükleme
    if LOAD_MODEL_PATH and os.path.exists(LOAD_MODEL_PATH):
        print(f"Önceki tecrübeler (Model) yükleniyor: {LOAD_MODEL_PATH}")
        # Modeli önceki ağırlıklarla ayağa kaldır ve çevreyi (norm_env) bağla
        model = SAC.load(LOAD_MODEL_PATH, env=norm_env, device="cuda")
    else:
        print("Model bulunamadı, ajan sıfırdan başlıyor.")
        model = SAC(
            policy="MlpPolicy",          # Vektörel veriler (Sensör verileri) için Çok Katmanlı Perceptron
            env=norm_env, 
            verbose=1,                   # Terminale çıktı versin
            learning_rate=0.0003,        # Öğrenme hızı (standart değer)
            batch_size=256,              # Her seferde 256 kez tecrübe öğrensin
            tensorboard_log="./fighter_tensorboard/",
            device="cuda"
        )

    # CALLBACK SİSTEMİ (Checkpoint + VecNormalize Kaydedici)

    checkpoint_callback = CheckpointCallback(
        save_freq=50000, 
        save_path='./fighter_checkpoints/',
        name_prefix='sac_f16_model'         
    )

    vec_normalize_callback = SaveVecNormalizeCallback(
        save_freq=50000,
        save_path='./fighter_checkpoints/',
        name_prefix='sac_f16_model'
    )

    # Stable Baselines listeler halinde birden fazla callback'i destekler
    callbacks = [checkpoint_callback, vec_normalize_callback]

    print("Eğitim Başlıyor...!")

    model.learn(total_timesteps=500000, 
                log_interval=4,
                callback=callbacks,
                reset_num_timesteps=False # Eğitimin kaldığı adımdan (örn 200.000) devam etmesi için FALSE OLMALI
                #tb_log_name="SAC_Resume_Ucusu"
                )

    model_path = "sac_fighter_phase1_rewardShaping_normalized"
    model.save(model_path)
    norm_env.save("sac_fighter_phase1_final_vec_normalize.pkl")
    print(f"Eğitim Tamamlandı! Model kaydedildi: {model_path}.zip")