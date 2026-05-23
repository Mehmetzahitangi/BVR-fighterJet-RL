import os
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecNormalize

class SaveVecNormalizeCallback(BaseCallback):
    """
    Her N adımda bir hem SB3 modelini (.zip) hem de VecNormalize istatistiklerini (.pkl)
    senkronize olarak kaydeder. Olası çökmelerde eğitimin hafıza kaybı yaşamasını önler.
    """
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
            if isinstance(self.training_env, VecNormalize):
                path = os.path.join(self.save_path, f"{self.name_prefix}_{self.num_timesteps}_steps_vec_normalize.pkl")
                self.training_env.save(path)
                if self.verbose > 0:
                    print(f"İstatistikler kaydedildi: {path}")
        return True