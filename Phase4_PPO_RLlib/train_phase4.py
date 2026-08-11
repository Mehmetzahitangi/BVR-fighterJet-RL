import os
import sys
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env

from phase4_envs.bvr_env_rllib import BVRCombatEnv
from core.callbacks import BVRPhaseCallback

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

def env_creator(env_config):
    return BVRCombatEnv(env_config)

if __name__ == "__main__":
    print("Faz 4: Hiyerarşik PPO (RLlib) Eğitim Başlatılıyor...")

    register_env("BVRCombatEnv_v0", env_creator)
    ray.init(ignore_reinit_error=True)

    config = (
        PPOConfig()
        .environment("BVRCombatEnv_v0")
        .framework("torch")
        .env_runners(num_env_runners=4, num_envs_per_env_runner=1)
        .resources(num_gpus=1)
        .fault_tolerance(restart_failed_env_runners=True)
        .training(
            lr=3e-4,
            gamma=0.99,
            clip_param=0.2,
            vf_loss_coeff=0.5,
            entropy_coeff=0.01,
            train_batch_size_per_learner=4000,
            minibatch_size=512,
            num_epochs=10
        )
        .callbacks(BVRPhaseCallback)
    )


    #print("\n[SİSTEM] Eğitim Motoru Ateşleniyor (Kayıtlar Komutanın Kontrolünde)...")
    #results = tune.run(
    #    "PPO",
    #    name="Phase4_BVR_Dogfight",
    #    config=config.to_dict(),
    #    stop={"training_iteration": 500}, # İsterseniz Ctrl+C ile erken durdurabilirsiniz
    #    storage_path=os.path.abspath("./ray_results"),
    #    verbose=1 ,
    #    checkpoint_freq=2,
    #    checkpoint_at_end=True,
    #    keep_checkpoints_num=3,
    #    resume=True,
    #)

    print("\n[SİSTEM] Eğitim Motoru Ateşleniyor (Kayıtlar Komutanın Kontrolünde)...")
    results = tune.run(
        "PPO",
        name="Phase4_BVR_Dogfight_V2",
        config=config.to_dict(),
        stop={"training_iteration": 500}, 
        storage_path=os.path.abspath("./ray_results"),
        verbose=1,
        checkpoint_freq=2,
        checkpoint_at_end=True,
        keep_checkpoints_num=3,
        
        # DİKKAT 2: resume=True satırını TAMAMEN SİLDİK, yerine aşağıdakini yazdık:
        #restore="C:/Zahit/BVR_fighterJet_AI/Phase4_PPO_RLlib/ray_results/Phase4_BVR_Dogfight/PPO_BVRCombatEnv_v0_8511a_00000_0_2026-08-08_12-57-09/checkpoint_000049" 
        resume=True,
    )
    
    print("\n Eğitim Tamamlandı!")
    ray.shutdown()