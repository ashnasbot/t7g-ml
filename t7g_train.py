from t7g_virt_env import MicroscopeEnv

from sb3_contrib.ppo_mask import MaskablePPO as PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env

if __name__ == "__main__":
    env = make_vec_env(MicroscopeEnv, 16, vec_env_cls=SubprocVecEnv)

    model = PPO('MultiInputPolicy', env, verbose=1, n_steps=2048)

    model = model.learn(total_timesteps=20_000_000, progress_bar=True)
    model.save("t7g")
