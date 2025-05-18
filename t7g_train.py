from env.env_virt import MicroscopeEnv

from sb3_contrib.ppo_mask import MaskablePPO as PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env

if __name__ == "__main__":
    env = make_vec_env(MicroscopeEnv, 8, vec_env_cls=SubprocVecEnv)

    model = PPO('MultiInputPolicy', env, verbose=1, n_steps=256, ent_coef=0.001, tensorboard_log="./tblog/")

    model = model.learn(total_timesteps=200_000, progress_bar=True)
    model.save("3-t7g-random-play")
