from env.env_virt_minimax import MicroscopeEnv

from sb3_contrib.ppo_mask import MaskablePPO as PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env


if __name__ == "__main__":
    env = make_vec_env(MicroscopeEnv, 8, vec_env_cls=SubprocVecEnv)

    model = PPO('MultiInputPolicy', env, verbose=1, n_steps=2048, ent_coef=0.001, tensorboard_log="./tblog/")

    model = model.learn(total_timesteps=409_600, progress_bar=True, reset_num_timesteps=True)
    model.save("PPO-t7g-shoxx-l3")
