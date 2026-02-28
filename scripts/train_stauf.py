from env.env_t7g import MicroscopeEnv

from sb3_contrib.ppo_mask import MaskablePPO as PPO

if __name__ == "__main__":
    env = MicroscopeEnv()

    model = PPO('MultiInputPolicy', env, verbose=1, n_steps=32, ent_coef=0.001, tensorboard_log="./tblog/")

    model = model.learn(total_timesteps=20_000, progress_bar=True)
    model.save("models/random-play-1")
