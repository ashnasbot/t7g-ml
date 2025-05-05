from t7g_env import MicroscopeEnv

from stable_baselines3.common.vec_env import DummyVecEnv

from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

env = DummyVecEnv([MicroscopeEnv])

model = MaskablePPO(MaskableActorCriticPolicy, env, verbose=1)

model = model.learn(total_timesteps=8192, progress_bar=True)
model.save("t7g")
