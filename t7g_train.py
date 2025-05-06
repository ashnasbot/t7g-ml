from t7g_game import MicroscopeEnv

from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.ppo_mask import MultiInputPolicy

env = MicroscopeEnv()

model = MaskablePPO(MultiInputPolicy, env, verbose=1)

model = model.learn(total_timesteps=81920, progress_bar=True)
model.save("t7g")
