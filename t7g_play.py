import os

from env.env_t7g import MicroscopeEnv

from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks

env = MicroscopeEnv()

if os.path.exists("t7g.zip"):
    model = MaskablePPO.load("t7g", env)
    print("Loaded model")

obs, _ = env.reset()
episode_over = False
while not episode_over:
    action_masks = get_action_masks(env)
    action, _ = model.predict(obs, action_masks=action_masks)
    obs, reward, terminated, truncated, info = env.step(action)

    episode_over = terminated or truncated
