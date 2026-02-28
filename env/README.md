# Environments Directory

This directory contains the Microscope game environments for training and playing.

## Environments

### [env_virt.py](env_virt.py) - Virtual Training Environment

**Primary environment for ML training.** Fast, configurable, with multiple reward functions.

#### Usage

```python
from env.env_virt import MicroscopeEnv, MicroscopeEnvDense, MicroscopeEnvSimple, MicroscopeEnvStrategic

# Base environment (original rewards - has bugs)
env = MicroscopeEnv()

# Or specify reward function
env = MicroscopeEnv(reward_fn='dense')     # Best for initial training
env = MicroscopeEnv(reward_fn='simple')    # Good baseline
env = MicroscopeEnv(reward_fn='strategic') # Advanced training

# Or use convenience classes (recommended)
env = MicroscopeEnvDense()      # Dense rewards - best for initial training
env = MicroscopeEnvSimple()     # Simple linear rewards
env = MicroscopeEnvStrategic()  # Strategic multi-component rewards
```

#### Reward Functions

See [reward_functions.py](../reward_functions.py) for implementation details.

**Recommended progression:**
1. Start with `MicroscopeEnvDense` (500k steps)
2. Continue with `MicroscopeEnvSimple` (1M steps)
3. Polish with `MicroscopeEnvStrategic` (2M steps)

### [minimax_wrapper.py](minimax_wrapper.py) - Minimax Opponent Wrapper

Wraps environment to add a minimax opponent for evaluation.

#### Usage

```python
from env.env_virt import MicroscopeEnvSimple
from env.minimax_wrapper import MinimaxOpponentWrapper

# Wrap environment to add minimax opponent
env = MicroscopeEnvSimple()
env = MinimaxOpponentWrapper(env, depth=3)

# Now opponent responds automatically after each agent move
obs, _ = env.reset()
action = model.predict(obs, action_masks=env.action_masks())[0]
obs, reward, terminated, truncated, info = env.step(action)  # Opponent moves automatically
```

### [env_t7g.py](env_t7g.py) - Actual Game Environment

**For playing on actual t7g Microscope game.**

#### Usage

```python
from env.env_t7g import T7GEnv

# Connect to game board
env = T7GEnv()
```

## Quick Start

```python
# Training (use virtual env)
from env.env_virt import MicroscopeEnvDense
from stable_baselines3.common.vec_env import SubprocVecEnv

env = SubprocVecEnv([lambda: MicroscopeEnvDense() for _ in range(8)])

# Playing (use virtual or hardware)
from env.env_virt import MicroscopeEnv

env = MicroscopeEnv(render_mode='human')
obs, _ = env.reset()

while True:
    action = model.predict(obs, action_masks=env.action_masks())[0]
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
```

## Environment Specs

- **Observation space**: `Box(0, 1, (7, 7, 2), bool)` - 7x7 board, 2 channels (player/opponent)
- **Action space**: `Discrete(49)` - 49 from positions, 25 to positions
- **Action masking**: Supported via `env.action_masks()`