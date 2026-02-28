# T7G Microscope — ML Gym

Train RL agents to play the Microscope minigame from *The 7th Guest* — an Ataxx-like board game on a 7×7 grid.

## Quick Start

```bash
# Train
python train.py

# Monitor
tensorboard --logdir=./tblog/

# Play against MCTS
python scripts/play_mcts.py

# Play with GUI
python scripts/play_gui.py
```

## Installation

```bash
git clone <repository-url>
cd t7g-ml-gym
pip install -r requirements.txt

# Compile the C minimax opponent (Windows)
gcc -O3 -march=native -ffast-math micro_3.c -o micro3.dll --shared
```

See [requirements.txt](requirements.txt) for GPU (ROCm/CUDA) torch install instructions.

## Project Structure

```
├── train.py                  # Main PPO training entry point
├── train_curriculum.py       # Curriculum learning variant
├── micro_3.c                 # C minimax implementation (compile → micro3.dll)
│
├── lib/
│   ├── t7g.py                # Game logic, board helpers
│   ├── networks.py           # CNN policy architectures
│   ├── dual_network.py       # Dual-head network (policy + value)
│   ├── reward_functions.py   # Reward shaping functions
│   └── mcts.py               # Monte Carlo Tree Search
│
├── env/
│   ├── env_virt.py           # Fast virtual self-play environment
│   ├── env_t7g.py            # Real game environment (screen capture)
│   ├── minimax_wrapper.py    # Minimax opponent wrapper
│   ├── random_opponent_wrapper.py
│   ├── action_mask_wrapper.py
│   ├── position_curriculum_wrapper.py
│   └── symmetry_wrapper.py
│
├── scripts/
│   ├── train_stauf.py        # Train against Stauf (game AI)
│   ├── train_mcts.py         # Train against MCTS opponent
│   ├── self_play.py          # Self-play training
│   ├── play_gui.py           # GUI play (requires pyglet)
│   ├── play_mcts.py          # Play against MCTS
│   └── generate_position_bank.py
│
└── tests/
```

## Acknowledgements

- Minimax implementation adapted from [Darkshoxx](https://github.com/darkshoxx)
- Built with [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) and PyTorch