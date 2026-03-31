# T7G Microscope — ML Gym

Train RL agents to play the Microscope minigame from *The 7th Guest* — an Ataxx-like board game on a 7×7 grid.

## History
initally this project started as building a BFS solver for the game to find the 'optimal' line to beat Stauf.
it turns out Microscope has a branching factor of 50-60, note Chess is ~35.
this leads us to a roughly several billion year calculation to 'solve' the game, so we quickly turned to heuristic models.
We may revisit a Retrograde analysis in the future.

`micro_3.c` optimised Darkshoxx's single move solver into a general 'find-best-move' calc, using various depths.
several rounds of optimisation later and it can calc to MM5 in less than a milisecond on average, giving us a very solid player to
train and eval against.

lots of experimentation followed, starting with PPO, action masking and much hair-pulling.

Eventually the plan shifted towards an AlphaZero implementation - the current model.
particular features and enhancements:
- C implmentation
  imposes a number of limits, but scoped to the current game
- game pool based inference
  had a lot of trouble getting a proper server working, so we run 32 games in parallel and batch the inferences
- MCGS
  we use a transposition table and loop detection to for a Monte Carlo Graph Search instead of AlphaZero's Trees
  This is due to the large number of symmetries in Microscope vs similar games, allowing for batching of sim results (not proven)
- Gumbel + Sequential Halving
- replacing policy targets with a soft minimax distribution to get around bootstrapping issues (may remove)
  This also produced micro4.c a slower/weaker minimaxer, but which shows better correllation in the midgame to W/L results
- prototype game gym harness - can play in the real game by clicking the mouse

## Current results

~2ply minimaxer - we've a long way to go, but we may trade with the 'Stauf' UI in the real game.

## Installation

```bash
git clone <repository-url>
cd t7g-ml-gym

# Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux / macOS

# Install dependencies
pip install -r requirements.txt

# Install this package in editable mode (adds the project root to sys.path)
pip install -e .
```

> **PyTorch**: the above installs the CPU build. For GPU training see the PyTorch
> section at the bottom of [requirements.txt](requirements.txt) for ROCm and CUDA variants.

### Compile the C DLLs

Two minimax solvers and the MCTS tree are compiled C extensions. Build them once after checkout:

```bash
make dll
```

This compiles `micro3.dll`, `micro4.dll`, `micro_mcts.dll`, and `cell_dll.dll` into `lib/`.
`make dll-native` rebuilds with `-march=native` for the local CPU. `make clean` removes all built DLLs.

---

## Training

Train a dual-head (policy + value) network via AlphaZero-style MCTS self-play:

```bash
.venv/Scripts/python scripts/train_mcts.py
```

Checkpoints are saved to `models/mcts/` every iteration. Resume from a checkpoint:

```bash
.venv/Scripts/python scripts/train_mcts.py --checkpoint models/mcts/iter_0050.pt
```

Key options:

| Flag | Default | Description |
|---|---|---|
| `--iterations` | 500 | Number of self-play → train iterations |
| `--games` | 250 | Self-play games per iteration |
| `--simulations` | 250 | MCTS simulations per move |
| `--checkpoint` | — | Resume from a saved checkpoint |
| `--logdir` | `tblog/mcts` | TensorBoard log directory |
| `--run-name` | auto | Run label shown in TensorBoard |

Monitor training:

```bash
tensorboard --logdir=tblog/
```

---

## Evaluation

Evaluate a checkpoint against the minimax opponent:

```bash
.venv/Scripts/python scripts/play_mcts.py --checkpoint models/mcts/iter_0050.pt
```

Options:

| Flag | Default | Description |
|---|---|---|
| `--checkpoint` | required | Model to evaluate |
| `--games` | 20 | Number of evaluation games |
| `--depth` | 2 | Minimax search depth |
| `--simulations` | 100 | MCTS simulations per move |
| `--watch` | off | Print board state each move |

---

## Playing

### GUI

Requires `pyglet` (`pip install pyglet`):

```bash
# Play against bundled MCTS model (default)
.venv/Scripts/python scripts/play_gui.py

# Play against a specific checkpoint
.venv/Scripts/python scripts/play_gui.py --checkpoint models/mcts/iter_0100.pt

# Play against minimax
.venv/Scripts/python scripts/play_gui.py --opponent micro4 --depth 3

# Play as Green
.venv/Scripts/python scripts/play_gui.py --human-color green
```

Options:

| Flag | Default | Description |
|---|---|---|
| `--checkpoint` | bundled | Model file |
| `--opponent` | `mcts` | `mcts`, `micro3`, `micro4`, or `stauf` |
| `--depth` | 2 | Minimax search depth |
| `--simulations` | 100 | MCTS simulations per move |
| `--human-color` | `blue` | Your piece colour (`blue` or `green`) |

Click a piece to select it, then click the destination. Clones land within 1 step; jumps land 2 steps away.

---

## Running Tests

```bash
.venv/Scripts/python -m pytest tests/ -v
```

---

## Acknowledgements

- Minimax implementation adapted from [Darkshoxx](https://github.com/darkshoxx)
- Built with PyTorch
