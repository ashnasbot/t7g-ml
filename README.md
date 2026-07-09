# T7G ML Gym

Board-game AI for two minigames from the Trilobyte games: *The 7th Guest* & *The 11th Hour*:

- **Microscope** — an Ataxx-like game on a 7×7 grid. Full AlphaZero-style
  self-play training pipeline plus fast C minimax engines to train and evaluate
  against.
- **The Beehive** — a hexagonal Ataxx variant on a 61-cell board, with a
  playable GUI and a C minimax opponent.

## History

The project started as a BFS solver hunting the optimal line to beat Stauf.
Microscope turns out to have a branching factor of 50–60 (chess is ~35), which
puts a full solve several billion years out of reach — so it quickly became a
heuristic-model problem. (A retrograde analysis may happen someday.)

`micro_3.c` grew out of Darkshoxx's single-move solver into a general
find-best-move search. After several rounds of optimisation it evaluates to
MM5 in under a millisecond on average, giving a solid opponent to train and
measure against.

A lot of experimentation followed — PPO, action masking, much hair-pulling —
before the plan settled on an AlphaZero implementation, the current model.
Notable pieces:

- **C game core** — the engine and search are compiled C, scoped tightly to
  these games for speed.
- **Pooled inference** — 32 games run in parallel and their inferences are
  batched, in place of a separate inference server.
- **MCGS** — a Monte-Carlo *Graph* Search (transposition table + loop
  detection) rather than AlphaZero's trees, to exploit Microscope's many board
  symmetries.
- **Gumbel + Sequential Halving** for root action selection.
- **Soft-minimax policy targets** to sidestep value bootstrapping issues on
  short tactical games. This also produced `micro_4.c`, a slower/weaker
  minimaxer that correlates better with midgame win/loss.

## Installation

```bash
git clone <repository-url>
cd t7g-ml-gym

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate      # Linux / macOS
# .venv\Scripts\activate       # Windows

# Install dependencies and this package (editable)
pip install -r requirements.txt
pip install -e .
```

> **PyTorch**: the above installs the CPU build. For GPU training see the
> PyTorch section at the bottom of [requirements.txt](requirements.txt) for
> ROCm and CUDA variants.

### Build the C engines

The minimax solvers and the MCTS graph search are compiled C extensions. Build
them once after checkout:

```bash
make dll
```

This compiles `micro3`, `micro4`, `micro_mcts`, `micro_mcts_heuristic`, and
`beehive4` into `lib/`. `make dll-native` rebuilds with `-march=native` for the
local CPU; `make clean` removes the built libraries.

---

## Microscope

### Training

Train a dual-head (policy + value) network via AlphaZero-style MCTS self-play:

```bash
python scripts/train_mcts.py
```

Checkpoints are saved to `models/mcts/` every iteration. Resume with
`--checkpoint models/mcts/iter_0050.pt`.

| Flag | Default | Description |
|---|---|---|
| `--checkpoint` | – | Resume from a saved checkpoint |
| `--iterations` | 500 | Self-play → train iterations |
| `--games` | 250 | Self-play games per iteration |
| `--simulations` | 500 | MCTS simulations per move |
| `--lr` | 1e-4 | Learning rate (constant for the run) |
| `--logdir` | `tblog/mcts` | TensorBoard log directory |
| `--relabel` | off | Relabel MCTS visit-count targets via minimax policy distillation |
| `--bc-warmup` | 0 | Pre-fill the replay buffer with N behavioural-cloning games before iteration 1 |
| `--bc-depth` | 3 | Minimax depth for BC warmup data |
| `--bc-epochs` | 100 | Training epochs on BC data before self-play begins |
| `--bc-cache` | – | Path to save/load BC data (`.npz`); `auto` derives it from params |

Monitor with `tensorboard --logdir=tblog/`.

### Evaluation

Evaluate a checkpoint against the minimax opponent:

```bash
python scripts/play_mcts.py --checkpoint models/mcts/iter_0050.pt
```

| Flag | Default | Description |
|---|---|---|
| `--checkpoint` | required | Model to evaluate |
| `--games` | 20 | Number of evaluation games |
| `--depth` | 2 | Minimax search depth |
| `--simulations` | 100 | MCTS simulations per move |
| `--watch` | off | Print the board each move |

### Play (GUI)

Requires `pyglet`. The C engines play immediately after `make dll`; the `mcts`
opponent needs a trained checkpoint.

```bash
# Play against a minimax engine (no model needed)
python scripts/play_gui.py --opponent micro3 --depth 3

# Play against a trained MCTS model
python scripts/play_gui.py --opponent mcts --checkpoint models/mcts/iter_0100.pt

# Play as Green
python scripts/play_gui.py --opponent micro3 --human-color green
```

| Flag | Default | Description |
|---|---|---|
| `--opponent` | `mcts` | `mcts`, `micro3`, `micro4t`, or `hmcts` |
| `--checkpoint` | – | MCTS model file (for `--opponent mcts`) |
| `--depth` | 2 | Minimax search depth |
| `--simulations` | 100 | MCTS simulations per move |
| `--human-color` | `blue` | Your colour (`blue` or `green`) |

Click a piece to select it, then click the destination. Clones land within 1
step; jumps land 2 steps away.

---

## The Beehive

A hexagonal Ataxx variant on a 61-cell board. Play the GUI against the
`beehive4` C minimax:

```bash
# Play against the minimax opponent
python scripts/play_beehive.py --opponent minimax --ai-time 2000

# Play as Red
python scripts/play_beehive.py --human-color red

# Practice against random moves
python scripts/play_beehive.py --opponent random
```

| Flag | Default | Description |
|---|---|---|
| `--opponent` | `minimax` | `minimax` or `random` |
| `--human-color` | `yellow` | Your colour (`yellow` or `red`) |
| `--ai-time` | 1000 | Minimax time budget per move (ms) |
| `--ai-delay` | 0.4 | Pause before the AI moves (s) |

Controls: click a piece then its destination (clone = 1 hex, jump = 2 hexes) ·
`E` toggles edit mode · `R` restarts · `Esc` quits.

---

## Running tests

```bash
python -m pytest tests/ -v
```

## Acknowledgements

- Minimax implementation adapted from [Darkshoxx](https://github.com/darkshoxx)
- Built with PyTorch
