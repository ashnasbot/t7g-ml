"""
Play Microscope in the real game (ScummVM) using the MCTS model.

The agent plays as Blue. Launch ScummVM and navigate to the Microscope
board before running this script — or it will attempt to launch ScummVM
and load save slot 18 automatically.

Usage:
    python scripts/play_real.py
    python scripts/play_real.py --checkpoint models/mcts/iter_0100.pt
    python scripts/play_real.py --simulations 200 --debug
"""
from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, ".")

from env.env_t7g import MicroscopeRealEnv          # noqa: E402
from lib.device_utils import load_compiled_network  # noqa: E402
from lib.mcgs import MCGS                           # noqa: E402
from lib.t7g import action_masks                    # noqa: E402

_BUNDLED = pathlib.Path(__file__).parent.parent / "lib" / "best.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description="Play Microscope vs the real game using MCTS")
    parser.add_argument("--checkpoint", type=str, default=str(_BUNDLED),
                        help="Path to model checkpoint (.pt)")
    parser.add_argument("--simulations", type=int, default=100,
                        help="MCTS simulations per move (default: 100)")
    parser.add_argument("--debug", action="store_true",
                        help="Print move details and piece counts")
    args = parser.parse_args()

    print(f"Loading model: {args.checkpoint}")
    network = load_compiled_network(args.checkpoint)
    mcts = MCGS(network, num_simulations=args.simulations)
    env = MicroscopeRealEnv(debug=args.debug)

    game_num = 0
    while True:
        game_num += 1
        print(f"\n=== Game {game_num}  ({args.simulations} sims/move) ===")
        board, _ = env.reset()
        mcts.clear()
        done = False

        while not done:
            if not action_masks(board, True).any():
                print("No legal moves — passing")
                break

            action_probs = mcts.search(board, True)
            action = mcts.select_action(action_probs, board=board, turn=True, temperature=0)
            board, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

        if reward > 0:
            print("Result: Win")
        elif reward < 0:
            print("Result: Loss")
        else:
            print("Result: Draw")

        if input("\nPlay again? [Y/n] ").strip().lower() == 'n':
            break

    env.close()


if __name__ == "__main__":
    main()
