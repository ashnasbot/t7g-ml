"""
Play Microscope in the real game (ScummVM) using the MCTS model or micro4t.

The agent plays as Blue. Launch ScummVM and navigate to the Microscope
board before running this script - or it will attempt to launch ScummVM
and load save slot 18 automatically.

Usage:
    python scripts/play_real.py
    python scripts/play_real.py --checkpoint models/mcts/iter_0100.pt
    python scripts/play_real.py --simulations 200 --debug
    python scripts/play_real.py --engine micro4t
    python scripts/play_real.py --engine micro4t --budget 1000
    python scripts/play_real.py --log stauf_log.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, ".")

from env.env_t7g import MicroscopeRealEnv          # noqa: E402
from lib.device_utils import load_compiled_network  # noqa: E402
from lib.mcgs import MCGS                           # noqa: E402
from lib.t7g import action_masks, draw_board, find_best_move  # noqa: E402

_BUNDLED = pathlib.Path(__file__).parent.parent / "lib" / "best.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description="Play Microscope vs the real game using MCTS")
    parser.add_argument("--checkpoint", type=str, default=str(_BUNDLED),
                        help="Path to model checkpoint (.pt)")
    parser.add_argument("--simulations", type=int, default=100,
                        help="MCTS simulations per move (default: 100)")
    parser.add_argument("--engine", choices=["mcts", "micro4t"], default="mcts",
                        help="Agent to use (default: mcts)")
    parser.add_argument("--budget", type=int, default=1000,
                        help="micro4t time budget in ms (default: 1000)")
    parser.add_argument("--line", type=str, default=None, metavar="FILE",
                        help="JSON file of Blue actions from find_stauf_line.py; "
                             "plays the sequence then falls back to --engine")
    parser.add_argument("--log", type=str, default=None, metavar="FILE",
                        help="Append (board_after_blue, board_after_stauf) JSON records to FILE "
                             "for Stauf move identification")
    parser.add_argument("--debug", action="store_true",
                        help="Print move details and piece counts")
    args = parser.parse_args()

    mcts = None
    if args.engine == "mcts":
        print(f"Loading model: {args.checkpoint}")
        network = load_compiled_network(args.checkpoint)
        mcts = MCGS(network, num_simulations=args.simulations)
        agent_label = f"MCTS ({args.simulations} sims/move)"
    else:
        agent_label = f"micro4t ({args.budget}ms/move)"

    line: list[int] = []
    if args.line:
        with open(args.line) as f:
            line = json.load(f)
        print(f"Loaded line: {len(line)} moves from {args.line}")

    print(f"Agent: {agent_label}")
    env = MicroscopeRealEnv(debug=args.debug)

    game_num = 0
    while True:
        game_num += 1
        print(f"\n=== Game {game_num}  [{agent_label}] ===")
        board, _ = env.reset()
        if mcts:
            mcts.clear()
        done = False
        move_idx = 0
        stauf_move_num = 0  # cumulative Stauf moves this game (for log)

        while not done:
            if not action_masks(board, True).any():
                print("No legal moves - passing")
                break

            if args.debug:
                draw_board(board)

            masks = action_masks(board, True)
            if move_idx < len(line) and masks[line[move_idx]]:
                action = line[move_idx]
                print(f"  Line move {move_idx + 1}/{len(line)}: action={action}")
            else:
                if move_idx < len(line):
                    print(f"  Line move {move_idx + 1} invalid on current board "
                          f"- falling back to {agent_label}")
                    move_idx = len(line)  # exhaust line, stay on fallback
                if args.engine == "micro4t":
                    action = find_best_move(board.tobytes(), args.budget, True, "micro4t")
                    if action in (-1, 1225):
                        print("micro4t returned no move - passing")
                        break
                else:
                    action_probs = mcts.search(board, True)
                    action = mcts.select_action(action_probs, board=board,
                                                turn=True, temperature=0)

            board, reward, terminated, truncated, info = env.step(action)
            if not info.get('invalid'):
                move_idx += 1
                if args.log and not (terminated or truncated):
                    # board is now board_after_stauf (Stauf has moved).
                    # after_blue comes from env (screen-captured after Blue's animation
                    # settles) so _infer_stauf_action can reliably reconstruct Stauf's move.
                    after_blue = info['after_blue']
                    record = {
                        "game":        game_num,
                        "round":       move_idx,        # 1-based Blue move index
                        "stauf_move":  stauf_move_num,  # 0-based before this Stauf move
                        "after_blue":  after_blue.tolist(),
                        "after_stauf": board.tolist(),
                    }
                    with open(args.log, "a") as lf:
                        lf.write(json.dumps(record) + "\n")
                stauf_move_num += 1
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
