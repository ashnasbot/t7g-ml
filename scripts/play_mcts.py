"""
Play or evaluate a trained MCTS agent against minimax.

Usage:
    # Evaluate against minimax depth-2 (20 games)
    python scripts/play_mcts.py --checkpoint models/mcts/iter_0050.pt --depth 2

    # Watch a single game with board visualization
    python scripts/play_mcts.py --checkpoint models/mcts/final.pt --watch

    # Quick eval with fewer simulations
    python scripts/play_mcts.py --checkpoint models/mcts/final.pt --simulations 50
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.dual_network import DualHeadNetwork
from lib.mcts import MCTS
from lib.t7g import new_board, apply_move, check_terminal, board_to_obs, find_best_move, count_cells, show_board, action_masks, action_to_move

import torch


def play_game(network, minimax_depth=2, num_simulations=100, verbose=False):
    """
    Play one game: MCTS (Blue) vs Minimax (Green).

    Returns:
        result: +1 Blue win, -1 Green win, 0 draw
        move_count: number of moves played
    """
    mcts = MCTS(network, num_simulations=num_simulations,
                dirichlet_epsilon=0.0)
    board = new_board()
    turn = True
    move_count = 0

    if verbose:
        print("\nStarting position:")
        show_board(board)

    while True:
        is_terminal, terminal_value = check_terminal(board, turn)
        if is_terminal:
            result = terminal_value if turn else -terminal_value
            break

        if turn:
            # MCTS agent (Blue)
            action_probs = mcts.search(board, turn)
            action = mcts.select_action(action_probs, temperature=0)
            if verbose:
                from_x, from_y, to_x, to_y, jump = action_to_move(action)
                move_type = "jump" if jump else "clone"
                print(f"\nMove {move_count + 1} - Blue (MCTS): "
                      f"({from_x},{from_y})->({to_x},{to_y}) [{move_type}]")
        else:
            # Minimax opponent (Green)
            board_bytes = board.tobytes()
            action = find_best_move(board_bytes, minimax_depth, False)
            if action in [-1, 1225]:
                result = 1.0  # Green can't move, Blue wins
                if verbose:
                    print(f"\nGreen has no moves - Blue wins!")
                break
            if verbose:
                from_x, from_y, to_x, to_y, jump = action_to_move(action)
                move_type = "jump" if jump else "clone"
                print(f"\nMove {move_count + 1} - Green (Minimax-{minimax_depth}): "
                      f"({from_x},{from_y})->({to_x},{to_y}) [{move_type}]")

        board = apply_move(board, action, turn)
        turn = not turn
        move_count += 1

        if verbose:
            show_board(board)
            blue, green = count_cells(board)
            print(f"  Score: Blue={blue} Green={green}")

        if move_count > 500:
            blue, green = count_cells(board)
            result = 1.0 if blue > green else (-1.0 if green > blue else 0.0)
            break

    if verbose:
        blue, green = count_cells(board)
        print(f"\nGame over after {move_count} moves!")
        print(f"Final score: Blue={blue} Green={green}")
        if result > 0:
            print("Result: Blue (MCTS) WINS!")
        elif result < 0:
            print(f"Result: Green (Minimax-{minimax_depth}) wins")
        else:
            print("Result: DRAW")

    return result, move_count


def evaluate(network, minimax_depth, num_games, num_simulations):
    """Run evaluation games and print summary."""
    wins, losses, draws = 0, 0, 0
    total_moves = 0

    print(f"\nEvaluating: MCTS ({num_simulations} sims) vs Minimax depth-{minimax_depth}")
    print(f"Playing {num_games} games...\n")

    for i in range(num_games):
        result, moves = play_game(
            network, minimax_depth=minimax_depth,
            num_simulations=num_simulations
        )
        total_moves += moves

        if result > 0:
            wins += 1
            marker = "W"
        elif result < 0:
            losses += 1
            marker = "L"
        else:
            draws += 1
            marker = "D"

        print(f"  Game {i + 1:3d}/{num_games}: {marker} ({moves} moves) "
              f"| Running: {wins}W-{losses}L-{draws}D "
              f"({wins / (i + 1):.0%})")

    print(f"\n{'=' * 50}")
    print(f"Results: {wins}W - {losses}L - {draws}D")
    print(f"Win rate: {wins / num_games:.1%}")
    print(f"Avg moves: {total_moves / num_games:.1f}")
    print(f"{'=' * 50}")


def main():
    parser = argparse.ArgumentParser(description="Play/evaluate MCTS agent")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--depth", type=int, default=2,
                        help="Minimax opponent depth (default: 2)")
    parser.add_argument("--games", type=int, default=20,
                        help="Number of evaluation games (default: 20)")
    parser.add_argument("--simulations", type=int, default=100,
                        help="MCTS simulations per move (default: 100)")
    parser.add_argument("--watch", action="store_true",
                        help="Watch a single game with board output")
    args = parser.parse_args()

    # Load network
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = DualHeadNetwork(num_actions=1225).to(device)

    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, weights_only=False)
    network.load_state_dict(checkpoint['network'])
    network.eval()
    print(f"Device: {device}")

    if args.watch:
        # Watch a single game with verbose output
        play_game(network, minimax_depth=args.depth,
                  num_simulations=args.simulations, verbose=True)
    else:
        # Run evaluation
        evaluate(network, minimax_depth=args.depth,
                 num_games=args.games, num_simulations=args.simulations)


if __name__ == "__main__":
    main()
