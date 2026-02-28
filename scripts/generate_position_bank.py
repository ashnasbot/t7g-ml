"""
Generate a bank of training positions from minimax vs minimax games.

Captures realistic aggressive positions by having the minimax engine play
itself and recording positions where big material swings happen.

Output: positions/position_bank.npz containing:
  - boards: (N, 7, 7, 2) bool array of board states
  - turns: (N,) bool array (True=Blue's turn)
  - tags: (N,) string array - position type tag
  - swings: (N,) int array - material swing that happened on this move

Usage:
    python scripts/generate_position_bank.py
    python scripts/generate_position_bank.py --games 200 --depth 2
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.t7g import (
    find_best_move, count_cells, action_to_move, action_masks,
    BLUE, GREEN, CLEAR
)


def apply_move_inplace(board, action, turn):
    """Apply a move to board (modifies in place). Returns number of conversions."""
    player_cell = BLUE if turn else GREEN
    opponent_cell = GREEN if turn else BLUE

    from_x, from_y, to_x, to_y, jump = action_to_move(action)

    if jump:
        board[from_y, from_x] = CLEAR
    board[to_y, to_x] = player_cell

    conversions = 0
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            ny, nx = to_y + dy, to_x + dx
            if 0 <= ny < 7 and 0 <= nx < 7:
                if np.array_equal(board[ny, nx], opponent_cell):
                    board[ny, nx] = player_cell
                    conversions += 1

    return conversions


def play_minimax_game(depth_blue=1, depth_green=1):
    """
    Play a full minimax vs minimax game and record all positions.

    Returns:
        positions: list of (board_copy, turn, move_number, swing, conversions, tag)
    """
    board = np.zeros((7, 7, 2), dtype=np.bool_)
    board[0, 0] = BLUE
    board[0, 6] = GREEN
    board[6, 0] = GREEN
    board[6, 6] = BLUE

    turn = True
    positions = []
    move_number = 0

    while True:
        blue_count, green_count = count_cells(board)

        # Check if current player can move
        masks = action_masks(board, turn)
        if not np.any(masks):
            break

        # Record pre-move position
        board_copy = board.copy()

        # Get minimax move
        depth = depth_blue if turn else depth_green
        action = find_best_move(board.tobytes(), depth, turn)

        if action in [-1, 1225]:
            break

        # Apply move and track swing
        pre_blue, pre_green = blue_count, green_count
        conversions = apply_move_inplace(board, action, turn)
        post_blue, post_green = count_cells(board)

        # Material swing from Blue's perspective
        pre_diff = pre_blue - pre_green
        post_diff = post_blue - post_green
        swing = abs(post_diff - pre_diff)

        # Tag the position
        total_pieces = pre_blue + pre_green
        if total_pieces <= 8:
            phase = "opening"
        elif total_pieces <= 20:
            phase = "midgame"
        elif total_pieces <= 35:
            phase = "late_midgame"
        else:
            phase = "endgame"

        if swing >= 5:
            tag = "big_swing"
        elif swing >= 3:
            tag = "swing"
        elif conversions >= 3:
            tag = "mass_convert"
        elif conversions >= 2:
            tag = "convert"
        else:
            tag = phase

        positions.append({
            'board': board_copy,
            'turn': turn,
            'move_number': move_number,
            'swing': swing,
            'conversions': conversions,
            'tag': tag,
            'blue_count': pre_blue,
            'green_count': pre_green,
        })

        turn = not turn
        move_number += 1

        if move_number > 200:
            break

    return positions


def generate_bank(num_games=100, depth_blue=1, depth_green=1):
    """Generate position bank from multiple minimax games."""
    all_positions = []
    tag_counts = {}

    for game_idx in range(num_games):
        positions = play_minimax_game(depth_blue, depth_green)
        all_positions.extend(positions)

        for p in positions:
            tag = p['tag']
            tag_counts[tag] = tag_counts.get(tag, 0) + 1

        if (game_idx + 1) % 25 == 0:
            print(f"  Games: {game_idx + 1}/{num_games}, "
                  f"positions: {len(all_positions)}")

    return all_positions, tag_counts


def main():
    parser = argparse.ArgumentParser(description="Generate position bank")
    parser.add_argument("--games", type=int, default=200,
                        help="Number of games to play (default: 200)")
    parser.add_argument("--depth", type=int, default=1,
                        help="Minimax depth for both players (default: 1)")
    parser.add_argument("--depth-blue", type=int, default=None,
                        help="Minimax depth for blue (overrides --depth)")
    parser.add_argument("--depth-green", type=int, default=None,
                        help="Minimax depth for green (overrides --depth)")
    parser.add_argument("--output", type=str, default="positions/position_bank.npz",
                        help="Output file path")
    args = parser.parse_args()

    depth_blue = args.depth_blue or args.depth
    depth_green = args.depth_green or args.depth

    print("=" * 60)
    print("Position Bank Generator")
    print("=" * 60)
    print(f"Games: {args.games}")
    print(f"Depth: Blue={depth_blue}, Green={depth_green}")
    print(f"Output: {args.output}")
    print("=" * 60)

    start = time.time()
    positions, tag_counts = generate_bank(
        args.games, depth_blue, depth_green
    )
    elapsed = time.time() - start

    print(f"\nGenerated {len(positions)} positions in {elapsed:.1f}s")
    print("\nTag distribution:")
    for tag, count in sorted(tag_counts.items(), key=lambda x: -x[1]):
        print(f"  {tag:20s}: {count:5d} ({100*count/len(positions):.1f}%)")

    # Convert to arrays
    boards = np.array([p['board'] for p in positions])
    turns = np.array([p['turn'] for p in positions])
    swings = np.array([p['swing'] for p in positions])
    tags = np.array([p['tag'] for p in positions])
    move_numbers = np.array([p['move_number'] for p in positions])

    # Save
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    np.savez_compressed(
        args.output,
        boards=boards,
        turns=turns,
        swings=swings,
        tags=tags,
        move_numbers=move_numbers,
    )

    file_size = os.path.getsize(args.output) / 1024
    print(f"\nSaved to {args.output} ({file_size:.1f} KB)")

    # Print some stats about swing positions
    big_swings = swings >= 5
    medium_swings = swings >= 3
    print(f"\nSwing positions:")
    print(f"  swing >= 5: {np.sum(big_swings)} ({100*np.mean(big_swings):.1f}%)")
    print(f"  swing >= 3: {np.sum(medium_swings)} ({100*np.mean(medium_swings):.1f}%)")


if __name__ == "__main__":
    main()
