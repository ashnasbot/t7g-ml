"""
Value curriculum generation for AlphaZero MCTS training.

Generates supervised examples from MM-N vs MM-N self-play with uniform
policy targets and game-outcome value labels.  Policy targets are
intentionally uniform — the curriculum calibrates the value head on
strong-play positions without re-introducing a BC-style policy prior.
Policy loss is masked to replay-only samples during training (see
lib/training.py) so these uniform targets never touch the policy gradient.

Used by
-------
scripts/generate_value_curriculum.py  — standalone pre-generation
scripts/train_mcts.py                 — lazy generation during ladder advancement
"""
from __future__ import annotations

import multiprocessing

import numpy as np
from tqdm import tqdm

from lib.t7g import (
    new_board, apply_move, check_terminal,
    board_to_obs, action_masks, count_cells, find_best_move,
)


def _worker_curriculum_game(args: tuple) -> list:
    """
    Play one MM-depth vs MM-depth game.

    Returns a list of (obs, uniform_policy, value_target) tuples —
    one per position, no symmetry augmentation (applied at training time).
    """
    depth, = args
    board = new_board()
    turn = bool(np.random.randint(2))
    board_history: dict = {}
    game_positions: list = []
    move_count = 0
    winner = 0.0

    # 2-3 random opening moves for positional diversity (same as BC)
    for _ in range(np.random.randint(2, 4)):
        is_terminal, _ = check_terminal(board, turn)
        if is_terminal:
            break
        legal = np.where(action_masks(board, turn))[0]
        if len(legal) == 0:
            turn = not turn
            continue
        board = apply_move(board, int(np.random.choice(legal)), turn)
        turn = not turn

    while True:
        state_key = board.tobytes() + bytes([turn])
        board_history[state_key] = board_history.get(state_key, 0) + 1
        if board_history[state_key] >= 3:
            blue, green = count_cells(board)
            winner = 1.0 if blue > green else (-1.0 if green > blue else 0.0)
            break

        is_terminal, terminal_value = check_terminal(board, turn)
        if is_terminal:
            assert terminal_value is not None
            winner = terminal_value if turn else -terminal_value
            break

        legal = np.where(action_masks(board, turn))[0]
        if len(legal) == 0:
            turn = not turn
            continue

        game_positions.append((board_to_obs(board, turn), legal, bool(turn)))

        action = find_best_move(board.tobytes(), depth, turn)
        if action in (-1, 1225):
            action = int(np.random.choice(legal))

        board = apply_move(board, action, turn)
        turn = not turn
        move_count += 1

        if move_count > 200:
            blue, green = count_cells(board)
            winner = 1.0 if blue > green else (-1.0 if green > blue else 0.0)
            break

    examples = []
    for obs, legal, ex_turn in game_positions:
        value_target = float(winner if ex_turn else -winner)
        policy_target = np.zeros(1225, dtype=np.float32)
        policy_target[legal] = 1.0 / len(legal)
        examples.append((obs, policy_target, value_target))
    return examples


def generate_curriculum_data(depth: int, n_games: int) -> list:
    """
    Generate value curriculum examples from MM-depth vs MM-depth games.

    Returns a list of (obs, uniform_policy, value_target) tuples.
    No symmetry augmentation — applied at training time per batch.
    """
    num_workers = min(16, n_games)
    task_args = [(depth,)] * n_games
    examples: list = []
    pbar = tqdm(total=n_games, desc=f"MM-{depth} curriculum", unit="game")
    with multiprocessing.Pool(processes=num_workers) as pool:
        for game_examples in pool.imap_unordered(_worker_curriculum_game, task_args):
            examples.extend(game_examples)
            pbar.update(1)
            pbar.set_postfix(examples=len(examples))
    pbar.close()
    return examples
