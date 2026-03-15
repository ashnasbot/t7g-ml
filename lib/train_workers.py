"""
Pure game-logic helpers for AlphaZero self-play training.

All functions here are stateless (no module-level globals) and take their
dependencies as arguments.  They are imported by the worker entry-point
functions in ``scripts/train_mcts.py``, which own the module-level globals
required for multiprocessing under Windows' ``spawn`` start method.
"""
import time

import numpy as np

from lib.mcgs import MCGS  # noqa: F401 (imported for type hints)
from lib.t7g import (
    new_board, apply_move, check_terminal,
    board_to_obs, action_masks, count_cells, find_best_move,
)


# ---------------------------------------------------------------------------
# Self-play
# ---------------------------------------------------------------------------

def self_play_game(mcts: MCGS):
    """
    Play one game via MCTS self-play, collecting training examples.

    Returns
    -------
    training_examples : list of (obs, policy_target, value_target)
    winner            : +1.0 Blue / -1.0 Green / 0.0 draw (Blue perspective)
    move_count        : number of half-moves played
    elapsed           : wall time in seconds
    truncated         : True if the hard 200-move cap triggered
    legal_move_counts : per-position branching factor samples
    """
    board = new_board()
    # Randomise who moves first so training data is not biased toward whichever
    # colour has the first-mover advantage.
    turn = bool(np.random.randint(2))
    examples = []
    move_count = 0
    truncated = False
    board_history: dict = {}
    legal_move_counts: list = []
    game_start = time.time()

    while True:
        # 3-fold repetition → resolve by cell count
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

        masks = action_masks(board, turn)
        legal_count = int(masks.sum())
        if legal_count == 0:
            mcts.advance_tree(1225)
            turn = not turn
            continue
        legal_move_counts.append(legal_count)

        action_probs = mcts.search(board, turn)
        obs = board_to_obs(board, turn)
        examples.append((obs, action_probs, turn))

        action = mcts.select_action(action_probs, temperature=0.0)

        board = apply_move(board, action, turn)
        mcts.advance_tree(action)
        turn = not turn
        move_count += 1

        if move_count > 200:
            blue, green = count_cells(board)
            winner = 1.0 if blue > green else (-1.0 if green > blue else 0.0)
            truncated = True
            break

    training_examples = []
    for obs, policy_target, example_turn in examples:
        value_target = winner if example_turn else -winner
        training_examples.append((obs, policy_target, value_target))

    elapsed = time.time() - game_start
    return training_examples, winner, move_count, elapsed, truncated, legal_move_counts


# ---------------------------------------------------------------------------
# Evaluation vs minimax
# ---------------------------------------------------------------------------

def play_eval_game(
    mcts: MCGS,
    minimax_depth: int,
    noise: float,
    engine: str,
    vary_depth: bool,
    mcts_is_blue: bool,
) -> float:
    """
    Play one evaluation game (MCTS vs minimax/stauf).

    Returns +1 win / -1 loss / 0 draw from the MCTS agent's perspective.
    """
    board = new_board()
    mcts.root = None
    turn = True  # Blue moves first (eval games always start standard)
    board_history: dict = {}
    move_count = 0

    while True:
        state_key = board.tobytes() + bytes([turn])
        board_history[state_key] = board_history.get(state_key, 0) + 1
        if board_history[state_key] >= 3:
            blue, green = count_cells(board)
            blue_result = 1.0 if blue > green else (-1.0 if green > blue else 0.0)
            break

        is_terminal, terminal_value = check_terminal(board, turn)
        if is_terminal:
            assert terminal_value is not None
            blue_result = terminal_value if turn else -terminal_value
            break

        mcts_turn = (turn == mcts_is_blue)

        if mcts_turn:
            if not np.any(action_masks(board, turn)):
                mcts.advance_tree(1225)
                turn = not turn
                continue
            action_probs = mcts.search(board, turn)
            action = mcts.select_action(action_probs, temperature=0)
            mcts.advance_tree(action)
        else:
            legal = np.where(action_masks(board, turn))[0]
            if len(legal) == 0:
                turn = not turn
                continue
            if np.random.random() < noise:
                action = int(np.random.choice(legal))
            else:
                depth = int(np.random.choice([4, minimax_depth])) if vary_depth else minimax_depth
                stauf_mc = int(np.random.randint(0, 3)) if engine == 'stauf' else -1
                action = find_best_move(board.tobytes(), depth, turn, engine, stauf_mc)
                if action in (-1, 1225):
                    turn = not turn
                    continue

        board = apply_move(board, action, turn)
        turn = not turn
        move_count += 1

        if move_count > 200:
            blue, green = count_cells(board)
            blue_result = 1.0 if blue > green else (-1.0 if green > blue else 0.0)
            break

    return blue_result if mcts_is_blue else -blue_result


# ---------------------------------------------------------------------------
# Gate: network vs network
# ---------------------------------------------------------------------------

def play_net_vs_net_game(mcts_new: MCGS, mcts_best: MCGS, new_is_blue: bool) -> float:
    """
    Play one gate game between two MCTS agents.

    Returns +1 if *mcts_new* wins, -1 if it loses, 0 for a draw.
    Starting colour is randomised to neutralise first-mover advantage.
    """
    board = new_board()
    mcts_new.root = None
    mcts_best.root = None
    turn = bool(np.random.randint(2))
    board_history: dict = {}
    move_count = 0

    while True:
        state_key = board.tobytes() + bytes([turn])
        board_history[state_key] = board_history.get(state_key, 0) + 1
        if board_history[state_key] >= 3:
            blue, green = count_cells(board)
            blue_result = 1.0 if blue > green else (-1.0 if green > blue else 0.0)
            break

        is_terminal, terminal_value = check_terminal(board, turn)
        if is_terminal:
            assert terminal_value is not None
            blue_result = terminal_value if turn else -terminal_value
            break

        new_turn = (turn == new_is_blue)
        mcts_active = mcts_new if new_turn else mcts_best
        mcts_passive = mcts_best if new_turn else mcts_new

        if not np.any(action_masks(board, turn)):
            mcts_active.advance_tree(1225)
            mcts_passive.advance_tree(1225)
            turn = not turn
            continue

        action_probs = mcts_active.search(board, turn)
        action = mcts_active.select_action(action_probs, temperature=0)
        mcts_active.advance_tree(action)
        mcts_passive.advance_tree(action)

        board = apply_move(board, action, turn)
        turn = not turn
        move_count += 1

        if move_count > 200:
            blue, green = count_cells(board)
            blue_result = 1.0 if blue > green else (-1.0 if green > blue else 0.0)
            break

    return blue_result if new_is_blue else -blue_result
