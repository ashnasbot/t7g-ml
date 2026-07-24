"""Legal-action mask for the training loss: obs-derived mask must match the
game's own action_masks, including on D4-augmented obs."""
import numpy as np

from lib.t7g import (action_masks, apply_obs_symmetry, board_to_obs, new_board,
                     apply_move, SYMMETRY_INV_PERMS)
from lib.training import illegal_action_mask


def _random_boards(n=40, seed=7):
    rng = np.random.default_rng(seed)
    boards = []
    for _ in range(n):
        board = new_board()
        turn = bool(rng.integers(2))
        for _ in range(int(rng.integers(0, 60))):
            legal = np.flatnonzero(action_masks(board, turn))
            if len(legal) == 0:
                turn = not turn
                legal = np.flatnonzero(action_masks(board, turn))
                if len(legal) == 0:
                    break
            board = apply_move(board, int(rng.choice(legal)), turn)
            turn = not turn
        boards.append((board, turn))
    return boards


def test_mask_matches_action_masks():
    boards = _random_boards()
    obs = np.stack([board_to_obs(b, t) for b, t in boards])
    got = illegal_action_mask(obs)
    want = np.stack([action_masks(b, t) for b, t in boards])
    assert got.shape == (len(boards), 1225)
    assert (got == want).all()


def test_mask_consistent_under_symmetry():
    """Mask computed from rotated obs == permuted mask of the original board,
    using the same inv_perms the training loop applies to policy targets."""
    boards = _random_boards(n=16, seed=11)
    obs = np.stack([board_to_obs(b, t) for b, t in boards])
    base = np.stack([action_masks(b, t) for b, t in boards])
    for k in range(8):
        got = illegal_action_mask(apply_obs_symmetry(obs, k))
        assert (got == base[:, SYMMETRY_INV_PERMS[k]]).all(), f"symmetry {k}"


def test_masked_positions_have_legal_targets():
    """Every position with any legal move keeps at least one unmasked action."""
    boards = _random_boards(n=30, seed=3)
    obs = np.stack([board_to_obs(b, t) for b, t in boards])
    got = illegal_action_mask(obs)
    want_any = np.array([action_masks(b, t).any() for b, t in boards])
    assert (got.any(axis=1) == want_any).all()
