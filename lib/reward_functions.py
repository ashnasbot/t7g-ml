"""
Reward functions for the Microscope game.

This module provides various reward shaping strategies to improve learning.
The minimax opponent optimizes pure material count (player_cells - opponent_cells).
Reward functions should align with this objective.
"""
import numpy as np
from lib.t7g import count_cells, action_masks, BLUE, GREEN, CLEAR


def calc_reward_simple(board, as_blue=True):
    """
    Tactical reward function - encourages smart, winning play.

    Key improvements over pure material counting:
    - Balances material (35%), our mobility (40%), opponent restriction (25%)
    - Rewards limiting opponent's options (key to beating minimax)
    - Strong intermediate signals (no excessive clipping)
    - Encourages active, aggressive play

    This is the "simple" tactical foundation for beating strategic opponents.
    """
    new_blue, new_green = count_cells(board)

    if as_blue:
        player_cells = new_blue
        opponent_cells = new_green
    else:
        player_cells = new_green
        opponent_cells = new_blue

    # Get action masks for mobility calculation
    player_moves = action_masks(board, as_blue)
    opponent_moves = action_masks(board, not as_blue)
    player_can_move = np.any(player_moves)
    opponent_can_move = np.any(opponent_moves)

    empty = 49 - new_blue - new_green

    # Calculate final positions for terminal states
    if not player_can_move and not opponent_can_move:
        final_player = player_cells
        final_opponent = opponent_cells
    elif not player_can_move:
        final_player = player_cells
        final_opponent = opponent_cells + empty
    elif not opponent_can_move:
        final_player = player_cells + empty
        final_opponent = opponent_cells
    else:
        final_player = player_cells
        final_opponent = opponent_cells

    # Check terminal conditions
    terminated = (not player_can_move or not opponent_can_move or
                  player_cells == 0 or opponent_cells == 0)

    if terminated:
        # Terminal reward: strong win/loss signal
        if final_player > final_opponent:
            win_margin = (final_player - final_opponent) / 49.0
            # Big win bonus (1.0 to 2.0 based on dominance)
            reward = 1.0 + (1.0 * win_margin)
        elif final_player < final_opponent:
            loss_margin = (final_opponent - final_player) / 49.0
            # Big loss penalty (-1.0 to -2.0)
            reward = -1.0 - (1.0 * loss_margin)
        else:
            reward = 0.0  # Draw
        return reward, terminated

    # --- Intermediate (non-terminal) rewards ---
    # These need to be strong enough to guide learning in 80+ move games

    # 1. Material component (35% weight)
    # Having more pieces is good, but not everything
    material_diff = player_cells - opponent_cells
    material_reward = (material_diff / 49.0) * 0.35

    # 2. Our mobility component (40% weight)
    # MORE OPTIONS = BETTER POSITION
    # This is crucial for controlling the game
    player_mobility_count = np.sum(player_moves)
    max_possible_moves = 49  # Theoretical maximum
    mobility_reward = (player_mobility_count / max_possible_moves) * 0.40

    # 3. Opponent restriction component (25% weight)
    # LIMIT OPPONENT OPTIONS = WINNING STRATEGY
    # This is how you beat minimax - restrict their tree search
    opponent_mobility_count = np.sum(opponent_moves)
    opponent_mobility_ratio = opponent_mobility_count / max_possible_moves

    # Use QUADRATIC penalty to emphasize "killer moves"
    # Reducing opponent from 40→20 moves is okay (penalty -0.20)
    # Reducing opponent from 20→5 moves is GREAT (penalty -0.01)
    # This naturally rewards tactical squeezes without explicit tracking
    restriction_reward = -(opponent_mobility_ratio ** 2) * 0.25

    # 4. Efficiency component - small penalty per move to encourage quick wins
    # This prevents stalling and "farming" behavior in winning positions
    # Over 200 moves, this accumulates to -0.2 (small but meaningful)
    efficiency_penalty = -0.001

    # Combine all components
    reward = material_reward + mobility_reward + restriction_reward + efficiency_penalty

    # Light clipping to prevent extreme values, but keep signal strong
    # With 80-move games, we need rewards in [-1.5, 1.5] range for good gradients
    reward = np.clip(reward, -1.5, 1.5)

    return reward, terminated


def calc_reward_strategic(board, as_blue=True):
    """
    Strategic reward with multiple components.

    Components:
    1. Material (cell count)
    2. Mobility (number of legal moves)
    3. Center control (pieces in center)
    4. Territory control (connected regions)

    This encourages more sophisticated play.
    """
    new_blue, new_green = count_cells(board)

    if as_blue:
        player_cells = new_blue
        opponent_cells = new_green
        player_mask = BLUE
    else:
        player_cells = new_green
        opponent_cells = new_blue
        player_mask = GREEN

    # Check for game end
    player_moves = action_masks(board, as_blue)
    opponent_moves = action_masks(board, not as_blue)
    player_can_move = np.any(player_moves)
    opponent_can_move = np.any(opponent_moves)

    empty = 49 - new_blue - new_green

    # Terminal state handling
    if not player_can_move and not opponent_can_move:
        final_player = player_cells
        final_opponent = opponent_cells
    elif not player_can_move:
        final_player = player_cells
        final_opponent = opponent_cells + empty
    elif not opponent_can_move:
        final_player = player_cells + empty
        final_opponent = opponent_cells
    else:
        final_player = player_cells
        final_opponent = opponent_cells

    terminated = (not player_can_move or not opponent_can_move or
                  player_cells == 0 or opponent_cells == 0)

    if terminated:
        # Terminal reward
        if final_player > final_opponent:
            win_margin = (final_player - final_opponent) / 49.0
            reward = 1.0 + (0.5 * win_margin)
        elif final_player < final_opponent:
            loss_margin = (final_opponent - final_player) / 49.0
            reward = -1.0 - (0.5 * loss_margin)
        else:
            reward = 0.0
        return reward, terminated

    # --- Strategic Components for Intermediate States ---

    # 1. Material component (cell count) - 40% weight
    material_diff = (player_cells - opponent_cells) / 49.0
    material_reward = material_diff * 0.4

    # 2. Mobility component (legal moves) - 30% weight
    player_mobility = np.sum(player_moves)
    opponent_mobility = np.sum(opponent_moves)
    if player_mobility + opponent_mobility > 0:
        mobility_diff = (player_mobility - opponent_mobility) / (player_mobility + opponent_mobility)
        mobility_reward = mobility_diff * 0.3
    else:
        mobility_reward = 0.0

    # 3. Center control component - 20% weight
    # Center squares are more valuable (3x3 center area)
    center_positions = [(2, 2), (2, 3), (2, 4), (3, 2), (3, 3), (3, 4), (4, 2), (4, 3), (4, 4)]
    player_center = sum(1 for y, x in center_positions if np.array_equal(board[y, x], player_mask))
    opponent_center = sum(1 for y, x in center_positions if np.array_equal(board[y, x], BLUE if as_blue else GREEN))
    center_diff = (player_center - opponent_center) / 9.0
    center_reward = center_diff * 0.2

    # 4. Potential (empty squares we can reach) - 10% weight
    # Count empty squares adjacent to our pieces
    player_potential = 0
    opponent_potential = 0
    for y in range(7):
        for x in range(7):
            if not np.any(board[y, x]):  # Empty square
                # Check adjacent squares
                for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < 7 and 0 <= nx < 7:
                        if np.array_equal(board[ny, nx], player_mask):
                            player_potential += 1
                            break
                for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < 7 and 0 <= nx < 7:
                        if np.array_equal(board[ny, nx], BLUE if not as_blue else GREEN):
                            opponent_potential += 1
                            break
    if player_potential + opponent_potential > 0:
        potential_diff = (player_potential - opponent_potential) / (player_potential + opponent_potential)
        potential_reward = potential_diff * 0.1
    else:
        potential_reward = 0.0

    # Combine all components
    reward = material_reward + mobility_reward + center_reward + potential_reward
    reward = np.clip(reward, -0.8, 0.8)  # Clip intermediate rewards

    return reward, terminated


def calc_reward_dense(board, as_blue=True):
    """
    Dense reward function - provides frequent, clear feedback for initial learning.

    Designed for self-play foundation building:
    - Material (50%) - learn that pieces matter
    - Our mobility (40%) - learn to maintain options
    - Opponent mobility penalty (10%) - learn aggressive play

    Stronger signals than before (removed weak clipping).
    Good for stage 1 where agent learns basic gameplay.
    """
    new_blue, new_green = count_cells(board)

    if as_blue:
        player_cells = new_blue
        opponent_cells = new_green
    else:
        player_cells = new_green
        opponent_cells = new_blue

    # Check for game end
    player_moves = action_masks(board, as_blue)
    opponent_moves = action_masks(board, not as_blue)
    player_can_move = np.any(player_moves)
    opponent_can_move = np.any(opponent_moves)

    empty = 49 - new_blue - new_green

    # Terminal handling
    if not player_can_move and not opponent_can_move:
        final_player = player_cells
        final_opponent = opponent_cells
    elif not player_can_move:
        final_player = player_cells
        final_opponent = opponent_cells + empty
    elif not opponent_can_move:
        final_player = player_cells + empty
        final_opponent = opponent_cells
    else:
        final_player = player_cells
        final_opponent = opponent_cells

    terminated = (not player_can_move or not opponent_can_move or
                  player_cells == 0 or opponent_cells == 0)

    if terminated:
        # Strong terminal signal for wins/losses
        if final_player > final_opponent:
            win_margin = (final_player - final_opponent) / 49.0
            reward = 1.0 + (1.0 * win_margin)  # 1.0 to 2.0
        elif final_player < final_opponent:
            loss_margin = (final_opponent - final_player) / 49.0
            reward = -1.0 - (1.0 * loss_margin)  # -1.0 to -2.0
        else:
            reward = 0.0
        return reward, terminated

    # Material (50%) - balanced importance
    material = (player_cells - opponent_cells) / 49.0 * 0.5

    # Our mobility (40%) - maintain options
    player_mobility = np.sum(player_moves)
    our_mobility = (player_mobility / 49.0) * 0.4

    # Opponent mobility penalty (10%) - learn aggression and killer moves
    opponent_mobility = np.sum(opponent_moves)
    opponent_ratio = opponent_mobility / 49.0
    # Quadratic to emphasize restricting opponent heavily
    opponent_penalty = -(opponent_ratio ** 2) * 0.1

    # Small efficiency penalty - encourages closing out games
    efficiency_penalty = -0.001

    reward = material + our_mobility + opponent_penalty + efficiency_penalty

    # Lighter clipping - keep signals strong for long episodes
    reward = np.clip(reward, -1.5, 1.5)

    return reward, terminated


def calc_reward_aggressive(board, as_blue=True):
    """
    Aggressive reward aligned with minimax objective: maximize material.

    The C minimax evaluates boards with pure material count (player - opponent).
    This reward mirrors that, plus frontier contact to encourage pushing into
    the opponent rather than spreading away.

    Key differences from previous reward functions:
    - NO own-mobility reward (was teaching agent to spread out and avoid contact)
    - Heavy material weight (70%) - this is what actually wins
    - Frontier contact (20%) - pieces adjacent to opponents = conversion potential
    - Opponent restriction (10%) - fewer opponent moves = closer to ending game

    A clone next to 3 enemy pieces = +1 (your new piece) + 3 converted = +4 swing.
    The agent needs to learn to seek these conversion opportunities.
    """
    new_blue, new_green = count_cells(board)

    if as_blue:
        player_cells = new_blue
        opponent_cells = new_green
        player_mask = BLUE
        opponent_mask = GREEN
    else:
        player_cells = new_green
        opponent_cells = new_blue
        player_mask = GREEN
        opponent_mask = BLUE

    # Check for game end
    player_moves = action_masks(board, as_blue)
    opponent_moves = action_masks(board, not as_blue)
    player_can_move = np.any(player_moves)
    opponent_can_move = np.any(opponent_moves)

    empty = 49 - new_blue - new_green

    # Terminal handling
    if not player_can_move and not opponent_can_move:
        final_player = player_cells
        final_opponent = opponent_cells
    elif not player_can_move:
        final_player = player_cells
        final_opponent = opponent_cells + empty
    elif not opponent_can_move:
        final_player = player_cells + empty
        final_opponent = opponent_cells
    else:
        final_player = player_cells
        final_opponent = opponent_cells

    terminated = (not player_can_move or not opponent_can_move or
                  player_cells == 0 or opponent_cells == 0)

    if terminated:
        if final_player > final_opponent:
            win_margin = (final_player - final_opponent) / 49.0
            reward = 1.0 + win_margin  # 1.0 to 2.0
        elif final_player < final_opponent:
            loss_margin = (final_opponent - final_player) / 49.0
            reward = -1.0 - loss_margin  # -1.0 to -2.0
        else:
            reward = 0.0
        return reward, terminated

    # --- Intermediate rewards ---

    # 1. Material difference (70%) - the core objective, same as minimax eval
    material_reward = (player_cells - opponent_cells) / 49.0 * 0.7

    # 2. Frontier contact (20%) - our pieces touching opponent pieces
    # This directly measures conversion potential: more frontier = more chances
    # to convert enemy pieces on next move. Encourages pushing INTO the opponent.
    frontier_count = 0
    for y in range(7):
        for x in range(7):
            if np.array_equal(board[y, x], player_mask):
                # Count adjacent opponent pieces
                for dy in range(-1, 2):
                    for dx in range(-1, 2):
                        if dy == 0 and dx == 0:
                            continue
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < 7 and 0 <= nx < 7:
                            if np.array_equal(board[ny, nx], opponent_mask):
                                frontier_count += 1
                                break  # Count each of our pieces once
    # Normalize: max frontier is ~25 pieces on the contact line
    frontier_reward = (frontier_count / 25.0) * 0.2

    # 3. Opponent restriction (10%) - fewer opponent moves = game ending soon
    opponent_mobility = np.sum(opponent_moves)
    # Linear penalty - simpler, more direct signal
    restriction_reward = -(opponent_mobility / 49.0) * 0.1

    reward = material_reward + frontier_reward + restriction_reward

    reward = np.clip(reward, -1.5, 1.5)

    return reward, terminated


# Reward function registry for easy switching
REWARD_FUNCTIONS = {
    'simple': calc_reward_simple,
    'strategic': calc_reward_strategic,
    'dense': calc_reward_dense,
    'aggressive': calc_reward_aggressive,
}


def get_reward_function(name='simple'):
    """
    Get a reward function by name.

    Args:
        name: One of 'original', 'simple', 'strategic', 'dense'

    Returns:
        Reward function
    """
    if name not in REWARD_FUNCTIONS:
        raise ValueError(
            f"Unknown reward function: {name}. "
            f"Choose from {list(REWARD_FUNCTIONS.keys())}"
        )
    return REWARD_FUNCTIONS[name]
