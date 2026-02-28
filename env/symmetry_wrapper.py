"""
Symmetry augmentation wrapper for 7x7 board game.

Exploits rotational and reflectional symmetry to augment training data 8x.
This is a key technique used in AlphaGo/AlphaZero for data efficiency.

The 7x7 board has 8 symmetries (dihedral group D4):
- 4 rotations: 0°, 90°, 180°, 270°
- 4 reflections: horizontal, vertical, diagonal, anti-diagonal

For each episode, we randomly pick one symmetry and apply it consistently to:
- Observations (board state)
- Actions (piece positions and move directions)
- Action masks
"""
import gymnasium as gym
import numpy as np


class SymmetryAugmentationWrapper(gym.Wrapper):
    """
    Applies random symmetry transformations to board and actions.

    This exploits the fact that Microscope gameplay is symmetric -
    rotating/flipping the board doesn't change optimal strategy.

    Augmentation is fixed per episode (chosen at reset), ensuring consistency.
    """

    # Define all 8 symmetries as transformation functions
    SYMMETRIES = {
        0: "identity",
        1: "rotate_90",
        2: "rotate_180",
        3: "rotate_270",
        4: "flip_horizontal",
        5: "flip_vertical",
        6: "flip_diagonal",      # main diagonal (top-left to bottom-right)
        7: "flip_anti_diagonal"  # anti-diagonal (top-right to bottom-left)
    }

    def __init__(self, env, augment_prob=1.0):
        """
        Args:
            env: Base environment to wrap
            augment_prob: Probability of applying augmentation (1.0 = always)
        """
        super().__init__(env)
        self.augment_prob = augment_prob
        self.current_symmetry = 0  # Identity by default

    def reset(self, **kwargs):
        """Reset and choose random symmetry for this episode"""
        obs, info = self.env.reset(**kwargs)

        # Choose random symmetry for this episode
        if np.random.random() < self.augment_prob:
            self.current_symmetry = np.random.randint(0, 8)
        else:
            self.current_symmetry = 0  # Identity

        # Transform observation
        obs = self._transform_observation(obs, self.current_symmetry)

        return obs, info

    def step(self, action):
        """Transform action to original space, step, transform observation back"""
        # Transform action from augmented space to original space
        original_action = self._inverse_transform_action(action, self.current_symmetry)

        # Step in original environment
        obs, reward, terminated, truncated, info = self.env.step(original_action)

        # Transform observation to augmented space
        obs = self._transform_observation(obs, self.current_symmetry)

        return obs, reward, terminated, truncated, info

    def action_masks(self):
        """Get action masks and transform them to augmented space"""
        masks = self.env.action_masks()
        return self._transform_action_masks(masks, self.current_symmetry)

    # ==================== Observation Transformations ====================

    def _transform_observation(self, obs, sym_id):
        """Transform observation according to symmetry"""
        if sym_id == 0:
            return obs  # Identity

        # obs shape: (7, 7, 4)
        # Channels: [Green, Blue, Turn, SelectedPiece]

        # Apply spatial transformation to each channel
        transformed = np.zeros_like(obs)
        for c in range(4):
            transformed[:, :, c] = self._transform_board(obs[:, :, c], sym_id)

        return transformed

    def _transform_board(self, board, sym_id):
        """Transform a single 7x7 board according to symmetry"""
        if sym_id == 0:
            return board
        elif sym_id == 1:  # Rotate 90° clockwise
            return np.rot90(board, k=-1)
        elif sym_id == 2:  # Rotate 180°
            return np.rot90(board, k=2)
        elif sym_id == 3:  # Rotate 270° clockwise (90° counter-clockwise)
            return np.rot90(board, k=1)
        elif sym_id == 4:  # Flip horizontal
            return np.fliplr(board)
        elif sym_id == 5:  # Flip vertical
            return np.flipud(board)
        elif sym_id == 6:  # Flip diagonal (transpose)
            return np.transpose(board)
        elif sym_id == 7:  # Flip anti-diagonal
            return np.rot90(np.transpose(board), k=2)
        else:
            raise ValueError(f"Unknown symmetry id: {sym_id}")

    # ==================== Action Transformations ====================

    def _inverse_transform_action(self, action, sym_id):
        """Transform action from augmented space back to original space"""
        if sym_id == 0:
            return action

        # Check which stage we're in
        if hasattr(self.env, 'action_stage'):
            stage = self.env.action_stage
        else:
            # If no stage info, assume stage 0 (piece selection)
            stage = 0

        if stage == 0:
            # Stage 0: piece selection (position 0-48)
            return self._inverse_transform_position(action, sym_id)
        else:
            # Stage 1: move selection (relative move 0-24)
            return self._inverse_transform_move(action, sym_id)

    def _inverse_transform_position(self, position, sym_id):
        """Transform position action (0-48) from augmented to original"""
        if sym_id == 0:
            return position

        # Decode to (x, y)
        x, y = position % 7, position // 7

        # Apply inverse transformation
        orig_x, orig_y = self._inverse_transform_coords(x, y, sym_id)

        # Encode back to position
        return orig_y * 7 + orig_x

    def _inverse_transform_move(self, move_idx, sym_id):
        """Transform relative move action (0-24) from augmented to original"""
        if sym_id == 0:
            return move_idx

        # Decode move (0-24) to (dx, dy)
        dx = (move_idx % 5) - 2  # -2, -1, 0, 1, 2
        dy = (move_idx // 5) - 2

        # Apply inverse transformation to direction
        orig_dx, orig_dy = self._inverse_transform_direction(dx, dy, sym_id)

        # Encode back to move index
        return (orig_dy + 2) * 5 + (orig_dx + 2)

    def _inverse_transform_coords(self, x, y, sym_id):
        """Apply inverse coordinate transformation"""
        # Inverse transformations (to go from augmented back to original)
        if sym_id == 1:  # Rotate 90° clockwise -> inverse is 270° (counter-clockwise)
            return y, 6 - x
        elif sym_id == 2:  # Rotate 180° -> inverse is 180°
            return 6 - x, 6 - y
        elif sym_id == 3:  # Rotate 270° -> inverse is 90°
            return 6 - y, x
        elif sym_id == 4:  # Flip horizontal -> inverse is flip horizontal
            return 6 - x, y
        elif sym_id == 5:  # Flip vertical -> inverse is flip vertical
            return x, 6 - y
        elif sym_id == 6:  # Flip diagonal -> inverse is flip diagonal
            return y, x
        elif sym_id == 7:  # Flip anti-diagonal -> inverse is flip anti-diagonal
            return 6 - y, 6 - x
        else:
            raise ValueError(f"Unknown symmetry id: {sym_id}")

    def _inverse_transform_direction(self, dx, dy, sym_id):
        """Apply inverse transformation to move direction"""
        # For rotations, direction transforms differently than position
        if sym_id == 1:  # Rotate 90° clockwise
            return dy, -dx
        elif sym_id == 2:  # Rotate 180°
            return -dx, -dy
        elif sym_id == 3:  # Rotate 270°
            return -dy, dx
        elif sym_id == 4:  # Flip horizontal
            return -dx, dy
        elif sym_id == 5:  # Flip vertical
            return dx, -dy
        elif sym_id == 6:  # Flip diagonal
            return dy, dx
        elif sym_id == 7:  # Flip anti-diagonal
            return -dy, -dx
        else:
            raise ValueError(f"Unknown symmetry id: {sym_id}")

    # ==================== Action Mask Transformations ====================

    def _transform_action_masks(self, masks, sym_id):
        """Transform action masks to augmented space"""
        if sym_id == 0:
            return masks

        # Check stage
        if hasattr(self.env, 'action_stage'):
            stage = self.env.action_stage
        else:
            stage = 0

        if stage == 0:
            # Stage 0: transform position masks
            return self._transform_position_masks(masks, sym_id)
        else:
            # Stage 1: transform move masks
            return self._transform_move_masks(masks, sym_id)

    def _transform_position_masks(self, masks, sym_id):
        """Transform position masks (0-48) to augmented space"""
        if sym_id == 0:
            return masks

        transformed = np.zeros_like(masks)
        for pos in range(49):
            # Get original position
            x, y = pos % 7, pos // 7
            # Transform to augmented position
            aug_x, aug_y = self._transform_coords(x, y, sym_id)
            aug_pos = aug_y * 7 + aug_x
            # Copy mask value
            transformed[aug_pos] = masks[pos]

        return transformed

    def _transform_move_masks(self, masks, sym_id):
        """Transform move masks (0-24) to augmented space"""
        if sym_id == 0:
            return masks

        transformed = np.zeros_like(masks)
        for move_idx in range(25):
            # Decode original move
            dx = (move_idx % 5) - 2
            dy = (move_idx // 5) - 2
            # Transform direction
            aug_dx, aug_dy = self._transform_direction(dx, dy, sym_id)
            # Encode augmented move
            aug_move = (aug_dy + 2) * 5 + (aug_dx + 2)
            # Copy mask value (handle out of bounds)
            if 0 <= aug_move < 49:
                transformed[aug_move] = masks[move_idx]

        # Preserve padding (positions 25-48 should stay False)
        transformed[25:] = False

        return transformed

    def _transform_coords(self, x, y, sym_id):
        """Apply forward coordinate transformation"""
        if sym_id == 1:  # Rotate 90° clockwise
            return 6 - y, x
        elif sym_id == 2:  # Rotate 180°
            return 6 - x, 6 - y
        elif sym_id == 3:  # Rotate 270°
            return y, 6 - x
        elif sym_id == 4:  # Flip horizontal
            return 6 - x, y
        elif sym_id == 5:  # Flip vertical
            return x, 6 - y
        elif sym_id == 6:  # Flip diagonal
            return y, x
        elif sym_id == 7:  # Flip anti-diagonal
            return 6 - y, 6 - x
        else:
            raise ValueError(f"Unknown symmetry id: {sym_id}")

    def _transform_direction(self, dx, dy, sym_id):
        """Apply forward transformation to move direction"""
        if sym_id == 1:  # Rotate 90° clockwise
            return -dy, dx
        elif sym_id == 2:  # Rotate 180°
            return -dx, -dy
        elif sym_id == 3:  # Rotate 270°
            return dy, -dx
        elif sym_id == 4:  # Flip horizontal
            return -dx, dy
        elif sym_id == 5:  # Flip vertical
            return dx, -dy
        elif sym_id == 6:  # Flip diagonal
            return dy, dx
        elif sym_id == 7:  # Flip anti-diagonal
            return -dy, -dx
        else:
            raise ValueError(f"Unknown symmetry id: {sym_id}")
