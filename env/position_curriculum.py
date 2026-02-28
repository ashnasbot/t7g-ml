"""
Position Curriculum for Microscope Training

Uses pre-generated minimax game positions for realistic training data.
The position bank (positions/position_bank.npz) contains positions from
minimax vs minimax games, tagged by type (swing, convert, midgame, etc).

If the bank doesn't exist, falls back to random playouts from standard starts.

Generate the bank with:
    python scripts/generate_position_bank.py
"""
import os
import numpy as np
import random
from lib.t7g import BLUE, GREEN, CLEAR, action_masks, action_to_move


# ============================================================
# Position Bank - loads pre-generated minimax positions
# ============================================================

class PositionBank:
    """
    Loads and samples from pre-generated minimax game positions.
    Positions are tagged by type for weighted sampling.
    """

    def __init__(self, bank_path="positions/position_bank.npz"):
        self.loaded = False
        self.boards = None
        self.turns = None
        self.tags = None
        self.swings = None

        # Tag -> index arrays for fast sampling
        self._tag_indices = {}
        self._swing_indices = None  # indices where swing >= 3
        self._big_swing_indices = None  # indices where swing >= 5

        if os.path.exists(bank_path):
            self._load(bank_path)

    def _load(self, path):
        data = np.load(path, allow_pickle=True)
        self.boards = data['boards']
        self.turns = data['turns']
        self.tags = data['tags']
        self.swings = data['swings']
        self.loaded = True

        # Build tag indices
        for tag in np.unique(self.tags):
            self._tag_indices[tag] = np.where(self.tags == tag)[0]

        # Build swing indices
        self._swing_indices = np.where(self.swings >= 3)[0]
        self._big_swing_indices = np.where(self.swings >= 5)[0]

    def sample(self, tag=None, min_swing=0):
        """
        Sample a position from the bank.

        Args:
            tag: specific tag to sample from, or None for any
            min_swing: minimum material swing value

        Returns:
            (board, turn) or None if bank not loaded
        """
        if not self.loaded:
            return None

        if min_swing >= 5 and len(self._big_swing_indices) > 0:
            idx = np.random.choice(self._big_swing_indices)
        elif min_swing >= 3 and len(self._swing_indices) > 0:
            idx = np.random.choice(self._swing_indices)
        elif tag and tag in self._tag_indices and len(self._tag_indices[tag]) > 0:
            idx = np.random.choice(self._tag_indices[tag])
        else:
            idx = np.random.randint(len(self.boards))

        board = self.boards[idx].copy()
        turn = bool(self.turns[idx])

        # Validate the position has legal moves
        if not np.any(action_masks(board, turn)):
            turn = not turn
            if not np.any(action_masks(board, turn)):
                return None

        return board, turn

    def sample_any(self):
        """Sample any position from bank."""
        return self.sample()

    def sample_swing(self):
        """Sample a position where a big material swing happened."""
        return self.sample(min_swing=3)

    def sample_big_swing(self):
        """Sample a position with swing >= 5."""
        return self.sample(min_swing=5)

    def sample_midgame(self):
        """Sample a midgame position."""
        return self.sample(tag='midgame')

    def sample_endgame(self):
        """Sample a late/endgame position."""
        return self.sample(tag='endgame') or self.sample(tag='late_midgame')


# ============================================================
# Fallback generator (when bank not available)
# ============================================================

class FallbackGenerator:
    """Generates positions via random playouts when bank isn't available."""

    def generate_standard_start(self):
        board = np.zeros((7, 7, 2), dtype=np.bool_)
        board[0, 0] = BLUE
        board[0, 6] = GREEN
        board[6, 0] = GREEN
        board[6, 6] = BLUE
        return board, True

    def generate_by_playout(self, num_moves=None):
        board, turn = self.generate_standard_start()
        if num_moves is None:
            num_moves = random.randint(4, 40)

        for _ in range(num_moves):
            masks = action_masks(board, turn)
            legal = np.where(masks)[0]
            if len(legal) == 0:
                break

            # Bias toward clones
            clone_moves = []
            jump_moves = []
            for action in legal:
                _, _, _, _, jump = action_to_move(action)
                if jump:
                    jump_moves.append(action)
                else:
                    clone_moves.append(action)

            if clone_moves and (not jump_moves or random.random() < 0.7):
                action = random.choice(clone_moves)
            elif jump_moves:
                action = random.choice(jump_moves)
            else:
                action = random.choice(legal)

            from_x, from_y, to_x, to_y, jump = action_to_move(action)
            player_cell = BLUE if turn else GREEN
            opponent_cell = GREEN if turn else BLUE

            if jump:
                board[from_y, from_x] = CLEAR
            board[to_y, to_x] = player_cell

            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    ny, nx = to_y + dy, to_x + dx
                    if 0 <= ny < 7 and 0 <= nx < 7:
                        if np.array_equal(board[ny, nx], opponent_cell):
                            board[ny, nx] = player_cell

            turn = not turn

        if not np.any(action_masks(board, turn)):
            turn = not turn
            if not np.any(action_masks(board, turn)):
                return self.generate_by_playout(num_moves)

        return board, turn


# ============================================================
# Curriculum
# ============================================================

class PositionCurriculum:
    """
    Samples training positions from minimax position bank.

    If bank exists: samples swing positions, midgame positions, etc.
    If no bank: falls back to random playout generation.

    Curriculum stages:
    - Early (0-500k): mostly standard starts + general bank positions
    - Mid (500k-1.5M): swing positions + midgame positions
    - Late (1.5M+): heavy on big swings and endgame positions
    """

    def __init__(self, bank_path="positions/position_bank.npz"):
        self.bank = PositionBank(bank_path)
        self.fallback = FallbackGenerator()

        if self.bank.loaded:
            print(f"[Curriculum] Loaded {len(self.bank.boards)} positions from bank")
            if len(self.bank._swing_indices) > 0:
                print(f"  Swing positions (>=3): {len(self.bank._swing_indices)}")
            if len(self.bank._big_swing_indices) > 0:
                print(f"  Big swing positions (>=5): {len(self.bank._big_swing_indices)}")
        else:
            print("[Curriculum] No position bank found, using random playouts")
            print("  Generate with: python scripts/generate_position_bank.py")

        self.stages = [
            # Stage 0: Standard starts + general positions (0-500k)
            {
                'max_timestep': 500_000,
                'distribution': {
                    'standard_start': 0.3,
                    'bank_any': 0.4,
                    'bank_midgame': 0.3,
                }
            },
            # Stage 1: Swing positions + contested midgame (500k-1.5M)
            {
                'max_timestep': 1_500_000,
                'distribution': {
                    'standard_start': 0.10,
                    'bank_any': 0.20,
                    'bank_midgame': 0.25,
                    'bank_swing': 0.30,
                    'bank_big_swing': 0.15,
                }
            },
            # Stage 2: Heavy on swings and endgame (1.5M+)
            {
                'max_timestep': float('inf'),
                'distribution': {
                    'standard_start': 0.10,
                    'bank_any': 0.10,
                    'bank_midgame': 0.20,
                    'bank_swing': 0.25,
                    'bank_big_swing': 0.20,
                    'bank_endgame': 0.15,
                }
            },
        ]

    def get_current_stage(self, timestep):
        for stage in self.stages:
            if timestep < stage['max_timestep']:
                return stage
        return self.stages[-1]

    def sample_position(self, timestep=0):
        stage = self.get_current_stage(timestep)
        position_type = random.choices(
            list(stage['distribution'].keys()),
            weights=list(stage['distribution'].values())
        )[0]

        return self._generate_position(position_type)

    def _generate_position(self, position_type):
        if position_type == 'standard_start':
            return self.fallback.generate_standard_start()

        # Try bank first
        if self.bank.loaded:
            result = None
            if position_type == 'bank_any':
                result = self.bank.sample_any()
            elif position_type == 'bank_swing':
                result = self.bank.sample_swing()
            elif position_type == 'bank_big_swing':
                result = self.bank.sample_big_swing()
            elif position_type == 'bank_midgame':
                result = self.bank.sample_midgame()
            elif position_type == 'bank_endgame':
                result = self.bank.sample_endgame()

            if result is not None:
                return result

        # Fallback to random playouts
        if position_type in ('bank_swing', 'bank_big_swing'):
            return self.fallback.generate_by_playout(num_moves=random.randint(10, 30))
        elif position_type == 'bank_endgame':
            return self.fallback.generate_by_playout(num_moves=random.randint(25, 45))
        else:
            return self.fallback.generate_by_playout(num_moves=random.randint(4, 25))


def sample_curriculum_position(timestep=0):
    """Quick function to sample a curriculum position."""
    curriculum = PositionCurriculum()
    return curriculum.sample_position(timestep)
