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
# Position Generator - phase-based and tactical position generation
# ============================================================

class PositionGenerator:
    """
    Generates diverse board positions for curriculum training.

    Provides positions at specific game phases and with specific tactical
    characteristics via random playouts.
    """

    phases = {
        'opening':   (4, 10),   # Just started, few pieces
        'early_mid': (8, 18),   # Early development
        'mid_game':  (14, 28),  # Mid-game complexity
        'late_game': (24, 38),  # Board filling up
        'endgame':   (32, 49),  # Nearly full board
    }

    def generate_standard_start(self):
        """Generate canonical 4-corner starting position."""
        board = np.zeros((7, 7, 2), dtype=np.bool_)
        board[0, 0] = BLUE
        board[0, 6] = GREEN
        board[6, 0] = GREEN
        board[6, 6] = BLUE
        return board, True  # Blue starts

    def generate_random_start(self):
        """Generate random 2v2 starting position."""
        board = np.zeros((7, 7, 2), dtype=np.bool_)
        positions = random.sample(range(49), 4)
        colors = [BLUE, BLUE, GREEN, GREEN]
        random.shuffle(colors)
        for pos, color in zip(positions, colors):
            y, x = pos // 7, pos % 7
            board[y, x] = color
        return board, random.choice([True, False])

    def generate_phase_position(self, phase, balance=None):
        """
        Generate a position at a specific game phase.

        Args:
            phase: One of 'opening', 'early_mid', 'mid_game', 'late_game', 'endgame'
            balance: None for any balance, 'balanced', 'blue_ahead', or 'green_ahead'

        Returns:
            (board, turn) tuple
        """
        min_p, max_p = self.phases[phase]
        fallback = FallbackGenerator()

        for _ in range(500):
            num_moves = random.randint(2, 100)
            board, turn = fallback.generate_by_playout(num_moves)
            total = int(np.sum(board))

            if not (min_p <= total <= max_p):
                continue

            if balance is None:
                return board, turn

            blue = int(np.sum(board[:, :, 1]))
            green = int(np.sum(board[:, :, 0]))

            if balance == 'balanced' and abs(blue - green) <= 1:
                return board, turn
            elif balance == 'blue_ahead' and blue > green:
                return board, turn
            elif balance == 'green_ahead' and green > blue:
                return board, turn

        # Fallback: return any valid position in range (ignore balance constraint)
        for _ in range(200):
            board, turn = fallback.generate_by_playout(random.randint(2, 100))
            if min_p <= int(np.sum(board)) <= max_p:
                return board, turn

        return self.generate_standard_start()

    def generate_tactical_position(self, tactic):
        """
        Generate a position with specific tactical characteristics.

        Args:
            tactic: One of 'conversion_battle', 'piece_advantage',
                    'mobility_crisis', 'endgame_race'

        Returns:
            (board, turn) tuple
        """
        if tactic == 'conversion_battle':
            return self.generate_phase_position('mid_game')
        elif tactic == 'piece_advantage':
            return self.generate_phase_position('mid_game', balance='blue_ahead')
        elif tactic == 'mobility_crisis':
            return self.generate_phase_position('late_game')
        elif tactic == 'endgame_race':
            return self.generate_phase_position('endgame')
        else:
            return self.generate_standard_start()


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
    Samples training positions across curriculum stages.

    If bank exists: uses minimax game positions for realistic training data.
    If no bank: falls back to PositionGenerator for phase-based positions.

    Curriculum stages:
    - Stage 0 (0-300k):   random starts, build basic intuition
    - Stage 1 (300k-1M):  opening + early game positions
    - Stage 2 (1M-2M):    midgame complexity
    - Stage 3 (2M-4M):    late game and endgame
    - Stage 4 (4M+):      endgame mastery and tactical positions
    """

    def __init__(self, bank_path="positions/position_bank.npz"):
        self.bank = PositionBank(bank_path)
        self.fallback = FallbackGenerator()
        self.generator = PositionGenerator()

        if self.bank.loaded:
            print(f"[Curriculum] Loaded {len(self.bank.boards)} positions from bank")
            if len(self.bank._swing_indices) > 0:
                print(f"  Swing positions (>=3): {len(self.bank._swing_indices)}")
            if len(self.bank._big_swing_indices) > 0:
                print(f"  Big swing positions (>=5): {len(self.bank._big_swing_indices)}")
        else:
            print("[Curriculum] No position bank found, using position generator")
            print("  Generate bank with: python scripts/generate_position_bank.py")

        self.stages = [
            # Stage 0: Random starts - build basic intuition (0-300k)
            {
                'max_timestep': 300_000,
                'distribution': {
                    'standard_start': 0.2,
                    'random_start': 0.8,
                }
            },
            # Stage 1: Opening + early game (300k-1M)
            {
                'max_timestep': 1_000_000,
                'distribution': {
                    'standard_start': 0.3,
                    'random_start': 0.2,
                    'opening': 0.3,
                    'early_mid': 0.2,
                }
            },
            # Stage 2: Midgame complexity (1M-2M)
            {
                'max_timestep': 2_000_000,
                'distribution': {
                    'standard_start': 0.1,
                    'opening': 0.1,
                    'early_mid': 0.2,
                    'mid_game': 0.4,
                    'late_game': 0.2,
                }
            },
            # Stage 3: Late game and endgame (2M-4M)
            {
                'max_timestep': 4_000_000,
                'distribution': {
                    'mid_game': 0.2,
                    'late_game': 0.3,
                    'endgame': 0.3,
                    'tactical': 0.2,
                }
            },
            # Stage 4: Endgame mastery (4M+)
            {
                'max_timestep': float('inf'),
                'distribution': {
                    'mid_game': 0.1,
                    'late_game': 0.2,
                    'endgame': 0.4,
                    'tactical': 0.3,
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
            return self.generator.generate_standard_start()

        if position_type == 'random_start':
            return self.generator.generate_random_start()

        if position_type == 'tactical':
            tactic = random.choice(['conversion_battle', 'piece_advantage',
                                    'mobility_crisis', 'endgame_race'])
            return self.generator.generate_tactical_position(tactic)

        if position_type in ('opening', 'early_mid', 'mid_game', 'late_game', 'endgame'):
            return self.generator.generate_phase_position(position_type)

        # Legacy bank-based keys (kept for compatibility)
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

        return self.fallback.generate_by_playout(num_moves=random.randint(4, 30))


def sample_curriculum_position(timestep=0):
    """Quick function to sample a curriculum position."""
    curriculum = PositionCurriculum()
    return curriculum.sample_position(timestep)
