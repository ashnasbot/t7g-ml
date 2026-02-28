"""
Tests for position curriculum generation and wrapper.
"""
import numpy as np
from env.position_curriculum import PositionGenerator, PositionCurriculum
from env.env_virt import MicroscopeEnv
from env.position_curriculum_wrapper import PositionCurriculumWrapper
from lib.t7g import action_masks


def test_standard_start():
    """Test standard starting position generation"""
    gen = PositionGenerator()
    board, turn = gen.generate_standard_start()

    # Check 4 pieces in corners
    assert np.sum(board) == 4
    assert board[0, 0, 1]  # Blue top-left
    assert board[6, 6, 1]  # Blue bottom-right
    assert board[0, 6, 0]  # Green top-right
    assert board[6, 0, 0]  # Green bottom-left
    assert turn  # Blue starts

    print("[OK] Standard start generation works")


def test_random_start():
    """Test random starting position generation"""
    gen = PositionGenerator()
    board, turn = gen.generate_random_start()

    # Should have 4 pieces
    assert np.sum(board) == 4

    # Should have 2 blue and 2 green
    blue_count = np.sum(board[:, :, 1])
    green_count = np.sum(board[:, :, 0])
    assert blue_count == 2
    assert green_count == 2

    print("[OK] Random start generation works")


def test_phase_positions():
    """Test generation of positions at different game phases"""
    gen = PositionGenerator()

    for phase in ['opening', 'early_mid', 'mid_game', 'late_game', 'endgame']:
        board, turn = gen.generate_phase_position(phase)

        total_pieces = np.sum(board)
        min_pieces, max_pieces = gen.phases[phase]

        # Check piece count is in expected range
        assert min_pieces <= total_pieces <= max_pieces, \
            f"{phase}: {total_pieces} pieces not in range [{min_pieces}, {max_pieces}]"

        # Check current player can move
        assert np.any(action_masks(board, turn)), \
            f"{phase}: Current player has no legal moves"

        print(f"[OK] {phase:12s} position: {total_pieces:2d} pieces, turn={'blue' if turn else 'green'}")


def test_balanced_positions():
    """Test balanced piece distributions"""
    gen = PositionGenerator()

    for _ in range(10):
        board, turn = gen.generate_phase_position('mid_game', balance='balanced')

        blue_count = np.sum(board[:, :, 1])
        green_count = np.sum(board[:, :, 0])

        # Balanced should be within 1 piece
        assert abs(blue_count - green_count) <= 1, \
            f"Balanced position has {blue_count} blue vs {green_count} green"

    print("[OK] Balanced positions work")


def test_imbalanced_positions():
    """Test imbalanced piece distributions"""
    gen = PositionGenerator()

    # Test blue ahead
    board, turn = gen.generate_phase_position('mid_game', balance='blue_ahead')
    blue_count = np.sum(board[:, :, 1])
    green_count = np.sum(board[:, :, 0])
    assert blue_count > green_count, "Blue should be ahead"

    # Test green ahead
    board, turn = gen.generate_phase_position('mid_game', balance='green_ahead')
    blue_count = np.sum(board[:, :, 1])
    green_count = np.sum(board[:, :, 0])
    assert green_count > blue_count, "Green should be ahead"

    print("[OK] Imbalanced positions work")


def test_tactical_positions():
    """Test tactical position generation"""
    gen = PositionGenerator()

    tactics = ['conversion_battle', 'piece_advantage', 'mobility_crisis', 'endgame_race']

    for tactic in tactics:
        board, turn = gen.generate_tactical_position(tactic)

        # Basic validation
        assert np.sum(board) >= 4, f"{tactic}: Too few pieces"
        assert np.any(action_masks(board, turn)), f"{tactic}: No legal moves"

        print(f"[OK] Tactical position '{tactic}' generates successfully")


def test_curriculum_progression():
    """Test curriculum stage progression"""
    curriculum = PositionCurriculum()

    # Test stage 0 (early training)
    stage = curriculum.get_current_stage(timestep=100_000)
    assert 'standard_start' in stage['distribution']
    assert stage['distribution']['random_start'] > 0.5  # Mostly random starts

    # Test stage 2 (mid training)
    stage = curriculum.get_current_stage(timestep=1_500_000)
    assert 'mid_game' in stage['distribution']

    # Test stage 4 (late training)
    stage = curriculum.get_current_stage(timestep=5_000_000)
    assert 'endgame' in stage['distribution']
    assert 'tactical' in stage['distribution']

    print("[OK] Curriculum progression works")


def test_curriculum_sampling():
    """Test sampling positions from curriculum"""
    curriculum = PositionCurriculum()

    # Sample positions at different timesteps
    for timestep in [0, 500_000, 1_500_000, 3_000_000, 5_000_000]:
        board, turn = curriculum.sample_position(timestep)

        # Basic validation
        assert board.shape == (7, 7, 2)
        assert np.sum(board) >= 4  # At least 4 pieces
        assert np.any(action_masks(board, turn))  # Can move

    print("[OK] Curriculum sampling works")


def test_wrapper_integration():
    """Test PositionCurriculumWrapper with environment"""
    # Create wrapped environment
    env = MicroscopeEnv()
    env = PositionCurriculumWrapper(env)

    # Test reset with different timesteps
    for timestep in [0, 1_000_000, 3_000_000]:
        obs, info = env.reset(options={'timestep': timestep})

        # Validate observation
        assert obs.shape == (7, 7, 4)
        assert np.any(obs[:, :, 0:2])  # Has pieces

        # Validate environment can step
        masks = env.action_masks()
        assert np.any(masks), "No valid actions after curriculum reset"

    print("[OK] Wrapper integration works")


def test_diversity():
    """Test that curriculum generates diverse positions"""
    curriculum = PositionCurriculum()

    # Generate 20 positions and check they're not all the same
    positions = []
    for _ in range(20):
        board, turn = curriculum.sample_position(timestep=2_000_000)
        positions.append(board.tobytes())

    # Should have at least 15 unique positions out of 20
    unique_positions = len(set(positions))
    assert unique_positions >= 15, f"Only {unique_positions}/20 positions were unique"

    print(f"[OK] Diversity check: {unique_positions}/20 unique positions")


if __name__ == "__main__":
    print("\n=== Testing Position Curriculum ===\n")

    test_standard_start()
    test_random_start()
    test_phase_positions()
    test_balanced_positions()
    test_imbalanced_positions()
    test_tactical_positions()
    test_curriculum_progression()
    test_curriculum_sampling()
    test_wrapper_integration()
    test_diversity()

    print("\n=== All tests passed! ===\n")
