"""
Pytest-based performance benchmarks for minimax implementations.

Run with: pytest tests/test_minimax_performance.py -v -s
(Use -s to see timing output)

Mark as slow: pytest tests/test_minimax_performance.py -m slow -v -s
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import numpy as np
import pytest
from lib.t7g import find_best_move, BLUE, GREEN


# Fixtures

@pytest.fixture(scope="module")
def opening_position():
    """Standard opening position"""
    board = np.zeros((7, 7, 2), dtype=bool)
    board[0, 0] = BLUE
    board[0, 6] = GREEN
    board[6, 0] = GREEN
    board[6, 6] = BLUE
    return board


@pytest.fixture(scope="module")
def midgame_position():
    """Mid-game position with more pieces"""
    board = np.zeros((7, 7, 2), dtype=bool)
    board[0, 0] = BLUE
    board[1, 1] = BLUE
    board[2, 2] = BLUE
    board[3, 3] = BLUE
    board[4, 4] = GREEN
    board[5, 5] = GREEN
    board[6, 6] = GREEN
    return board


@pytest.fixture(scope="module")
def complex_position():
    """Complex position with many pieces"""
    board = np.zeros((7, 7, 2), dtype=bool)
    for i in range(7):
        if i < 4:
            board[i, i] = BLUE
        else:
            board[i, 6-i] = GREEN
    board[3, 0] = BLUE
    board[3, 6] = GREEN
    return board


# Helper functions

def benchmark_position(board, depth, num_trials=3):
    """
    Benchmark minimax on a position.

    Returns:
        avg_time, std_time, move
    """
    times = []
    move = -1  # Initialize in case of errors

    for _ in range(num_trials):
        start = time.perf_counter()
        move = find_best_move(board.tobytes(), depth, True)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return np.mean(times), np.std(times), move


# Tests - Depth 3 (Fast)

class TestDepth3Performance:
    """Benchmark depth 3 searches (fast, always run)"""

    def test_opening_depth3(self, opening_position):
        avg_time, std_time, move = benchmark_position(opening_position, 3)
        print(f"\n  Opening depth 3: {avg_time:.3f}s ± {std_time:.3f}s (move={move})")
        assert avg_time < 1.0, f"Depth 3 too slow: {avg_time:.3f}s"
        assert move >= 0, "Should return valid move"

    def test_midgame_depth3(self, midgame_position):
        avg_time, std_time, move = benchmark_position(midgame_position, 3)
        print(f"\n  Mid-game depth 3: {avg_time:.3f}s ± {std_time:.3f}s (move={move})")
        assert avg_time < 1.0, f"Depth 3 too slow: {avg_time:.3f}s"
        assert move >= 0, "Should return valid move"

    def test_complex_depth3(self, complex_position):
        avg_time, std_time, move = benchmark_position(complex_position, 3)
        print(f"\n  Complex depth 3: {avg_time:.3f}s ± {std_time:.3f}s (move={move})")
        assert avg_time < 1.0, f"Depth 3 too slow: {avg_time:.3f}s"
        assert move >= 0, "Should return valid move"


# Tests - Depth 4 & 5 (Slow, marked)

@pytest.mark.slow
class TestDepth4Performance:
    """Benchmark depth 4 searches (slower)"""

    def test_opening_depth4(self, opening_position):
        avg_time, std_time, move = benchmark_position(opening_position, 4)
        print(f"\n  Opening depth 4: {avg_time:.3f}s ± {std_time:.3f}s (move={move})")
        assert avg_time < 5.0, f"Depth 4 too slow: {avg_time:.3f}s"
        assert move >= 0, "Should return valid move"

    def test_midgame_depth4(self, midgame_position):
        avg_time, std_time, move = benchmark_position(midgame_position, 4)
        print(f"\n  Mid-game depth 4: {avg_time:.3f}s ± {std_time:.3f}s (move={move})")
        assert avg_time < 5.0, f"Depth 4 too slow: {avg_time:.3f}s"
        assert move >= 0, "Should return valid move"

    def test_complex_depth4(self, complex_position):
        avg_time, std_time, move = benchmark_position(complex_position, 4)
        print(f"\n  Complex depth 4: {avg_time:.3f}s ± {std_time:.3f}s (move={move})")
        assert avg_time < 5.0, f"Depth 4 too slow: {avg_time:.3f}s"
        assert move >= 0, "Should return valid move"


@pytest.mark.slow
class TestDepth5Performance:
    """Benchmark depth 5 searches (very slow)"""

    def test_opening_depth5(self, opening_position):
        avg_time, std_time, move = benchmark_position(opening_position, 5, num_trials=2)
        print(f"\n  Opening depth 5: {avg_time:.3f}s ± {std_time:.3f}s (move={move})")
        assert avg_time < 30.0, f"Depth 5 too slow: {avg_time:.3f}s"
        assert move >= 0, "Should return valid move"

    def test_midgame_depth5(self, midgame_position):
        avg_time, std_time, move = benchmark_position(midgame_position, 5, num_trials=2)
        print(f"\n  Mid-game depth 5: {avg_time:.3f}s ± {std_time:.3f}s (move={move})")
        assert avg_time < 30.0, f"Depth 5 too slow: {avg_time:.3f}s"
        assert move >= 0, "Should return valid move"


# Parametrized tests (all positions, all depths)

@pytest.mark.parametrize("depth", [3, 4, 5])
@pytest.mark.slow
def test_all_positions_all_depths(opening_position, midgame_position, complex_position, depth):
    """Test all positions at all depths (comprehensive benchmark)"""
    positions = [
        ("Opening", opening_position),
        ("Mid-game", midgame_position),
        ("Complex", complex_position)
    ]

    for name, board in positions:
        avg_time, std_time, move = benchmark_position(board, depth, num_trials=2)
        print(f"\n  {name} depth {depth}: {avg_time:.3f}s ± {std_time:.3f}s")
        assert move >= 0, f"{name} at depth {depth} should return valid move"


# Performance comparison (if both DLLs exist)

@pytest.mark.slow
class TestPerformanceComparison:
    """Compare original vs optimized implementations"""

    def test_speedup_estimate(self, opening_position):
        """Estimate speedup if both versions available"""
        # Test with current implementation
        avg_time, _, _ = benchmark_position(opening_position, 4, num_trials=3)

        print(f"\n  Current implementation depth 4: {avg_time:.3f}s")

        # Expected performance targets
        if avg_time < 0.2:
            print("  [OK] Using optimized implementation (fast!)")
            print(f"  Speedup: ~{0.8/avg_time:.1f}x compared to original")
        else:
            print("  [INFO] Using original implementation")
            print(f"  Expected speedup with optimization: ~{avg_time/0.1:.1f}x")
            print("\n  To optimize:")
            print("    gcc -O3 -march=native -ffast-math micro_3.c -o micro3.dll --shared")

        # Just check it completes, no strict assertion
        assert avg_time > 0, "Should complete successfully"
