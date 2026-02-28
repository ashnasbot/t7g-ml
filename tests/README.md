# Test Suite

Comprehensive test coverage for the Microscope ML training environment.

## Test Files

### Environment & Game Logic Tests

**[test_environment.py](test_environment.py)** - MicroscopeEnv environment tests
- ✅ Observation shape and format (7x7x3 with turn indicator)
- ✅ Turn alternation and consistency
- ✅ Episode lifecycle (reset, step, termination)
- ✅ Action masking integration
- ✅ Reward function configuration
- ✅ Observation consistency after moves
- ✅ Truncation at turn limit

**[test_game_logic.py](test_game_logic.py)** - Core game logic tests
- ✅ Action encoding/decoding (action_to_move)
- ✅ Move validation (is_action_valid)
- ✅ Capture mechanics (3x3 radius)
- ✅ Jump vs non-jump moves
- ✅ Action mask generation
- ✅ Board state queries (count_cells)
- ✅ Edge cases (corners, center, full board)

### Core Algorithm Tests

**[test_minimax_correctness.py](test_minimax_correctness.py)** - Unit tests for minimax algorithm
- ✅ Returns valid moves
- ✅ Handles stuck positions (no moves available)
- ✅ Takes winning moves when available
- ✅ Avoids capture
- ✅ Prefers captures over safe moves
- ✅ Works at all depths (1-3)
- ✅ Handles symmetric positions
- ✅ Works with edge cases (full board, single piece)
- ✅ Alternates between blue/green perspectives
- ✅ Deterministic (same position → same move)
- ✅ Full game simulation

**[test_minimax_performance.py](test_minimax_performance.py)** - Performance benchmarks
- Depth 3 benchmarks (fast, always run)
- Depth 4-5 benchmarks (slower, marked with `@pytest.mark.slow`)
- Tests on opening, mid-game, and complex positions
- Performance comparison and speedup estimates

### Environment Tests

**[test_minimax_wrapper.py](test_minimax_wrapper.py)** - Minimax opponent wrapper tests
- ✅ Wrapper initialization
- ✅ Opponent makes moves after agent
- ✅ Games complete successfully
- ✅ Different depths work (1-3)
- ✅ Multiple games consistency

**[test_debug_rendering.py](test_debug_rendering.py)** - Debug rendering behavior
- ✅ debug=False hides per-move output
- ✅ debug=True shows per-move output
- ✅ Final board always shown regardless of debug flag
- ✅ Wrapper respects render settings

## Running Tests

### Run all fast tests
```bash
pytest tests/ -v
```

### Run specific test file
```bash
pytest tests/test_minimax_correctness.py -v
```

### Run with output visible (good for debugging)
```bash
pytest tests/test_minimax_correctness.py -v -s
```

### Run including slow tests
```bash
pytest tests/ -v --run-slow
# or
pytest tests/ -v -m slow
```

### Run only slow tests
```bash
pytest tests/ -v -m slow
```

### Run with coverage report
```bash
pytest tests/ --cov=env --cov=util --cov-report=html
```

## Test Statistics

```bash
# Count tests
pytest tests/ --collect-only | grep "test session starts" -A 1

# Summary by file
pytest tests/ --collect-only -q
```

## Test Organization

### Markers

- `@pytest.mark.slow` - Tests that take >1 second
- No marker - Fast tests that always run

### Test Categories

1. **Correctness** - Algorithm behavior is correct
2. **Performance** - Speed benchmarks
3. **Integration** - Components work together
4. **Edge Cases** - Boundary conditions handled

## Writing New Tests

### Template for correctness test
```python
def test_feature_description():
    """One-line description of what this tests"""
    # Setup
    board = setup_board(
        blue_positions=[(0, 0)],
        green_positions=[(6, 6)]
    )

    # Execute
    result = function_under_test(board)

    # Assert
    assert result == expected, "Why this should be true"
```

### Template for performance test
```python
@pytest.mark.slow
def test_performance_feature():
    """Performance benchmark for specific scenario"""
    import time

    start = time.perf_counter()
    result = function_under_test()
    elapsed = time.perf_counter() - start

    assert elapsed < threshold, f"Too slow: {elapsed:.3f}s"
```

## Continuous Integration

Tests are designed to:
- ✅ Run quickly (fast tests < 2s total)
- ✅ Be deterministic (no random failures)
- ✅ Provide clear failure messages
- ✅ Clean up after themselves

## Troubleshooting

**Tests fail with import errors?**
- Make sure you're in the project root directory
- Tests add parent directory to path automatically

**Slow tests taking too long?**
- Skip them with `pytest tests/ -k "not slow"`
- Or run fast tests only (default behavior)

**Performance tests failing?**
- Check if minimax DLL is compiled with optimizations
- See [test_minimax_performance.py](test_minimax_performance.py) for optimization instructions

## Dependencies

All tests use only standard dependencies:
- **pytest** - Test framework
- **numpy** - Array operations
- **Standard library** - pathlib, time, sys

No additional test dependencies needed! 🎯
