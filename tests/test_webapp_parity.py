"""Cross-language parity: webapp/engine.mjs vs the Python reference rules.

The browser build re-implements the rules layer in JS beside the exported ONNX
net.  A matching export does NOT imply matching rules, and a divergence here is
invisible to every eval we run -- the ladder drives the Python/C path, so a
browser-only bug can only ever be caught by comparing the two directly.

Precedent: the halfmove clock was inverted in engine.mjs (clones ticked and
jumps reset, the exact opposite of tick_clock) from the webapp's introduction
until 2026-07-26.  Because clock/CLOCK_LIMIT is fed to the net as obs channel 3
whenever clockObs is on, every browser game handed the net a false observation
that drifted further from the truth the more clones were played.

Runs the JS under emsdk's bundled node, so no separate toolchain is needed.
"""
import json
import pathlib
import subprocess

import pytest

from lib.t7g import (PASS_ACTION, action_masks, action_to_move, apply_move,
                     new_board, tick_clock)
from lib.uai_engine import _square_to_xy, board_to_fen, parse_uai_move

ROOT = pathlib.Path(__file__).resolve().parent.parent
ENGINE = ROOT / "webapp" / "engine.mjs"
NODE = next(iter(sorted(
    (ROOT / "3rd_party" / "emsdk" / "node").glob("*/bin/node"))), None)

# Dumps every quantity we can compare without reimplementing the board in JSON.
_DUMP_JS = """
import * as e from {engine!r};
const out = {{ tick: [], uai: [] }};
for (let a = 0; a <= 1225; a++) {{
  out.tick.push(e.tickClock(0, a));
  out.uai.push(e.actionToUAI(a));
}}
out.fen_x = e.boardToFEN(e.newBoard(), true);
out.fen_o = e.boardToFEN(e.newBoard(), false);
console.log(JSON.stringify(out));
"""


@pytest.fixture(scope="module")
def js(tmp_path_factory):
    if NODE is None or not NODE.exists():
        pytest.skip("no node under 3rd_party/emsdk/node (run 3rd_party/emsdk/emsdk install)")
    script = tmp_path_factory.mktemp("webapp") / "dump.mjs"
    script.write_text(_DUMP_JS.format(engine=str(ENGINE)))
    proc = subprocess.run([str(NODE), str(script)], capture_output=True, text=True)
    assert proc.returncode == 0, f"node failed:\n{proc.stderr}"
    return json.loads(proc.stdout)


def test_tick_clock_matches_reference(js):
    """Clones reset the halfmove clock; jumps and passes tick it."""
    mismatched = [a for a in range(1226) if js["tick"][a] != tick_clock(0, a)]
    assert not mismatched, (
        f"{len(mismatched)}/1226 actions disagree, e.g. action {mismatched[0]}: "
        f"js={js['tick'][mismatched[0]]} py={tick_clock(0, mismatched[0])}"
    )


def test_action_to_uai_round_trips(js):
    """Clones name only their destination; jumps name source then destination."""
    for action in range(1225):
        fx, fy, tx, ty, jump = action_to_move(action)
        if not (0 <= tx <= 6 and 0 <= ty <= 6):
            continue                       # off-board index, never legal
        move = js["uai"][action]
        assert _square_to_xy(move[-2:]) == (tx, ty), f"action {action} -> {move}"
        expected_from = (fx, fy) if jump else None
        got_from = _square_to_xy(move[:2]) if len(move) == 4 else None
        assert got_from == expected_from, f"action {action} -> {move}"


def test_pass_is_the_null_move(js):
    assert js["uai"][PASS_ACTION] == "0000"


@pytest.mark.parametrize("turn, key", [(True, "fen_x"), (False, "fen_o")])
def test_board_to_fen_matches_reference(js, turn, key):
    assert js[key] == board_to_fen(new_board(), turn)


# The unit checks above pin individual functions.  This one plays whole games
# in JS and replays them in Python, so it also covers legalMoves, applyMove and
# terminal detection -- the parts a per-function test can agree on in isolation
# while still disagreeing about an actual game.
_GAMES_JS = """
import * as e from {engine!r};

// xorshift32 so the games are identical on every run.
let s = 0x2545f491;
const rnd = (n) => (s ^= s << 13, s ^= s >>> 17, s ^= s << 5, ((s >>> 0) % n));

const games = [];
for (let g = 0; g < {n_games}; g++) {{
  let b = e.newBoard(), turn = true, clock = 0;
  const plies = [];
  for (let ply = 0; ply < {max_plies}; ply++) {{
    if (e.checkTerminal(b, turn).terminal || clock >= e.CLOCK_LIMIT) break;
    const legal = e.legalMoves(b, turn);
    const action = legal.length ? legal[rnd(legal.length)] : e.PASS_ACTION;
    if (action !== e.PASS_ACTION) b = e.applyMove(b, action, turn);
    clock = e.tickClock(clock, action);
    turn = !turn;
    plies.push({{ uai: e.actionToUAI(action), fen: e.boardToFEN(b, turn), clock }});
  }}
  games.push(plies);
}}
console.log(JSON.stringify(games));
"""


@pytest.fixture(scope="module")
def js_games(tmp_path_factory):
    if NODE is None or not NODE.exists():
        pytest.skip("no node under 3rd_party/emsdk/node")
    script = tmp_path_factory.mktemp("webapp") / "games.mjs"
    script.write_text(_GAMES_JS.format(engine=str(ENGINE), n_games=25, max_plies=400))
    proc = subprocess.run([str(NODE), str(script)], capture_output=True, text=True)
    assert proc.returncode == 0, f"node failed:\n{proc.stderr}"
    return json.loads(proc.stdout)


def test_random_games_replay_identically(js_games):
    """Every JS ply must be legal in Python and reach the same board + clock."""
    assert sum(len(g) for g in js_games) > 500, "harness produced too few plies"

    for i, plies in enumerate(js_games):
        board, turn, clock = new_board(), True, 0
        for ply, step in enumerate(plies):
            where = f"game {i} ply {ply} ({step['uai']})"
            if step["uai"] == "0000":
                assert not action_masks(board, turn).any(), f"{where}: passed with moves available"
                action = PASS_ACTION
            else:
                action = parse_uai_move(step["uai"], board, turn)
                board = apply_move(board, action, turn)
            clock = tick_clock(clock, action)
            turn = not turn
            assert board_to_fen(board, turn) == step["fen"], f"{where}: board diverged"
            assert clock == step["clock"], f"{where}: clock diverged"
