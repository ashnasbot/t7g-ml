"""The browser Stauf (wasm) must play identically to the native cell_dll Stauf.

Both wrap the same GPLv3 ScummVM CellGame, but via different wrappers
(src/stauf_wasm.cpp vs 3rd_party/cell/cell_dll.cpp) and different compilers.
That matters beyond ordinary build hygiene: cell_dll's Stauf is the *sole fixed
anchor* of the rating ladder (lib/eval_db.py, STAUF_ANCHOR_ELO), so if the
browser build diverged, the SPA would be advertising an opponent that is not the
thing the published Elo numbers describe.

Skipped unless `make stauf-wasm` has been run and emsdk's node is available.
"""
import json
import os
import subprocess
import textwrap

import numpy as np
import pytest

from lib.t7g import (new_board, action_masks, apply_move, find_best_move,
                     check_terminal)
from lib.eval_db import STAUF_DEPTH

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_WASM = os.path.join(_ROOT, "build", "wasm", "stauf.mjs")
_NODE = os.path.join(_ROOT, "3rd_party", "emsdk", "node", "22.16.0_64bit", "bin", "node")

pytestmark = pytest.mark.skipif(
    not (os.path.exists(_WASM) and os.path.exists(_NODE)),
    reason="needs `make stauf-wasm` and emsdk node",
)

# Mirrors webapp/stauf.worker.mjs: stage the 98-byte board, call through,
# compare the returned flat action index.
_DRIVER = textwrap.dedent("""
    import fs from 'node:fs';
    import { pathToFileURL } from 'node:url';
    const { default: Stauf } = await import(pathToFileURL(process.argv[2]).href);
    const cases = JSON.parse(fs.readFileSync(process.argv[3], 'utf8'));
    const mod = await Stauf();
    const ptr = mod._malloc(98);
    const out = [];
    for (const c of cases) {
      mod.HEAPU8.set(Uint8Array.from(c.board), ptr);
      out.push(mod.ccall('stauf_find_best_move', 'number',
                         ['number','number','number','number'],
                         [ptr, c.depth, c.as_blue, c.move_count]));
    }
    mod._free(ptr);
    fs.writeFileSync(process.argv[4], JSON.stringify(out));
""")


def _native_cases(n_games=40, max_plies=40, seed=12345):
    """Play semi-random games, recording Stauf's native answer at each ply.

    Stauf's move index is tracked *per side* and passed as move_count, because
    CellGame cycles its real search depth on move_count % 3 -- the same
    convention play_engine_vs_engine() uses in lib/train_workers.py.  Getting
    this wrong would still produce legal moves, just a different opponent, so
    the cases deliberately span many different move_count values.
    """
    rng = np.random.default_rng(seed)
    cases = []
    for _ in range(n_games):
        board, turn = new_board(), True
        stauf_move_idx = {True: 0, False: 0}
        for _ in range(max_plies):
            terminal, _v = check_terminal(board, turn)
            if terminal:
                break
            masks = action_masks(board, turn)
            if not np.any(masks):
                turn = not turn
                continue
            mc = stauf_move_idx[turn]
            expect = find_best_move(board.tobytes(), STAUF_DEPTH, turn,
                                    engine="stauf", move_count=mc)
            cases.append({"board": board.astype(np.uint8).flatten().tolist(),
                          "depth": STAUF_DEPTH, "as_blue": int(turn),
                          "move_count": mc, "expect": int(expect)})
            stauf_move_idx[turn] += 1
            # Advance with a random legal move rather than Stauf's own, so the
            # positions probed are not confined to Stauf's self-play line.
            board = apply_move(board, int(rng.choice(np.where(masks)[0])), turn)
            turn = not turn
    return cases


def test_wasm_stauf_matches_native_stauf(tmp_path):
    cases = _native_cases()
    assert len(cases) > 500, "expected a few thousand probe positions"

    cases_json = tmp_path / "cases.json"
    out_json = tmp_path / "out.json"
    driver = tmp_path / "driver.mjs"
    cases_json.write_text(json.dumps(cases))
    driver.write_text(_DRIVER)

    proc = subprocess.run([_NODE, str(driver), _WASM, str(cases_json),
                           str(out_json)], capture_output=True, text=True)
    assert proc.returncode == 0, f"wasm driver failed:\n{proc.stderr}"
    got = json.loads(out_json.read_text())

    assert len(got) == len(cases)
    bad = [(i, c["expect"], g) for i, (c, g) in enumerate(zip(cases, got))
           if c["expect"] != g]
    assert not bad, (
        f"{len(bad)}/{len(cases)} browser-Stauf moves differ from native Stauf; "
        f"first few (index, native, wasm): {bad[:5]}"
    )
