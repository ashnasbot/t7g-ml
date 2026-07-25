// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Ashnas
//
// stauf_wasm.cpp - emscripten entry point wrapping ScummVM's Groovie CellGame
// (the original 7th Guest microscope AI) for the browser SPA.
//
// This file is MIT, but it is compiled and linked against the GPLv3 CellGame in
// thirdparty/scummvm-cell/, so *the resulting stauf.wasm is a combined work
// conveyed under GPLv3*.  See thirdparty/scummvm-cell/README.md.
//
// It is the browser sibling of the native 3rd_party/cell/cell_dll.cpp used by
// lib/t7g.py, and deliberately keeps the same board layout and action encoding
// so the two agree move-for-move.
//
// Build:
//   source 3rd_party/emsdk/emsdk_env.sh && make stauf-wasm

#include <emscripten.h>
#include <cstdint>
#include <cstdarg>
#include <cstdlib>

#include "cell.h"

using namespace Groovie;

// ---------------------------------------------------------------------------
// ScummVM runtime stubs.  Declared by shim/common/textconsole.h; CellGame calls
// warning() from its four out-of-range getters, and never calls error().
// ---------------------------------------------------------------------------
void warning(const char * /*s*/, ...) {}
void error(const char * /*s*/, ...) { abort(); }

// ---------------------------------------------------------------------------
// stauf_find_best_move - pick Stauf's move for `board`.
//
// board       : 98 bytes, the same layout engine.mjs uses (and the same as a
//               (7,7,2) numpy bool_ array in C order):
//                 board[(y*7 + x)*2 + 0] = green presence at (x,y)
//                 board[(y*7 + x)*2 + 1] = blue  presence at (x,y)
// depth       : CellGame difficulty selector, NOT a ply count.  It indexes a
//               lookup table that yields the real search depth (see below).
//               6 is the canonical original-game setting (lib/eval_db.py,
//               STAUF_DEPTH).
// as_blue     : non-zero = Stauf plays Blue, zero = Stauf plays Green.
// move_count  : Stauf's own cumulative move index this game (0 for its first
//               move, then 1, 2, ...).  CellGame cycles its real search depth
//               with `depths[3*(depth-2) + move_count%3]`, so passing a
//               *per-side cumulative* counter is what reproduces the real
//               game's move-to-move depth variation.  This mirrors
//               play_engine_vs_engine() in lib/train_workers.py, which tracks
//               the same per-side index.  Pass -1 for CellGame's default.
//
// Returns a flat action index matching engine.mjs's encoding:
//   (fy*7 + fx)*25 + (dy + 2)*5 + (dx + 2)
// or 1225 (engine.mjs PASS_ACTION) when no legal move was produced.
// ---------------------------------------------------------------------------
extern "C" EMSCRIPTEN_KEEPALIVE
int stauf_find_best_move(const uint8_t *board, int depth, int as_blue,
                         int move_count) {
    CellGame game(/*easierAi=*/false);
    if (move_count >= 0)
        game.setMoveCount(move_count);

    // CellGame::run() wants a 49-byte "script board" using the original game's
    // cell encoding: 66 ('B') = the side CellGame is playing, 50 ('2') = the
    // opponent, 0 = empty.  CellGame always searches for GREEN internally, so
    // when Stauf is Blue we swap which colour maps to 66.  Coordinates are
    // unaffected either way.
    uint8_t script[49];
    for (int y = 0; y < 7; ++y) {
        for (int x = 0; x < 7; ++x) {
            const bool green = board[(y * 7 + x) * 2 + 0];
            const bool blue  = board[(y * 7 + x) * 2 + 1];
            const bool mine  = as_blue ? blue : green;
            const bool yours = as_blue ? green : blue;
            script[y * 7 + x] = mine ? 66 : (yours ? 50 : 0);
        }
    }

    game.run(static_cast<uint16>(depth), script);

    const uint8_t sx = game.getStartX(), sy = game.getStartY();
    const uint8_t ex = game.getEndX(),   ey = game.getEndY();

    // When no move was available CellGame leaves _startX = 255 and the getters
    // fall back to (0,6)->(1,6), which need not be legal on this board.  Verify
    // the move before trusting it rather than special-casing the sentinel.
    const int dx = (int)ex - (int)sx, dy = (int)ey - (int)sy;
    if (sx >= 7 || sy >= 7 || ex >= 7 || ey >= 7
            || dx < -2 || dx > 2 || dy < -2 || dy > 2)
        return 1225;

    const bool from_mine = board[(sy * 7 + sx) * 2 + (as_blue ? 1 : 0)];
    const bool to_empty  = !board[(ey * 7 + ex) * 2 + 0]
                        && !board[(ey * 7 + ex) * 2 + 1];
    if (!from_mine || !to_empty)
        return 1225;

    return (sy * 7 + sx) * 25 + (dy + 2) * 5 + (dx + 2);
}
