// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Ashnas
//
// Web Worker hosting the Stauf engine (ScummVM's Groovie CellGame, the original
// 7th Guest microscope AI) compiled to wasm.
//
// WHY A WORKER.  stauf.wasm is GPLv3 (see thirdparty/scummvm-cell/README.md);
// the rest of the SPA is MIT.  Keeping it in its own worker behind the narrow
// message protocol below means the GPL code is never linked with ours -- the
// boundary is message passing between separate modules, the same arm's-length
// arrangement lib/uai_engine.py uses for external engines like autaxx.  Nothing
// of ours (engine.mjs, net2.onnx) is part of that combined work.
//
// It also happens to keep Stauf's search off the UI thread, though at depth 6
// that search is a couple of plies and effectively instant.
//
// Protocol:
//   <- {type:'init'}                                    -> {type:'ready'} | {type:'error', message}
//   <- {type:'move', board, asBlue, moveCount, depth}   -> {type:'move', action, ms}
// `board` is the 98-byte layout from engine.mjs; `action` is its flat action
// index, or PASS_ACTION (1225) when Stauf has no move.

import Stauf from './stauf.mjs';

// CellGame difficulty selector, not a ply count: it indexes depths[] in
// cell.cpp, which yields the real depth.  6 is the original-game setting.
// Out of range reads past that table, hence the clamp.
const STAUF_DEPTH = 6;
const MIN_DEPTH = 2, MAX_DEPTH = 8;

const pickDepth = (d) =>
  Number.isInteger(d) ? Math.min(MAX_DEPTH, Math.max(MIN_DEPTH, d)) : STAUF_DEPTH;

let mod = null, boardPtr = 0;

self.onmessage = async (e) => {
  const msg = e.data;
  try {
    if (msg.type === 'init') {
      if (!mod) {
        mod = await Stauf();
        boardPtr = mod._malloc(98);
      }
      self.postMessage({ type: 'ready' });
      return;
    }

    if (msg.type === 'move') {
      const t0 = performance.now();
      mod.HEAPU8.set(msg.board, boardPtr);
      // moveCount is Stauf's own cumulative move index *for this side*: CellGame
      // varies its real search depth on moveCount % 3, so a per-side counter is
      // what reproduces the original game's move-to-move variation.  Restarting
      // it at 0 each move would pin Stauf to one depth slot and make it a
      // slightly different (and differently-rated) opponent.
      const action = mod.ccall(
        'stauf_find_best_move', 'number',
        ['number', 'number', 'number', 'number'],
        [boardPtr, pickDepth(msg.depth), msg.asBlue ? 1 : 0, msg.moveCount],
      );
      self.postMessage({ type: 'move', action, ms: Math.round(performance.now() - t0) });
      return;
    }
  } catch (err) {
    self.postMessage({ type: 'error', message: String(err && err.message || err) });
  }
};
