// t7g browser engine: Ataxx rules (mirrored from lib/t7g.py) + the MCGS search
// pull-loop driver.  Environment-agnostic: the caller supplies the loaded wasm
// Module and a `runNet(obsFloat32, n)` callback, so this same file runs under
// node (wasm EP) and in the browser (WebGPU).  Rules are cross-checked against
// the Python reference by scripts/parity_rules — never trusted blind.
//
// Board layout matches numpy (7,7,2) bool in C order:
//   idx(x,y,ch) = (y*7 + x)*2 + ch,  ch 0 = green, ch 1 = blue.
// turn: true = Blue to move, false = Green.

export const PASS_ACTION = 1225;
export const CLOCK_LIMIT = 100;

const idx = (x, y, ch) => (y * 7 + x) * 2 + ch;

export function newBoard() {
  const b = new Uint8Array(98);
  const set = (x, y, blue) => { b[idx(x, y, 1)] = blue ? 1 : 0; b[idx(x, y, 0)] = blue ? 0 : 1; };
  set(0, 0, true); set(6, 0, false); set(0, 6, false); set(6, 6, true);  // BLUE tl/br, GREEN tr/bl
  return b;
}

export function countCells(b) {
  let blue = 0, green = 0;
  for (let i = 0; i < 49; i++) { green += b[i * 2]; blue += b[i * 2 + 1]; }
  return { blue, green };
}

// action = piece*25 + move; piece = fy*7+fx; move = (dy+2)*5 + (dx+2)
export function actionToMove(action) {
  const piece = Math.floor(action / 25), move = action % 25;
  const fx = piece % 7, fy = Math.floor(piece / 7);
  const dx = (move % 5) - 2, dy = Math.floor(move / 5) - 2;
  const tx = fx + dx, ty = fy + dy;
  const jump = Math.abs(dx) === 2 || Math.abs(dy) === 2;
  return { fx, fy, tx, ty, jump };
}

export function moveToAction(fx, fy, tx, ty) {
  const dx = tx - fx, dy = ty - fy;
  return (fy * 7 + fx) * 25 + (dy + 2) * 5 + (dx + 2);
}

export function tickClock(clock, action) {
  if (action === PASS_ACTION) return clock + 1;
  const mv = action % 25;
  const jump = Math.abs((mv % 5) - 2) === 2 || Math.abs(Math.floor(mv / 5) - 2) === 2;
  return jump ? 0 : clock + 1;
}

// Legal action indices for `turn`.  Mirrors action_masks: own piece at source,
// destination in-bounds and completely empty.
export function legalMoves(b, turn) {
  const pc = turn ? 1 : 0;
  const out = [];
  for (let fy = 0; fy < 7; fy++) for (let fx = 0; fx < 7; fx++) {
    if (!b[idx(fx, fy, pc)]) continue;
    for (let dy = -2; dy <= 2; dy++) for (let dx = -2; dx <= 2; dx++) {
      const tx = fx + dx, ty = fy + dy;
      if (tx < 0 || tx > 6 || ty < 0 || ty > 6) continue;
      if (b[idx(tx, ty, 0)] || b[idx(tx, ty, 1)]) continue;  // dest not empty
      out.push((fy * 7 + fx) * 25 + (dy + 2) * 5 + (dx + 2));
    }
  }
  return out;
}

// {"fx,fy": [{action, tx, ty, jump}, ...]} for click-to-highlight in the UI.
export function legalMovesBySource(b, turn) {
  const map = new Map();
  for (const a of legalMoves(b, turn)) {
    const m = actionToMove(a);
    const key = `${m.fx},${m.fy}`;
    if (!map.has(key)) map.set(key, []);
    map.get(key).push({ action: a, tx: m.tx, ty: m.ty, jump: m.jump });
  }
  return map;
}

export function applyMove(b, action, turn) {
  const nb = b.slice();
  const pc = turn ? 1 : 0, oc = 1 - pc;
  const { fx, fy, tx, ty, jump } = actionToMove(action);
  if (jump) { nb[idx(fx, fy, 0)] = 0; nb[idx(fx, fy, 1)] = 0; }
  nb[idx(tx, ty, pc)] = 1; nb[idx(tx, ty, oc)] = 0;
  for (let yy = Math.max(0, ty - 1); yy <= Math.min(6, ty + 1); yy++)
    for (let xx = Math.max(0, tx - 1); xx <= Math.min(6, tx + 1); xx++)
      if (nb[idx(xx, yy, oc)]) { nb[idx(xx, yy, pc)] = 1; nb[idx(xx, yy, oc)] = 0; }
  return nb;
}

// (terminal, value from `turn`'s perspective) — mirrors check_terminal.
export function checkTerminal(b, turn) {
  const { blue, green } = countCells(b);
  if (blue === 0) return { terminal: true, value: turn ? -1 : 1 };
  if (green === 0) return { terminal: true, value: turn ? 1 : -1 };
  if (legalMoves(b, turn).length || legalMoves(b, !turn).length)
    return { terminal: false, value: null };
  const score = turn ? blue - green : green - blue;
  return { terminal: true, value: score > 0 ? 1 : score < 0 ? -1 : 0 };
}

// ---- MCGS search over the wasm module -------------------------------------

function softmaxRowInto(dst, src, off, n) {  // in-place softmax of src[off..off+n)
  let m = -Infinity;
  for (let i = 0; i < n; i++) if (src[off + i] > m) m = src[off + i];
  let s = 0;
  for (let i = 0; i < n; i++) { const e = Math.exp(src[off + i] - m); dst[i] = e; s += e; }
  const inv = 1 / s;
  for (let i = 0; i < n; i++) dst[i] *= inv;
}

// Drive one MCGS search to completion; returns {action, probs, rootValue}.
// `runNet(obsF32, n)` -> {policy: Float32Array(n*1225 logits), value: Float32Array(n)}.
export async function searchMove(mod, runNet, board, turn, clock, cfg) {
  const {
    sims = 500, cPuct = 1.3, gumbelK = 16,
    completionN0 = 50.0, sigmaScale = 1.0, clockObs = true,
    seed = (Math.random() * 2 ** 53) >>> 0,
  } = cfg || {};

  mod._mcgs_init();
  const inst = mod._mcgs_create_ex(sims, cPuct, gumbelK, 0, 0);
  mod._mcgs_set_completion_n0(inst, completionN0);
  mod._mcgs_set_sigma_scale(inst, sigmaScale);
  mod._mcgs_set_clock_obs(inst, clockObs ? 1 : 0);
  mod._mcgs_set_rng_seed(inst, BigInt(seed >>> 0));

  const boardPtr = mod._malloc(98);
  mod.HEAPU8.set(board, boardPtr);
  const ss = mod._mcgs_start_search(inst, boardPtr, turn ? 1 : 0, clock | 0);

  const MAXN = Math.max(gumbelK, 8);
  const obsPtr = mod._malloc(MAXN * 7 * 7 * 4 * 4);
  const polPtr = mod._malloc(MAXN * 1225 * 4);
  const valPtr = mod._malloc(MAXN * 4);
  const row = new Float32Array(1225);

  let guard = 0;
  while (!mod._mcgs_is_done(ss)) {
    const n = mod._mcgs_pending_count(ss);
    if (n > 0) {
      mod._mcgs_get_pending_obs(ss, obsPtr);
      const obs = mod.HEAPF32.slice(obsPtr >> 2, (obsPtr >> 2) + n * 196);
      const { policy, value } = await runNet(obs, n);          // logits, value
      const pol = mod.HEAPF32;
      for (let k = 0; k < n; k++) {
        softmaxRowInto(row, policy, k * 1225, 1225);
        pol.set(row, (polPtr >> 2) + k * 1225);
      }
      mod.HEAPF32.set(value.subarray(0, n), valPtr >> 2);
      mod._mcgs_commit_batch(ss, polPtr, valPtr, n);
    }
    mod._mcgs_step(ss);
    if (++guard > 200000) throw new Error('search runaway');
  }

  mod._mcgs_get_result(ss, polPtr);                            // reuse polPtr for 1225 out
  const probs = mod.HEAPF32.slice(polPtr >> 2, (polPtr >> 2) + 1225);
  const rootValue = mod._mcgs_get_root_value(ss);

  let action = -1, best = -1;
  for (let i = 0; i < 1225; i++) if (probs[i] > best) { best = probs[i]; action = i; }

  mod._mcgs_search_destroy(ss);
  mod._mcgs_destroy(inst);
  mod._free(boardPtr); mod._free(obsPtr); mod._free(polPtr); mod._free(valPtr);

  return { action: best > 0 ? action : -1, probs, rootValue };
}
