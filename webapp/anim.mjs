// Move animation for the board: a jump leaps, a clone divides, and every
// capture is another clone of the cell that just landed.  Purely presentational
// -- the caller owns the game state and has already committed it before calling
// in here; this module only controls *when* the display catches up, via the
// `setDisplay` callback.
//
// The display therefore lags the truth for the length of one animation.  That
// is the whole design: app.mjs keeps a `displayBoard` that render() paints, and
// playMove() walks it through the intermediate states (source lifted -> mover
// landed -> captures taken one by one) as the transitions run.
//
// Everything uses the Web Animations API rather than CSS classes plus
// transitionend, because every phase has to be abandonable: "New game" mid-move
// must not leave a stone stranded in flight, and transitionend never fires for
// a node that gets torn out from under it.  cancelAll() bumps a generation
// counter that every await in here re-checks.
import * as engine from './engine.mjs';

const REDUCE = matchMedia('(prefers-reduced-motion: reduce)');

// One capture clone is always DUR.capture, however many a move takes: a
// sequence that speeds up when it is long reads as a glitch, and eight captures
// is a big swing that deserves the ~1.8s it costs to watch.
const BASE = { leap: 240, goo: 380, capture: 220 };
// Debug hook: set window.t7gAnimSlow = 10 in the console to stretch every phase
// 10x.  A one-frame artefact is impossible to localise at full speed -- slowed
// down it is obvious which phase (flight, handover, capture, settle) drops it.
const DUR = new Proxy(BASE, { get: (t, k) => t[k] * (globalThis.t7gAnimSlow || 1) });
// Neighbour order for captures: clockwise from the upper-left, the order the
// original game resolves them in.
const RING = [[-1, -1], [0, -1], [1, -1], [1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0]];

let gen = 0;                  // bumped by cancelAll(); awaits below compare against it
let seq = 0;                  // bumped per playMove; guards its deferred cleanups
let layer = null;             // .fx overlay, one per board, holds the flying stones
const touched = new Set();    // cells wearing a transient class, so we can strip them

// Abandon whatever is in flight.  The caller must follow up by resetting its
// displayBoard to the real board and re-rendering: this only tears down the
// visuals, it does not know what the truth is.
export function cancelAll() {
  gen++;
  if (layer) layer.replaceChildren();
  clearMarks();
}

function clearMarks() {
  for (const c of touched) c.classList.remove('lift', 'snap');
  touched.clear();
}

function mark(cell, cls) { cell.classList.add(cls); touched.add(cell); }

// Run once the browser has actually presented the pending frame.  One rAF only
// guarantees the style change was *committed*; rasterisation is off-thread, so
// tearing down a composited layer that soon can still beat the repaint that
// replaces it to the screen.  Same double-rAF idiom as app.mjs's nextPaint.
const afterPaint = (fn) => requestAnimationFrame(() => requestAnimationFrame(fn));

function ensureLayer(boardEl) {
  if (layer && layer.parentNode === boardEl) return layer;
  layer = document.createElement('div');
  layer.className = 'fx';
  boardEl.appendChild(layer);     // after the 49 cells, so :nth-child stays intact
  return layer;
}

// Cell box in the board's own coordinates.  offsetLeft/offsetTop are relative to
// #board (position: relative), which is exactly the origin the .fx layer uses,
// so no getBoundingClientRect and no scroll-position bookkeeping.
function geom(cells, x, y) {
  const el = cells[y * 7 + x];
  return { l: el.offsetLeft, t: el.offsetTop, w: el.offsetWidth, h: el.offsetHeight };
}

// How far apart the mitosis blobs stay fused, which has to scale with the cell
// or the effect vanishes on a phone and smears on a desktop.  Paired with the
// alpha cutoff in #goo: raising one without lowering the other just dilates
// both blobs instead of thickening the join.  Set once per move -- the filter
// is shared, and every clone in a move is the same size.
function tuneGoo(cellW) {
  const blur = document.getElementById('goo-blur');
  if (blur) blur.setAttribute('stdDeviation', (cellW * 0.145).toFixed(2));
}

// Squares the move converts: adjacent to the destination, opponent before, ours
// after.  Derived from the two boards rather than re-deriving the rules, and
// walked in RING order so the caller gets them clockwise from the upper-left.
function convertedCells(prev, next, tx, ty, turn) {
  const pc = turn ? 1 : 0, oc = 1 - pc;
  const out = [];
  for (const [dx, dy] of RING) {
    const x = tx + dx, y = ty + dy;
    if (x < 0 || x > 6 || y < 0 || y > 6) continue;
    const i = (y * 7 + x) * 2;
    if (prev[i + oc] && next[i + pc]) out.push({ x, y });
  }
  return out;
}

// The post-move board as it looks *before* the captures resolve: mover landed,
// converted squares still in the opponent's colour.
function midBoard(next, conv, turn) {
  const pc = turn ? 1 : 0, oc = 1 - pc;
  const m = next.slice();
  for (const { x, y } of conv) { const i = (y * 7 + x) * 2; m[i + pc] = 0; m[i + oc] = 1; }
  return m;
}

function fxStone(col, g, cls = 'fx-stone') {
  const e = document.createElement('div');
  e.className = `${cls} ${col}`;
  const s = g.w * 0.66;
  e.style.width = e.style.height = `${s}px`;
  e.style.left = `${g.l + (g.w - s) / 2}px`;
  e.style.top = `${g.t + (g.h - s) / 2}px`;
  return e;
}

// ---- the leap -------------------------------------------------------------
// A jump vacates its source, so the stone itself travels.  The real source
// stone is hidden (.lift) and a copy flies in the overlay, which keeps the
// reconciling renderer out of the way of the transform.
// Returns { node, done }: the caller removes `node` itself, a frame after it
// has painted the real stone underneath.  Clearing the whole overlay instead
// would take out a capture clone that has already started.
function leap(L, col, src, dst) {
  const el = fxStone(col, src);
  L.appendChild(el);
  const dx = dst.l - src.l, dy = dst.t - src.t;
  const done = el.animate([
    { transform: 'translate(0,0) scale(1)' },
    // Mid-flight lift + swell: without the arc it reads as a slide, not a leap.
    { transform: `translate(${dx * 0.5}px, ${dy * 0.5 - src.h * 0.28}px) scale(1.16)`, offset: 0.5 },
    { transform: `translate(${dx}px, ${dy}px) scale(1)` },
    // fill:forwards is load-bearing, not cosmetic: `finished` resolves a frame
    // after the last animation frame, and without a fill the copy reverts to
    // its base transform (back at the source) for exactly that frame.  The
    // destination has nothing in it yet, so the stone blinks out of existence.
  ], { duration: DUR.leap, easing: 'cubic-bezier(.3,0,.2,1)', fill: 'forwards' }).finished;
  return { node: el, done };
}

// ---- mitosis --------------------------------------------------------------
// Two blobs inside one element filtered by #goo (blur + alpha contrast, with
// the crisp originals blended back on top).  While the daughter is still
// overlapping the parent the threshold fuses them into a single elongated blob;
// as it pulls clear the connecting meniscus thins and snaps.  That pinch is the
// whole effect -- it is a property of the filter, not of any keyframe.
function mitosis(L, col, src, dst, duration = DUR.goo) {
  const l = Math.min(src.l, dst.l), t = Math.min(src.t, dst.t);
  const box = document.createElement('div');
  box.className = `fx-goo ${col}`;
  box.style.left = `${l}px`;
  box.style.top = `${t}px`;
  box.style.width = `${Math.abs(dst.l - src.l) + src.w}px`;
  box.style.height = `${Math.abs(dst.t - src.t) + src.h}px`;

  // Both daughters start stacked on the parent square; coordinates are relative
  // to the goo box, hence the shift by its origin.
  const at = { l: src.l - l, t: src.t - t, w: src.w, h: src.h };
  const a = fxStone(col, at, 'fx-blob'), b = fxStone(col, at, 'fx-blob');
  box.append(a, b);
  L.appendChild(box);

  const dx = dst.l - src.l, dy = dst.t - src.t;
  a.animate([
    { transform: 'scale(1)' },
    { transform: 'scale(1.2, .86)', offset: 0.35 },   // parent squashes as it buds
    { transform: 'scale(.95, 1.05)', offset: 0.7 },
    { transform: 'scale(1)' },
  ], { duration, easing: 'ease-in-out', fill: 'forwards' });
  const done = b.animate([
    { transform: 'translate(0,0) scale(.5)' },
    { transform: `translate(${dx * 0.28}px, ${dy * 0.28}px) scale(.85)`, offset: 0.4 },
    { transform: `translate(${dx}px, ${dy}px) scale(1.08)`, offset: 0.85 },
    { transform: `translate(${dx}px, ${dy}px) scale(1)` },
  ], { duration, easing: 'cubic-bezier(.45,.05,.3,1)', fill: 'forwards' }).finished;
  return { node: box, done };
}

// ---- captures -------------------------------------------------------------
// As in the original: the cell that just landed clones onto each captured
// neighbour in turn, clockwise from the upper-left.  Strictly sequential -- the
// whole point is that you can watch it work its way round.
//
// The destination stays lifted for the entire sequence: every clone's parent
// blob stands in for it, and consecutive overlays overlap by a frame (each is
// removed only after the next has been created), so it is covered throughout.
//
// Each captured cell's colour class is poked directly rather than re-rendered,
// because the display board holds the pre-capture colours for this whole phase
// -- see midBoard().
async function cloneCaptures(L, cells, conv, col, dstCell, dst, stale) {
  const theirs = col === 'blue' ? 'green' : 'blue';
  mark(dstCell, 'lift');
  for (const { x, y } of conv) {
    const cell = cells[y * 7 + x];
    const { node, done } = mitosis(L, col, dst, geom(cells, x, y), DUR.capture);
    try { await done; } catch { return; }
    if (stale()) return;
    // The daughter has landed squarely on the captured stone and is the same
    // size and fill, so swapping the colour underneath it is invisible; the
    // overlay is dropped a frame later, once the real stone has painted.
    cell.classList.remove(theirs);
    cell.classList.add(col);
    afterPaint(() => node.remove());
  }
}

// Play one move.  `prev` and `next` are the boards either side of it, `action`
// and `turn` identify it, and `setDisplay(board)` hands a board back to the
// caller to paint.  Resolves once the display has caught up with `next`; also
// resolves early (leaving the display wherever it was) if cancelAll() fires,
// which the caller detects with its own generation guard.
export async function playMove({ cells, boardEl, prev, next, action, turn, setDisplay }) {
  const my = gen, mySeq = ++seq;
  const stale = () => my !== gen;
  // Deferred cleanups fire two frames late, by which time the *next* move may
  // have started and marked cells of its own.  Stripping classes then would
  // un-hide a stone that is currently mid-flight, so anything deferred has to
  // confirm it is still the move in charge.  Node removal is exempt: it is
  // always correct, and skipping it would leak the overlay.
  const current = () => mySeq === seq && my === gen;
  if (REDUCE.matches) { setDisplay(next); return; }

  const { fx, fy, tx, ty, jump } = engine.actionToMove(action);
  const conv = convertedCells(prev, next, tx, ty, turn);
  const col = turn ? 'blue' : 'green';
  const L = ensureLayer(boardEl);
  const src = geom(cells, fx, fy), dst = geom(cells, tx, ty);
  tuneGoo(src.w);

  // The overlay carries the mover for both kinds of move: for a jump the source
  // is vacated anyway, and for a clone the parent blob has to be the thing that
  // squashes, so the static stone underneath would double-draw.
  mark(cells[fy * 7 + fx], 'lift');
  let mover;
  try {
    mover = jump ? leap(L, col, src, dst) : mitosis(L, col, src, dst);
    await mover.done;
  } catch { return; }              // cancelled mid-flight
  if (stale()) return;

  // Hand the mover over to the real board.  A move with captures paints them in
  // their *old* colour here, leaving the capture clones something to take; one
  // without goes straight to the final board.
  //
  // Both endpoints suppress the stone's appear/disappear transition for this
  // one paint: the overlay already left them looking finished, so letting the
  // transition run would scale the arriving stone up from nothing a second time.
  clearMarks();
  const dstCell = cells[ty * 7 + tx];
  const ends = [dstCell, cells[fy * 7 + fx]];
  for (const c of ends) mark(c, 'snap');
  setDisplay(conv.length ? midBoard(next, conv, turn) : next);

  // The mover's overlay deliberately outlives the handover by a frame.  Its
  // copies sit exactly on top of the real stones -- same position, size and
  // fill -- so the overlap is invisible, whereas removing them in this task
  // races the repaint that reveals the real stones: the composited overlay
  // layer can vanish a frame before the stone underneath is rasterised, and
  // both endpoints blink.  Only this node goes, never the whole layer: by then
  // the first capture clone is already running in it.
  afterPaint(() => {
    mover.node.remove();
    if (current()) for (const c of ends) c.classList.remove('snap');
  });
  if (!conv.length) return;

  await cloneCaptures(L, cells, conv, col, dstCell, dst, stale);
  if (stale()) return;
  // Order matters: paint the final board first, and only drop the transient
  // classes a frame later.  Doing both at once means the style recalc that
  // re-enables `transition` lands in the same frame as a class change, and any
  // difference between the poked and rendered state fires a 120ms fade.
  setDisplay(next);
  afterPaint(() => { if (current()) clearMarks(); });
}
