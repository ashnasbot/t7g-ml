// Browser entry: wires onnxruntime-web (WebGPU, wasm fallback) + the micro_mcts
// wasm search + engine.mjs into a human-vs-AI game against a choice of
// opponents.  Fully client-side — no server.  See engine.mjs for the rules +
// search driver (shared with the node end-to-end test).
//
// The onnxruntime-web runtime is pulled from CDN rather committed to the repo, pinned to ORT_VER.
//
// Both engines load lazily, on first use: playing Stauf should not drag in
// ORT and the model, and playing net2 should not fetch the Stauf module.  That
// also means the page should work offline.
import * as engine from './engine.mjs';
import * as anim from './anim.mjs';
import MicroMCTS from './micro_mcts.mjs';

// 1.27.0 or newer is required, not merely preferred: the WebGPU EP in 1.20.x
// computes only the first item of a batched run and leaves the rest to whatever
// was in the buffer.  The search dispatches gumbelK=16 at a time, so on that
// version 15 of every 16 priors were noise and the net played like a beginner
// wherever WebGPU was available -- silently, since nothing errors.  Measured
// against the wasm EP: 1/16 batch items correct on 1.20.1, 16/16 on 1.27.0.
const ORT_VER = '1.27.0';
const ORT_CDN = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VER}/dist/`;

const SIMS = 500;                 // canonical net2 config (eval_db DEFAULT_CONFIG)
const CFG = { sims: SIMS, cPuct: 1.3, gumbelK: 16, completionN0: 50.0, sigmaScale: 1.0, clockObs: true };
// Which side the human plays: true = Blue (moves first), false = Green.  Blue
// has the first-move advantage, so Green is the harder half of the choice.
// Both engines take the side as a parameter (micro_mcts' start_search, and
// the worker's asBlue), so nothing below this line is colour-specific.
let HUMAN = true;
const SIDE_KEY = 't7g.side';      // buttons aren't restored on reload the way a <select> is
const colour = (t) => (t ? 'Blue' : 'Green');

// Selectable opponents.  Stauf is the original T7G AI (via ScummVM's Groovie
// CellGame), and lives behind a worker because it is GPLv3 while this file
// is not -- see stauf.worker.mjs.
// `depth` is CellGame's difficulty selector (2-8), not a ply count.
const OPPONENTS = {
  'stauf-easy': { label: 'Stauf (Easy)', depth: 2,
                  meta: 'the original AI at its lowest setting · 1 ply · GPLv3' },
  stauf:        { label: 'Stauf',        depth: 6,
                  meta: 'the original 7th Guest AI · ScummVM CellGame · GPLv3' },
  net:          { label: 'AshnasBot',    meta: 'net2 · 500-sim MCGS' },
};

const $ = (id) => document.getElementById(id);
// Resolve after the browser has actually painted (two rAFs) so the player's
// move is on screen before the synchronous search setup can block the thread.
const nextPaint = () => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)));
const boardEl = $('board'), statusEl = $('status'), scoreEl = $('score');
const metaEl = $('meta'), opponentEl = $('opponent'), sideEl = $('side'), sideNoteEl = $('side-note');
const veilEl = $('veil');

let ort, mod, session, netReady = null;
let backend = null;               // 'webgpu' | 'wasm', reported in the move status
// One MCGS instance for the page, so its transposition table survives across
// moves the way self-play and eval keep theirs.  Cleared at the start of each
// game rather than destroyed.
// The clear is deferred to the next search because it is only safe when none is in flight.
let searcher = null, treeStale = false;
let board, turn, clock, selected = null, busy = true, gameOver = false;
// What render() paints.  Equal to `board` except while a move animation is in
// flight, when it lags behind by design -- the game state commits immediately
// and the display catches up through anim.playMove's intermediate boards.
let displayBoard;
// The 49 cell elements, built once by buildBoard() and thereafter only
// reclassified.  Rebuilding them per render (as this used to) destroys the
// stone elements every move, leaving nothing with an identity to animate.
const cells = [];
// null until the player picks one: there is no default opponent, and no game
// runs before the choice is made.  Everything that names the AI has to tolerate
// it (see aiName), and newGame() stops short of starting play.
let opponent = null;
// UAI move list for the game in progress, one entry per ply (passes included),
// so a finished game can be copied out and replayed against the Python engine.
let moves = [];
// Stauf's own cumulative move index this game.  CellGame varies its real search
// depth on moveCount % 3, so this must count *Stauf's* moves (not plies) and
// reset per game, matching how the game ladder drives it.
let staufMoves = 0;
let staufWorker = null, staufPending = null;
// Bumped by newGame().  An AI turn captures it before awaiting and drops its
// result if it changed meanwhile, so a search still in flight when you hit "New
// game" (or switch opponent) can't land a move on the fresh board.
let gen = 0;
const aiName = () => OPPONENTS[opponent]?.label ?? 'Opponent';

// Does the active session compute a *batch* correctly?  onnxruntime-web 1.20.x
// filled only the first slot of a batched WebGPU run and left the rest garbage,
// which no exception reports: the search just gets noise for 15 of every 16
// leaves and the net plays badly.  A broken EP is a property of the driver and
// browser as much as of ORT, so check at boot rather than trusting a version
// number, and drop to wasm if the answer is wrong.
//
// Two distinct positions, batched, each compared against itself run alone.
async function batchIsSane() {
  const one = () => {
    const o = new Float32Array(196);
    for (let c = 0; c < 49; c++) o[c * 4 + 2] = 1;   // side-to-move plane
    return o;
  };
  const a = one(), b = one();
  b[0] = 1; b[4 * 4 + 1] = 1;                        // two stones, so b differs from a
  const pair = new Float32Array(392);
  pair.set(a, 0); pair.set(b, 196);

  // Strictly sequential: an InferenceSession is not re-entrant, and overlapping
  // run() calls on one session wedge it.  Everything else here drives it one
  // batch at a time (see driveSearch), so this must too.
  const batched = await runNet(pair, 2);
  const soloA = await runNet(a, 1);
  const soloB = await runNet(b, 1);
  // Tolerance is loose: this catches "wrong item entirely", not fp drift, and
  // GPU and CPU kernels legitimately differ in the last few digits.
  return Math.abs(batched.value[0] - soloA.value[0]) < 1e-2
      && Math.abs(batched.value[1] - soloB.value[0]) < 1e-2;
}

async function runNet(obsF32, n) {
  const t = new ort.Tensor('float32', obsF32, [n, 7, 7, 4]);
  const res = await session.run({ obs: t });
  return { policy: res.policy_logits.data, value: res.value.data };
}

// ---- Stauf worker ---------------------------------------------------------
// Lazy loaded on first use, so visitors who only play net2 never fetch the
// module at all.  One request at a time (the game is strictly turn-based),
// so a single pending promise is sufficient.
function bootStauf() {
  if (staufWorker) return staufWorker.ready;
  const w = new Worker(new URL('./stauf.worker.mjs', import.meta.url), { type: 'module' });
  let resolveReady, rejectReady;
  const ready = new Promise((res, rej) => { resolveReady = res; rejectReady = rej; });
  w.onmessage = (e) => {
    const msg = e.data;
    if (msg.type === 'ready') return resolveReady();
    if (msg.type === 'error') {
      const err = new Error(msg.message);
      rejectReady(err);
      if (staufPending) { staufPending.reject(err); staufPending = null; }
      return;
    }
    if (msg.type === 'move' && staufPending) {
      const p = staufPending; staufPending = null;
      p.resolve(msg);
    }
  };
  w.onerror = (e) => rejectReady(new Error(e.message || 'Stauf worker failed to load'));
  staufWorker = { w, ready };
  w.postMessage({ type: 'init' });
  return ready;
}

function askStauf(asBlue) {
  return new Promise((resolve, reject) => {
    staufPending = { resolve, reject };
    staufWorker.w.postMessage({ type: 'move', board, asBlue, moveCount: staufMoves,
                               depth: OPPONENTS[opponent].depth });
  });
}

// ---- net2 (ORT + MCGS search) ---------------------------------------------
// Memoised on the promise, so concurrent callers share one load
// and a failure can be retried by clearing it.
function bootNet() {
  if (netReady) return netReady;
  netReady = (async () => {
    ort = await import(`${ORT_CDN}ort.webgpu.bundle.min.mjs`);
    ort.env.wasm.wasmPaths = ORT_CDN;          // fetch the ORT wasm binary from CDN too
    ort.env.wasm.numThreads = 1;               // single-thread → no COOP/COEP needed
    ort.env.logLevel = 'error';

    mod = await MicroMCTS();
    const bytes = new Uint8Array(await (await fetch('./models/net2.onnx')).arrayBuffer());

    // Probe for a core adapter rather than letting session creation throw:
    // compatibility-mode-only setups (Chromium without its Vulkan backend, so
    // Dawn falls back to ANGLE) expose navigator.gpu but yield no core adapter.
    const haveGPU = !!(await navigator.gpu?.requestAdapter().catch(() => null));
    if (haveGPU) {
      try {
        session = await ort.InferenceSession.create(bytes, { executionProviders: ['webgpu'] });
        if (await batchIsSane()) backend = 'webgpu';
        else { session = null; console.warn('WebGPU EP failed the batch check — falling back to wasm'); }
      } catch (err) {
        session = null;
        console.warn('WebGPU EP unavailable — falling back to wasm', err);
      }
    }
    if (!session) {
      session = await ort.InferenceSession.create(bytes, { executionProviders: ['wasm'] });
      backend = 'wasm';
    }
  })().catch(err => { netReady = null; throw err; });
  return netReady;
}

function boot() {
  // Read the control rather than assuming the default: browsers restore a
  // <select>'s value across a reload, so a reload mid-game keeps its opponent
  // instead of dropping back to the placeholder.
  opponent = opponentEl.value in OPPONENTS ? opponentEl.value : null;
  try { HUMAN = localStorage.getItem(SIDE_KEY) !== 'green'; } catch { /* private mode */ }
  syncSideUI();
  buildBoard();
  newGame();
}

// Push HUMAN into the parts of the page that name a colour: the switch's
// pressed state, the subtitle, and --me, which tints the move markers so the
// hints are in your colour rather than always blue.
function syncSideUI() {
  document.body.dataset.side = HUMAN ? 'blue' : 'green';
  sideNoteEl.textContent = `· you play ${colour(HUMAN)}`;
  for (const b of sideEl.querySelectorAll('button'))
    b.setAttribute('aria-pressed', String((b.dataset.side === 'blue') === HUMAN));
}

function newGame() {
  gen++;
  anim.cancelAll();          // a stone may be mid-flight; gen++ makes it give up
  board = engine.newBoard(); displayBoard = board;
  turn = true; clock = 0; selected = null; gameOver = false; busy = false;
  staufMoves = 0;
  moves = [];
  treeStale = true;          // drop the previous game's tree at the next search
  clearStatusHold();

  // No opponent yet: paint the starting position behind the veil and stop.
  // busy keeps the cells inert, so the board can't be played against nobody.
  if (!opponent) {
    busy = true;
    veilEl.hidden = false;
    metaEl.textContent = 'Pick an opponent to start a game.';
    render();
    setStatus('Select an opponent from the dropdown.');
    return;
  }

  veilEl.hidden = true;
  metaEl.textContent = `${aiName()} · ${OPPONENTS[opponent].meta}`;
  render();
  // Blue always moves first, so as Green you are waiting on the AI, not on you.
  setStatus(HUMAN ? `Your move (${colour(HUMAN)}).` : `${aiName()} moves first.`);
  maybeAiTurn();
}

// ---- status line ----------------------------------------------------------
// A line set with a hold stays put for at least that long.  Without it the AI's
// move report is wiped by the next prompt the instant the move animation ends,
// which is far too brief to read.  A line arriving during a hold waits its turn,
// and only the newest waiting line survives -- the status always describes the
// latest state, never a backlog.
const STATUS_HOLD = 1600;
const STICKY = Infinity;      // held until something explicitly supersedes it

let holdUntil = 0, queuedStatus = null, holdTimer = null;

function setStatus(text, hold = 0) {
  const now = performance.now();
  if (now < holdUntil) {
    queuedStatus = { text, hold };
    // A sticky hold has no expiry, so it gets no timer: it is released only by
    // clearStatusHold(), which drops the queue along with it.
    if (!holdTimer && holdUntil !== STICKY)
      holdTimer = setTimeout(releaseStatus, holdUntil - now);
    return;
  }
  statusEl.textContent = text;
  holdUntil = hold ? now + hold : 0;
}

function releaseStatus() {
  holdTimer = null; holdUntil = 0;
  const q = queuedStatus; queuedStatus = null;
  if (q) setStatus(q.text, q.hold);
}

// Drop the hold and anything waiting behind it: the caller is about to say
// something that supersedes both.
function clearStatusHold() {
  holdUntil = 0; queuedStatus = null;
  if (holdTimer) { clearTimeout(holdTimer); holdTimer = null; }
}

// Built once, at boot.  Listeners are bound per cell here rather than per
// render, so the hint maps they consult live at module scope instead of being
// captured in a closure that render() has to refresh.
function buildBoard() {
  boardEl.innerHTML = '';
  cells.length = 0;
  for (let y = 0; y < 7; y++) for (let x = 0; x < 7; x++) {
    const cell = document.createElement('div');
    cell.className = 'cell';
    cell.dataset.xy = `${x},${y}`;
    cell.appendChild(Object.assign(document.createElement('div'), { className: 'stone' }));
    cell.addEventListener('click', () => onCell(x, y));
    boardEl.appendChild(cell);
    cells.push(cell);
  }
}

// Current move hints, recomputed by render() and read by onCell().
let bySource = new Map(), dests = null;

function render() {
  const b = displayBoard;
  const { blue, green } = engine.countCells(b);
  scoreEl.innerHTML = `<span class="dot blue"></span>${blue} &nbsp; <span class="dot green"></span>${green}`;
  bySource = (!busy && !gameOver && turn === HUMAN) ? engine.legalMovesBySource(b, turn) : new Map();
  dests = selected ? new Map((bySource.get(selected) || []).map(d => [`${d.tx},${d.ty}`, d])) : null;

  for (let y = 0; y < 7; y++) for (let x = 0; x < 7; x++) {
    const i = y * 7 + x, cell = cells[i], key = `${x},${y}`;
    const blueP = b[i * 2 + 1], greenP = b[i * 2];
    cell.classList.toggle('has', !!(blueP || greenP));
    cell.classList.toggle('blue', !!blueP);
    cell.classList.toggle('green', !!greenP);
    cell.classList.toggle('selectable', bySource.has(key));
    cell.classList.toggle('selected', selected === key);
    const isDest = !!(dests && dests.has(key));
    cell.classList.toggle('dest', isDest);
    cell.classList.toggle('jump', isDest && dests.get(key).jump);
  }
}

// Paint an intermediate board mid-animation.  Handed to anim.playMove, which
// walks the display from the pre-move position to the post-move one.
function setDisplay(b) { displayBoard = b; render(); }

// Run a move's animation, having already committed it to `board`.  Returns
// false if the game moved on underneath us (New game, opponent switch).
async function animateMove(prev, action, mover, myGen) {
  await anim.playMove({ cells, boardEl, prev, next: board, action, turn: mover, setDisplay });
  if (myGen !== gen) return false;
  // playMove normally lands on `board` itself, so this is usually a no-op.  It
  // is the backstop for any path that bails out without a final setDisplay:
  // the invariant is that display == truth whenever no animation is running,
  // and leaving displayBoard behind without repainting would strand the board.
  if (displayBoard !== board) setDisplay(board);
  return true;
}

function onCell(x, y) {
  if (busy || gameOver || turn !== HUMAN) return;
  const key = `${x},${y}`;
  if (dests && dests.has(key)) { humanMove(dests.get(key).action); return; }
  if (bySource.has(key)) { selected = (selected === key) ? null : key; render(); }
  else { selected = null; render(); }
}

// The original throws a taunt whenever a turn takes five or more cells; this is
// the half of it you get to enjoy.  Counted as opponent stones that vanished,
// which is exactly "cells taken" and needs no separate capture list.  Sticky, so
// it rides out your own animation and the reply's thinking time, and is released
// by the AI's move report -- i.e. it stands until the next move is played.
const TAUNT_AT = 5;
function taunt(prev, mover) {
  const theirs = mover ? 'green' : 'blue';
  const taken = engine.countCells(prev)[theirs] - engine.countCells(board)[theirs];
  if (taken >= TAUNT_AT) setStatus('Curses!', STICKY);
}

// State commits synchronously; only the display waits.  The animation runs
// before the AI's search rather than alongside it: the wasm search blocks the
// main thread between net evaluations, which would visibly stutter the leap.
async function humanMove(action) {
  const myGen = gen, mover = turn, prev = board;
  board = engine.applyMove(board, action, turn);
  clock = engine.tickClock(clock, action);
  moves.push(engine.actionToUAI(action));
  selected = null;
  // Your move supersedes whatever the last one said, including a prompt still
  // queued behind the AI's move report.
  clearStatusHold();
  taunt(prev, mover);
  busy = true; render();          // drops the hints; the board still shows `prev`
  if (!await animateMove(prev, action, mover, myGen)) return;
  busy = false;
  advance();
}

async function maybeAiTurn() {
  if (gameOver || turn === HUMAN) return;
  const term = engine.checkTerminal(board, turn);
  if (term.terminal) return finish(term);
  if (engine.legalMoves(board, turn).length === 0) {   // the AI must pass
    clearStatusHold();
    setStatus(`${aiName()} passes.`, STATUS_HOLD);
    clock = engine.tickClock(clock, engine.PASS_ACTION);
    moves.push(engine.actionToUAI(engine.PASS_ACTION));
    turn = !turn; render(); return void afterMove();
  }

  busy = true; render();
  const thinking = opponent === 'net' ? `thinking (${SIMS} sims)…` : 'thinking…';
  setStatus(`${aiName()} is ${thinking}`);
  await nextPaint();                             // let your move + status render first

  const myGen = gen;
  let action, dt;
  try {
    if (opponent === 'net') {
      if (!session) { setStatus(`Loading ${aiName()}…`); await bootNet(); setStatus(`${aiName()} is ${thinking}`); }
      if (!searcher) searcher = engine.createSearcher(mod, CFG);
      // Safe here and only here: the game is turn-based and `busy` gates this
      // path, so no search can be in flight while we drop the tree.
      if (treeStale) { searcher.clear(); treeStale = false; }
      const t0 = performance.now();
      ({ action } = await searcher.search(runNet, board, turn, clock));
      dt = Math.round(performance.now() - t0);
      // We only get here with a legal move available (checked above), so an
      // empty result means the search itself failed -- an exhausted arena
      // returns all-zero visit counts rather than erroring.  Without this the
      // net would silently forfeit its turn as if it had passed.
      if (action < 0) throw new Error('search returned no move (arena exhausted?)');
    } else {
      if (!staufWorker) setStatus(`Loading ${aiName()}…`);
      await bootStauf();
      const res = await askStauf(turn);
      // The worker reports PASS_ACTION when CellGame produced no legal move;
      // normalise to the -1 that the net path uses for "no move".
      action = res.action === engine.PASS_ACTION ? -1 : res.action;
      dt = res.ms;
      staufMoves++;
    }
  } catch (err) {
    if (myGen !== gen) return;                   // abandoned game — stay quiet
    busy = false;
    clearStatusHold();
    setStatus(`${aiName()} failed: ${err.message}`);
    console.error(err);
    return;
  }

  if (myGen !== gen) return;                     // a new game started while we searched

  const mover = turn;
  if (action >= 0) {
    const prev = board;
    board = engine.applyMove(board, action, turn);
    clock = engine.tickClock(clock, action);
    if (!await animateMove(prev, action, mover, myGen)) return;
  }
  // Reported once the board has settled, and held: this is also what releases
  // an outstanding "Curses!", so the taunt survives until the reply lands.
  clearStatusHold();
  setStatus(`${aiName()} moved (${dt} ms${opponent === 'net' && backend ? `, ${backend}` : ''}).`,
            STATUS_HOLD);
  moves.push(engine.actionToUAI(action >= 0 ? action : engine.PASS_ACTION));
  busy = false;
  advance(true);
}

// Shared post-move flow: flip turn, check terminal / human-pass, hand off.
function advance(fromAi = false) {
  turn = !turn;
  render();
  afterMove(fromAi);
}

function afterMove(fromAi = false) {
  const term = engine.checkTerminal(board, turn);
  if (term.terminal) return finish(term);
  if (turn === HUMAN && engine.legalMoves(board, turn).length === 0) {
    clearStatusHold();
    setStatus('You have no moves — you pass.', STATUS_HOLD);
    clock = engine.tickClock(clock, engine.PASS_ACTION);
    moves.push(engine.actionToUAI(engine.PASS_ACTION));
    turn = !turn; render();
    return maybeAiTurn();
  }
  if (turn === HUMAN) { if (!fromAi) return; setStatus(`Your move (${colour(HUMAN)}).`); }
  else maybeAiTurn();
}

function finish(term) {
  gameOver = true; busy = false; selected = null; render();
  clearStatusHold();          // the result outranks whatever is on screen
  const { blue, green } = engine.countCells(board);
  const mine = HUMAN ? blue : green, theirs = HUMAN ? green : blue;
  setStatus(mine === theirs ? `Draw — ${mine}–${theirs}.`
    : mine > theirs ? `You win ${mine}–${theirs}! 🎉`
    : `${aiName()} wins ${theirs}–${mine}.`);
}

// ---- game export ----------------------------------------------------------
// A UAI `position` line plus enough header to identify ourselves.
// Helpful to diagnose a browser-side game after the fact.
function gameText() {
  const { blue, green } = engine.countCells(board);
  const who = (t) => (t === HUMAN ? 'human' : aiName());
  const result = !gameOver ? `in progress, ${blue}-${green}`
    : blue === green ? `draw ${blue}-${green}`
    : blue > green ? `Blue (${who(true)}) wins ${blue}-${green}`
    : `Green (${who(false)}) wins ${green}-${blue}`;
  return [
    `# t7g-ml webapp · ${new Date().toISOString().replace(/\.\d+Z$/, 'Z')}`,
    `# Blue: ${who(true)} · Green: ${who(false)} (${aiName()}: ${OPPONENTS[opponent]?.meta ?? 'not selected'})`,
    `# result: ${result} · halfmove clock ${clock}`,
    moves.length ? `position startpos moves ${moves.join(' ')}` : 'position startpos',
    `# final fen: ${engine.boardToFEN(board, turn)}`,
  ].join('\n');
}

$('copy-game').addEventListener('click', async () => {
  const text = gameText();
  try {
    await navigator.clipboard.writeText(text);
    setStatus('Game copied to clipboard.');
  } catch {
    // The Clipboard API needs a secure context and can be denied outright.
    // Fall back to a selected textarea so the move list is still recoverable
    // by hand rather than lost.
    const ta = document.createElement('textarea');
    ta.value = text;
    ta.setAttribute('readonly', '');
    ta.style.cssText = 'position:fixed;z-index:9;top:50%;left:50%;transform:translate(-50%,-50%);'
                     + 'width:min(90vw,460px);height:10em;font:12px/1.4 ui-monospace,monospace';
    document.body.appendChild(ta);
    ta.focus(); ta.select();
    ta.addEventListener('blur', () => ta.remove(), { once: true });
    setStatus('Clipboard blocked — copy the text, then tap outside it.');
  }
});

$('new-game').addEventListener('click', newGame);
// Switching opponent restarts: Stauf's depth cycling is keyed to its own move
// count, so handing it a game already in progress would misrepresent it.
opponentEl.addEventListener('change', () => {
  opponent = opponentEl.value in OPPONENTS ? opponentEl.value : null;
  newGame();          // safe mid-search: the gen guard discards the stale result
});
// Swapping colours restarts too — you can't hand over a game in progress
// without also handing over its position.
sideEl.addEventListener('click', (e) => {
  const btn = e.target.closest('button[data-side]');
  if (!btn) return;
  const wantBlue = btn.dataset.side === 'blue';
  if (wantBlue === HUMAN) return;
  HUMAN = wantBlue;
  try { localStorage.setItem(SIDE_KEY, HUMAN ? 'blue' : 'green'); } catch { /* private mode */ }
  syncSideUI();
  newGame();
});
boot();
