// Browser entry: wires onnxruntime-web (WebGPU, wasm fallback) + the micro_mcts
// wasm search + engine.mjs into a human-vs-AI game against a choice of
// opponents.  Fully client-side — no server.  See engine.mjs for the rules +
// search driver (shared with the node end-to-end test).
//
// The onnxruntime-web runtime is pulled from CDN rather committed to the repo, pinned to ORT_VER.
//
// Both opponents load lazily, on first use: playing Stauf should not drag in
// ORT and the model, and playing net2 should not fetch the Stauf module.  That
// also means the page should work offline.
import * as engine from './engine.mjs';
import MicroMCTS from './micro_mcts.mjs';

const ORT_VER = '1.20.1';
const ORT_CDN = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VER}/dist/`;

const SIMS = 500;                 // canonical net2 config (eval_db DEFAULT_CONFIG)
const CFG = { sims: SIMS, cPuct: 1.3, gumbelK: 16, completionN0: 50.0, sigmaScale: 1.0, clockObs: true };
const HUMAN = true;               // human plays Blue and moves first; the AI plays Green

// Selectable opponents.  Stauf is the original T7G AI (via ScummVM's Groovie
// CellGame), and lives behind a worker because it is GPLv3 while this file
// is not -- see stauf.worker.mjs.
const OPPONENTS = {
  net:   { label: 'AshnasBot', meta: 'net2 · 500-sim MCGS' },
  stauf: { label: 'Stauf',     meta: 'the original 7th Guest AI · ScummVM CellGame · GPLv3' },
};

const $ = (id) => document.getElementById(id);
// Resolve after the browser has actually painted (two rAFs) so the player's
// move is on screen before the synchronous search setup can block the thread.
const nextPaint = () => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)));
const boardEl = $('board'), statusEl = $('status'), scoreEl = $('score');
const metaEl = $('meta'), opponentEl = $('opponent');

let ort, mod, session, netReady = null;
// One MCGS instance for the page, so its transposition table survives across
// moves the way self-play and eval keep theirs.  Cleared at the start of each
// game rather than destroyed.
// The clear is deferred to the next search because it is only safe when none is in flight.
let searcher = null, treeStale = false;
let board, turn, clock, selected = null, busy = true, gameOver = false;
let opponent = 'stauf';
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
const aiName = () => OPPONENTS[opponent].label;

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
    staufWorker.w.postMessage({ type: 'move', board, asBlue, moveCount: staufMoves });
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
    try {
      session = await ort.InferenceSession.create(bytes, { executionProviders: ['webgpu'] });
    } catch {
      session = await ort.InferenceSession.create(bytes, { executionProviders: ['wasm'] });
    }
  })().catch(err => { netReady = null; throw err; });
  return netReady;
}

function boot() {
  // Read the control rather than assuming the default: browsers restore a
  // <select>'s value across a reload, so this must follow the restored UI.
  opponent = opponentEl.value in OPPONENTS ? opponentEl.value : 'stauf';
  newGame();
}

function newGame() {
  gen++;
  board = engine.newBoard(); turn = true; clock = 0; selected = null; gameOver = false; busy = false;
  staufMoves = 0;
  moves = [];
  treeStale = true;          // drop the previous game's tree at the next search
  metaEl.textContent = `${aiName()} · ${OPPONENTS[opponent].meta}`;
  render();
  setStatus('Your move (Blue).');
  maybeAiTurn();
}

function setStatus(t) { statusEl.textContent = t; }

function render() {
  const { blue, green } = engine.countCells(board);
  scoreEl.innerHTML = `<span class="dot blue"></span>${blue} &nbsp; <span class="dot green"></span>${green}`;
  const bySource = (!busy && !gameOver && turn === HUMAN) ? engine.legalMovesBySource(board, turn) : new Map();
  const dests = selected ? new Map((bySource.get(selected) || []).map(d => [`${d.tx},${d.ty}`, d])) : null;

  boardEl.innerHTML = '';
  for (let y = 0; y < 7; y++) for (let x = 0; x < 7; x++) {
    const cell = document.createElement('div');
    cell.className = 'cell';
    const blueP = board[(y * 7 + x) * 2 + 1], greenP = board[(y * 7 + x) * 2];
    if (blueP) cell.classList.add('has', 'blue');
    else if (greenP) cell.classList.add('has', 'green');

    const key = `${x},${y}`;
    if (bySource.has(key)) cell.classList.add('selectable');
    if (selected === key) cell.classList.add('selected');
    if (dests && dests.has(key)) {
      cell.classList.add('dest');
      cell.dataset.dest = key;
      if (dests.get(key).jump) cell.classList.add('jump');
    }
    cell.dataset.xy = key;
    cell.addEventListener('click', () => onCell(x, y, bySource, dests));
    boardEl.appendChild(cell);
  }
}

function onCell(x, y, bySource, dests) {
  if (busy || gameOver || turn !== HUMAN) return;
  const key = `${x},${y}`;
  if (dests && dests.has(key)) { humanMove(dests.get(key).action); return; }
  if (bySource.has(key)) { selected = (selected === key) ? null : key; render(); }
  else { selected = null; render(); }
}

function humanMove(action) {
  board = engine.applyMove(board, action, turn);
  clock = engine.tickClock(clock, action);
  moves.push(engine.actionToUAI(action));
  selected = null;
  advance();
}

async function maybeAiTurn() {
  if (gameOver || turn === HUMAN) return;
  const term = engine.checkTerminal(board, turn);
  if (term.terminal) return finish(term);
  if (engine.legalMoves(board, turn).length === 0) {   // the AI must pass
    setStatus(`${aiName()} passes.`);
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
    setStatus(`${aiName()} failed: ${err.message}`);
    console.error(err);
    return;
  }

  if (myGen !== gen) return;                     // a new game started while we searched

  if (action >= 0) { board = engine.applyMove(board, action, turn); clock = engine.tickClock(clock, action); }
  moves.push(engine.actionToUAI(action >= 0 ? action : engine.PASS_ACTION));
  busy = false;
  setStatus(`${aiName()} moved (${dt} ms).`);
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
    setStatus('You have no moves — you pass.');
    clock = engine.tickClock(clock, engine.PASS_ACTION);
    moves.push(engine.actionToUAI(engine.PASS_ACTION));
    turn = !turn; render();
    return maybeAiTurn();
  }
  if (turn === HUMAN) { if (!fromAi) return; setStatus('Your move (Blue).'); }
  else maybeAiTurn();
}

function finish(term) {
  gameOver = true; busy = false; selected = null; render();
  const { blue, green } = engine.countCells(board);
  const humanWon = blue > green, draw = blue === green;
  setStatus(draw ? `Draw — ${blue}–${green}.` : humanWon ? `You win ${blue}–${green}! 🎉` : `${aiName()} wins ${green}–${blue}.`);
}

// ---- game export ----------------------------------------------------------
// A UAI `position` line plus enough header to identify ourselves.
// Helpful to diagnose a browser-side game after the fact.
function gameText() {
  const { blue, green } = engine.countCells(board);
  const result = !gameOver ? `in progress, ${blue}-${green}`
    : blue === green ? `draw ${blue}-${green}`
    : blue > green ? `Blue (human) wins ${blue}-${green}`
    : `Green (${aiName()}) wins ${green}-${blue}`;
  return [
    `# t7g-ml webapp · ${new Date().toISOString().replace(/\.\d+Z$/, 'Z')}`,
    `# Blue: human · Green: ${aiName()} (${OPPONENTS[opponent].meta})`,
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
  opponent = opponentEl.value in OPPONENTS ? opponentEl.value : 'stauf';
  newGame();          // safe mid-search: the gen guard discards the stale result
});
boot();
