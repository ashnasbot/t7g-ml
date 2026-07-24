// Browser entry: wires onnxruntime-web (WebGPU, wasm fallback) + the micro_mcts
// wasm search + engine.mjs into a human-vs-net2 game.  Fully client-side — no
// server.  See engine.mjs for the rules + search driver (shared with the node
// end-to-end test).
//
// The onnxruntime-web runtime (its 21 MB wasm binary) is loaded from the
// jsdelivr CDN rather than committed to the repo — pinned to ORT_VER for
// reproducibility.  Everything else (our wasm engine, the net2 model) is served
// from the same origin as this page.
import * as engine from './engine.mjs';
import MicroMCTS from './micro_mcts.mjs';

const ORT_VER = '1.20.1';
const ORT_CDN = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VER}/dist/`;
const ort = await import(`${ORT_CDN}ort.webgpu.bundle.min.mjs`);

const SIMS = 500;                 // canonical net2 config (eval_db DEFAULT_CONFIG)
const CFG = { sims: SIMS, cPuct: 1.3, gumbelK: 16, completionN0: 50.0, sigmaScale: 1.0, clockObs: true };
const HUMAN = true;               // human plays Blue and moves first; net2 plays Green

const $ = (id) => document.getElementById(id);
// Resolve after the browser has actually painted (two rAFs) so the player's
// move is on screen before the synchronous search setup can block the thread.
const nextPaint = () => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)));
const boardEl = $('board'), statusEl = $('status'), scoreEl = $('score');

let mod, session, board, turn, clock, selected = null, busy = true, gameOver = false;

async function runNet(obsF32, n) {
  const t = new ort.Tensor('float32', obsF32, [n, 7, 7, 4]);
  const res = await session.run({ obs: t });
  return { policy: res.policy_logits.data, value: res.value.data };
}

async function boot() {
  setStatus('Loading engine…');
  ort.env.wasm.wasmPaths = ORT_CDN;            // fetch the ORT wasm binary from CDN too
  ort.env.wasm.numThreads = 1;                 // single-thread → no COOP/COEP needed
  ort.env.logLevel = 'error';

  mod = await MicroMCTS();
  const bytes = new Uint8Array(await (await fetch('./models/net2.onnx')).arrayBuffer());

  try {
    session = await ort.InferenceSession.create(bytes, { executionProviders: ['webgpu'] });
  } catch {
    session = await ort.InferenceSession.create(bytes, { executionProviders: ['wasm'] });
  }

  newGame();
}

function newGame() {
  board = engine.newBoard(); turn = true; clock = 0; selected = null; gameOver = false; busy = false;
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
  selected = null;
  advance();
}

async function maybeAiTurn() {
  if (gameOver || turn === HUMAN) return;
  const term = engine.checkTerminal(board, turn);
  if (term.terminal) return finish(term);
  if (engine.legalMoves(board, turn).length === 0) {   // AshnasBot must pass
    setStatus('AshnasBot passes.'); clock = engine.tickClock(clock, engine.PASS_ACTION);
    turn = !turn; render(); return void afterMove();
  }
  busy = true; render(); setStatus(`AshnasBot is thinking (${SIMS} sims)…`);
  await nextPaint();                             // let your move + status render first
  const t0 = performance.now();
  const { action } = await engine.searchMove(mod, runNet, board, turn, clock, CFG);
  const dt = Math.round(performance.now() - t0);
  if (action >= 0) { board = engine.applyMove(board, action, turn); clock = engine.tickClock(clock, action); }
  busy = false;
  setStatus(`AshnasBot moved (${dt} ms).`);
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
  setStatus(draw ? `Draw — ${blue}–${green}.` : humanWon ? `You win ${blue}–${green}! 🎉` : `AshnasBot wins ${green}–${blue}.`);
}

$('new-game').addEventListener('click', newGame);
boot().catch(e => { setStatus('Error: ' + e.message); console.error(e); });
