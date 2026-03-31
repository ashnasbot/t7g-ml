# Performance Notes

## Inference throughput (bench_sims.py)

Measured GPU vs CPU inference speed across batch sizes and pool sizes, to
determine whether GPU acceleration pays off at MCTS batch sizes and what
pool size maximises sims/sec.

**Conclusions:**
- CPU throughput: ~2k sims/sec (single-game, uncompiled).
- GPU throughput: up to ~40k sims/sec (ideal fully-batched); realistic ceiling
  ~28k sims/sec accounting for pipeline stalls and Python overhead.
- `torch.compile` gives 2–3× speedup over uncompiled network; compiled GPU is
  always the preferred training path. CPU (uncompiled) is for test/debug only.
- Pipelining is critical: CPU must stay waiting on GPU. If the relationship
  inverts (GPU waiting on CPU), throughput halves or worse.
- `POOL_SIZE = 32` chosen as the practical optimum for this hardware.

**Setup:** DualHeadNetwork (128 filters, 6 res blocks), 1225-action head,
`torch.compile` enabled where supported.

---

## CPU sims/sec by num_simulations (iter_0100, single-threaded, no pool)

3 games per setting, temperature=0, C_PUCT=0.75, K=8.

| num_sims | avg moves/game | s/game | sims/sec | ms/move |
|---|---|---|---|---|
|    50 |  91.0 |  7.8s |  583 |   85.7ms |
|   100 | 101.0 | 15.1s |  670 |  149.3ms |
|   250 |  92.7 | 37.7s |  614 |  407.1ms |
|   500 |  54.7 | 50.5s |  541 |  923.5ms |
|  1000 |  31.0 | 58.7s |  528 | 1894.7ms |
|  2000 |  18.7 | 63.8s |  586 | 3415.8ms |

---

## micro3 move timing (bench_solvers.py, arena.py --bench-only)

Timing range: lower bound from opening position (50 reps), upper bound from a
fixed mid-game position (8 reps, 12 random moves from start, seed 42).
Single-threaded, Windows 11 AMD CPU (`-march=native`).

**ms/move by depth:**

| depth | opening | mid-game |
|---|---|---|
| 2 |  0.01 |   0.32 |
| 3 |  0.02 |   1.40 |
| 4 |  0.13 |  16.12 |
| 5 |  1.16 |  73.91 |
| 6 |     — | 731.37 |

**Conclusions:**
- Mid-game branching factor is ~60–70× more expensive than the opening at depth 5.
- Practical mid-game budget: depth 3 ≈ 1.4 ms, depth 5 ≈ 74 ms per move.

---

## Policy distillation calibration (validate_policy_labels.py)

Measured label entropy and top-move agreement vs MM-5 oracle at various
(depth, temperature) pairs for both micro3 and micro4 DLLs.

**Target:** match Gumbel K=8 ceiling = log(8) ≈ 2.08 nats.

**Conclusions:**
- `POLICY_DISTILL_DEPTH = 3` (odd depth avoids even-depth horizon artefact).
- `POLICY_DISTILL_TEMP = 0.075` produces median entropy ≈ 2.08 nats for micro4.
- Top-move agreement between MM-3 and MM-5: **TODO**%

---

## Move quality distribution (analyze_move_quality.py, analyze_move_winrate.py)

Sampled ~500 positions at MM-5 depth, measuring best-vs-2nd-best score gap
and capture-heuristic ranking accuracy.

**Conclusions:**
- TODO (fill in: typical gap, how often capture heuristic ranks best move in top-K)
