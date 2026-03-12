"""
Benchmark MCTS throughput at different simulation counts using the live
worker + InferenceServer model. Runs N_GAMES games per sim count, reports
wall time, games/sec, and sims/sec. Loads the latest checkpoint by default.

Usage:
    python scripts/benchmark_sims.py
    python scripts/benchmark_sims.py --checkpoint models/mcts/iter_0200.pt
"""
import argparse
import multiprocessing
import sys
import time
from collections import deque

import torch
torch.set_float32_matmul_precision('high')

sys.path.insert(0, ".")
from lib.dual_network import DualHeadNetwork                # noqa: E402
from lib.inference_server import InferenceServer            # noqa: E402
from lib.remote_mcts import RemoteMCTS                      # noqa: E402
from lib.t7g import new_board, apply_move, check_terminal   # noqa: E402
from lib.t7g import action_masks                            # noqa: E402

SIM_COUNTS   = [750]
NUM_WORKERS  = 4
N_GAMES      = 10          # games per sim count
C_PUCT       = 1.5
DIRICHLET_ALPHA = 0.8

# ── per-process state (same pattern as train_mcts.py) ──────────────────────

_worker_id:      int                   = -1
_request_queue:  multiprocessing.Queue = None   # type: ignore[assignment]
_result_queue:   multiprocessing.Queue = None   # type: ignore[assignment]

# ── per-process state for local-model workers ───────────────────────────────
_local_network = None


def _local_worker_init(state_dict, do_compile):
    global _local_network
    import torch
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("high")
    device = _get_device()
    from lib.dual_network import DualHeadNetwork
    net = DualHeadNetwork(num_actions=1225)
    net.load_state_dict(state_dict)
    net.to(device)
    net.eval()
    if do_compile:
        try:
            net = torch.compile(net, mode="reduce-overhead")
        except Exception:
            pass
    _local_network = net


def _local_worker_task(args):
    num_simulations, = args
    from lib.mcts import MCTS
    from lib.t7g import new_board, apply_move, check_terminal, action_masks
    import time
    mcts = MCTS(_local_network, num_simulations=num_simulations,  # type: ignore[arg-type]
                c_puct=C_PUCT, dirichlet_alpha=DIRICHLET_ALPHA, dirichlet_epsilon=0.0)
    board = new_board()
    turn = True
    moves = 0
    total_sims = 0
    t0 = time.time()
    while moves < 150:
        is_terminal, _ = check_terminal(board, turn)
        if is_terminal:
            break
        if not any(action_masks(board, turn)):
            turn = not turn
            continue
        action_probs = mcts.search(board, turn)
        total_sims += num_simulations
        action = mcts.select_action(action_probs, temperature=0)
        mcts.advance_tree(action)
        board = apply_move(board, action, turn)
        turn = not turn
        moves += 1
    return total_sims, time.time() - t0


def run_local_model_benchmark(state_dict, num_simulations: int,
                              num_workers: int, n_games: int,
                              do_compile: bool = False) -> dict:
    """Workers each hold their own local network copy — no InferenceServer."""
    t0 = time.time()
    with multiprocessing.Pool(
        processes=num_workers,
        initializer=_local_worker_init,
        initargs=(state_dict, do_compile),
    ) as pool:
        results = pool.map(_local_worker_task, [(num_simulations,)] * n_games)
    wall_time = time.time() - t0  # true wall clock: pool create → all games done

    total_sims = sum(r[0] for r in results)
    total_time = sum(r[1] for r in results)
    return {
        "sims_per_sec_wall":  total_sims / wall_time,   # actual throughput seen from outside
        "sims_per_sec_cpu":   total_sims / total_time,  # total CPU-sims / total CPU-time
        "wall_time":          wall_time,
    }


def _worker_init(id_queue, request_queue, result_queues):
    global _worker_id, _request_queue, _result_queue
    torch.set_num_threads(1)
    _worker_id      = id_queue.get()
    _request_queue  = request_queue
    _result_queue   = result_queues[_worker_id]


def _play_game(num_simulations: int) -> tuple[int, float]:
    """Play one complete game; return (move_count, elapsed_seconds)."""
    mcts = RemoteMCTS(
        worker_id=_worker_id,
        request_queue=_request_queue,
        result_queue=_result_queue,
        num_simulations=num_simulations,
        c_puct=C_PUCT,
        dirichlet_alpha=DIRICHLET_ALPHA,
        dirichlet_epsilon=0.0,
    )
    board = new_board()
    turn  = True
    moves = 0
    t0    = time.time()
    while True:
        is_terminal, _ = check_terminal(board, turn)
        if is_terminal or moves > 300:
            break
        if not any(action_masks(board, turn)):
            turn = not turn
            continue
        probs  = mcts.search(board, turn)
        action = mcts.select_action(probs, temperature=0)
        mcts.advance_tree(action)
        board  = apply_move(board, action, turn)
        turn   = not turn
        moves += 1
    return moves, time.time() - t0


def _worker_task(args):
    num_simulations, = args
    return _play_game(num_simulations)


def run_benchmark(state_dict, num_simulations: int) -> dict:
    request_queue: multiprocessing.Queue = multiprocessing.Queue()
    result_queues = [multiprocessing.Queue() for _ in range(NUM_WORKERS)]
    id_queue: multiprocessing.Queue = multiprocessing.Queue()
    for i in range(NUM_WORKERS):
        id_queue.put(i)

    server = InferenceServer(state_dict, request_queue, result_queues, NUM_WORKERS)
    server.start()

    game_moves = []
    game_times = []

    try:
        with multiprocessing.Pool(
            processes=NUM_WORKERS,
            initializer=_worker_init,
            initargs=(id_queue, request_queue, result_queues),
        ) as pool:
            pending = deque()
            for _ in range(min(NUM_WORKERS, N_GAMES)):
                pending.append(pool.apply_async(_worker_task, ((num_simulations,),)))

            completed = 0
            while completed < N_GAMES:
                moves, elapsed = pending.popleft().get()
                game_moves.append(moves)
                game_times.append(elapsed)
                completed += 1
                if completed + len(pending) < N_GAMES:
                    pending.append(pool.apply_async(_worker_task, ((num_simulations,),)))
    finally:
        for _ in range(NUM_WORKERS):
            request_queue.put(None)
        server.join(timeout=5)
        if server.is_alive():
            server.terminate()

    avg_moves   = sum(game_moves) / len(game_moves)
    avg_time    = sum(game_times) / len(game_times)
    total_sims  = sum(m * num_simulations for m in game_moves)
    total_time  = sum(game_times)
    return {
        "sims":          num_simulations,
        "avg_moves":     avg_moves,
        "avg_time_s":    avg_time,
        "games_per_sec": N_GAMES / total_time,
        "sims_per_sec":  total_sims / total_time,
    }


def _get_device() -> "torch.device":
    if torch.cuda.is_available():
        return torch.device("cuda")
    try:
        import torch_directml  # type: ignore[import-untyped]
        return torch_directml.device()
    except ImportError:
        return torch.device("cpu")


def _time_inference(net, dummy: torch.Tensor,
                    n_calls: int, device: "torch.device") -> float:
    """Returns elapsed seconds for n_calls forward passes."""
    with torch.no_grad():
        for _ in range(50):          # warmup
            net(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        for _ in range(n_calls):
            net(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return time.time() - t0


def run_direct_inference_benchmark(network: DualHeadNetwork, n_calls: int = 2000) -> dict:
    """
    Bypass queues entirely: measure raw single-call GPU throughput at batch=1,
    with and without torch.compile, to see if compilation helps.
    """
    device = _get_device()
    network = network.to(device)
    network.eval()

    dummy = torch.zeros(1, 7, 7, 4, dtype=torch.float32, device=device)
    dummy[0, :, :, 1] = 1.0

    # Eager baseline
    elapsed_eager = _time_inference(network, dummy, n_calls, device)

    # torch.compile with cudagraphs (no Triton needed on Windows)
    compile_label = "n/a"
    calls_compiled = None
    try:
        compiled = torch.compile(network, mode="reduce-overhead")
        elapsed_compiled = _time_inference(compiled, dummy, n_calls, device)
        calls_compiled = n_calls / elapsed_compiled
        compile_label = f"{calls_compiled:.0f} calls/s  ({1000*elapsed_compiled/n_calls:.3f}ms)"
    except Exception as e:
        compile_label = f"failed: {type(e).__name__}: {e}"

    # TF32: free matmul speedup on Ampere+ GPUs, no precision change for convolutions
    tf32_label = "n/a"
    calls_tf32 = None
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")  # enables TF32
        elapsed_tf32 = _time_inference(network, dummy, n_calls, device)
        calls_tf32 = n_calls / elapsed_tf32
        tf32_label = f"{calls_tf32:.0f} calls/s  ({1000*elapsed_tf32/n_calls:.3f}ms)"
        torch.set_float32_matmul_precision("highest")  # restore

    calls_eager = n_calls / elapsed_eager
    return {
        "device":         str(device),
        "n_calls":        n_calls,
        "calls_per_sec":  calls_eager,
        "ms_per_call":    1000 * elapsed_eager / n_calls,
        "compile_label":  compile_label,
        "calls_compiled": calls_compiled,
        "tf32_label":     tf32_label,
        "calls_tf32":     calls_tf32,
    }


def run_singlethread_game_benchmark(network: torch.nn.Module, num_simulations: int,
                                    n_games: int = 6, compile: bool = False) -> dict:
    """
    Run full self-play games single-threaded with a local network — no queues,
    no IPC. Optionally compile the network first. Measures sims/sec to compare
    directly against the multi-worker server number.
    """
    device = _get_device()
    network = network.to(device)
    network.eval()
    net: object = network
    if compile:
        try:
            net = torch.compile(network, mode="reduce-overhead")
        except Exception as e:
            return {"error": str(e)}

    from lib.mcts import MCTS
    from lib.t7g import new_board, apply_move, check_terminal, action_masks

    total_sims = 0
    t0 = time.time()

    for _ in range(n_games):
        mcts = MCTS(net, num_simulations=num_simulations, c_puct=C_PUCT,  # type: ignore[arg-type]
                    dirichlet_alpha=DIRICHLET_ALPHA, dirichlet_epsilon=0.0)
        board = new_board()
        turn = True
        moves = 0
        while moves < 150:
            is_terminal, _ = check_terminal(board, turn)
            if is_terminal:
                break
            if not any(action_masks(board, turn)):
                turn = not turn
                continue
            action_probs = mcts.search(board, turn)
            total_sims += num_simulations
            action = mcts.select_action(action_probs, temperature=0)
            mcts.advance_tree(action)
            board = apply_move(board, action, turn)
            turn = not turn
            moves += 1

    elapsed = time.time() - t0
    return {
        "sims_per_sec": total_sims / elapsed,
        "elapsed":      elapsed,
        "compiled":     compile,
    }


def run_cpu_inference_benchmark(network: DualHeadNetwork,
                                n_calls: int = 3000,
                                thread_counts: list[int] | None = None) -> list[dict]:
    """
    Measure raw batch=1 inference throughput on CPU at varying thread counts.
    Useful for finding the sweet-spot number of threads per worker.
    """
    if thread_counts is None:
        thread_counts = [1, 2, 4, 8]
    net = network.to("cpu").eval()
    dummy = torch.zeros(1, 7, 7, 4)
    results = []
    for t in thread_counts:
        torch.set_num_threads(t)
        with torch.no_grad():
            for _ in range(100):   # warmup
                net(dummy)
        t0 = time.time()
        with torch.no_grad():
            for _ in range(n_calls):
                net(dummy)
        elapsed = time.time() - t0
        results.append({
            "threads":      t,
            "calls_per_sec": n_calls / elapsed,
            "ms_per_call":   1000 * elapsed / n_calls,
        })
    torch.set_num_threads(1)  # restore
    return results


def run_onnx_inference_benchmark(network: DualHeadNetwork,
                                 n_calls: int = 3000,
                                 thread_counts: list[int] | None = None) -> list[dict] | str:
    """
    Export to ONNX, run with OnnxRuntime CPUExecutionProvider at varying
    intra_op thread counts.  Returns an error string if onnxruntime is missing.
    """
    try:
        import onnxruntime as ort  # type: ignore[import-untyped]
    except ImportError:
        return "onnxruntime not installed (pip install onnxruntime)"

    import tempfile, os
    if thread_counts is None:
        thread_counts = [1, 2, 4, 8]

    net = network.to("cpu").eval()
    dummy = torch.zeros(1, 7, 7, 4)

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        onnx_path = f.name
    try:
        torch.onnx.export(
            net, dummy, onnx_path,
            input_names=["obs"], output_names=["policy", "value"],
            dynamic_axes={"obs": {0: "batch"}},
            opset_version=17,
        )

        results = []
        dummy_np = dummy.numpy()
        for t in thread_counts:
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = t
            opts.inter_op_num_threads = 1
            sess = ort.InferenceSession(onnx_path, sess_options=opts,
                                        providers=["CPUExecutionProvider"])
            for _ in range(100):   # warmup
                sess.run(None, {"obs": dummy_np})
            t0 = time.time()
            for _ in range(n_calls):
                sess.run(None, {"obs": dummy_np})
            elapsed = time.time() - t0
            results.append({
                "threads":       t,
                "calls_per_sec": n_calls / elapsed,
                "ms_per_call":   1000 * elapsed / n_calls,
            })
        return results
    finally:
        os.unlink(onnx_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="models/mcts/iter_0200.pt")
    args = parser.parse_args()

    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, weights_only=False, map_location="cpu")
    network = DualHeadNetwork(num_actions=1225)
    network.load_state_dict(checkpoint["network"])
    network.eval()
    state_dict = {k: v.cpu() for k, v in network.state_dict().items()}

    # --- CPU raw inference: PyTorch vs ONNX Runtime at varying thread counts ---
    print("\n--- CPU batch=1 inference (no queue overhead) ---")
    print(f"  {'Backend':<18} {'Threads':>7}  {'calls/s':>9}  {'ms/call':>8}")
    print("  " + "-" * 48)

    pt_results = run_cpu_inference_benchmark(network, n_calls=3000)
    for r in pt_results:
        print(f"  {'PyTorch (eager)':<18} {r['threads']:>7}  "
              f"{r['calls_per_sec']:>9.0f}  {r['ms_per_call']:>8.3f}ms")

    ort_results = run_onnx_inference_benchmark(network, n_calls=3000)
    if isinstance(ort_results, str):
        print(f"  ONNX Runtime: {ort_results}")
    else:
        for r in ort_results:
            print(f"  {'ONNX Runtime':<18} {r['threads']:>7}  "
                  f"{r['calls_per_sec']:>9.0f}  {r['ms_per_call']:>8.3f}ms")

    # GPU reference point (if available)
    gpu_device = _get_device()
    if gpu_device.type != "cpu":
        net_gpu = network.to(gpu_device).eval()
        dummy_gpu = torch.zeros(1, 7, 7, 4, device=gpu_device)
        elapsed = _time_inference(net_gpu, dummy_gpu, 3000, gpu_device)
        cps = 3000 / elapsed
        print(f"  {'GPU (eager)':<18} {'n/a':>7}  {cps:>9.0f}  {1000/cps:>8.3f}ms  "
              f"[{str(gpu_device)}]")
        try:
            net_compiled = torch.compile(net_gpu, mode="reduce-overhead")
            elapsed_c = _time_inference(net_compiled, dummy_gpu, 3000, gpu_device)
            cps_c = 3000 / elapsed_c
            print(f"  {'GPU (compiled)':<18} {'n/a':>7}  {cps_c:>9.0f}  {1000/cps_c:>8.3f}ms")
        except Exception as e:
            print(f"  GPU (compiled): failed: {e}")
    print()

    # --- Single-threaded direct inference baseline ---
    # print("\n--- Direct inference (no queue, batch=1) ---")
    # direct = run_direct_inference_benchmark(network)
    # print(f"  Device:          {direct['device']}")
    # print(f"  Eager:           {direct['calls_per_sec']:.0f} calls/s  ({direct['ms_per_call']:.3f}ms)")
    # print(f"  Compiled:        {direct['compile_label']}")
    # if direct['calls_compiled']:
    #     print(f"  Compile speedup: {direct['calls_compiled'] / direct['calls_per_sec']:.2f}x")
    # print(f"  TF32:            {direct['tf32_label']}")
    # if direct['calls_tf32']:
    #     print(f"  TF32 speedup:    {direct['calls_tf32'] / direct['calls_per_sec']:.2f}x")
    # print(f"  (multi-worker target to beat: ~2200 sims/sec)\n")

    # --- Single-threaded full-game benchmark (eager vs compiled) ---
    # print(f"--- Single-thread game benchmark ({SIM_COUNTS[0]} sims, {N_GAMES} games) ---")
    # st_eager = run_singlethread_game_benchmark(network, SIM_COUNTS[0], n_games=N_GAMES)
    # print(f"  Eager:    {st_eager['sims_per_sec']:.0f} sims/sec  ({st_eager['elapsed']:.1f}s)")
    # st_compiled = run_singlethread_game_benchmark(network, SIM_COUNTS[0], n_games=N_GAMES, compile=True)
    # if "error" in st_compiled:
    #     print(f"  Compiled: failed: {st_compiled['error']}")
    # else:
    #     print(f"  Compiled: {st_compiled['sims_per_sec']:.0f} sims/sec  ({st_compiled['elapsed']:.1f}s)  "
    #           f"({st_compiled['sims_per_sec']/st_eager['sims_per_sec']:.2f}x)")
    # print(f"  (multi-worker server: ~2200 sims/sec)\n")

    # --- Local-model multi-worker benchmark ---
    n_workers = NUM_WORKERS
    n_games   = N_GAMES
    print(f"--- Local-model workers ({n_workers} workers, {n_games} games, {SIM_COUNTS[0]} sims) ---")
    for do_compile in (True,):
        r = run_local_model_benchmark(state_dict, SIM_COUNTS[0],
                                      num_workers=n_workers, n_games=n_games,
                                      do_compile=do_compile)
        label = "Compiled" if do_compile else "Eager   "
        print(f"  {label}: {r['sims_per_sec_wall']:.0f} sims/sec wall  "
              f"({r['sims_per_sec_cpu']:.0f} sims/sec cpu-total)  "
              f"wall={r['wall_time']:.1f}s")
    print()

    # --- Multi-worker queue benchmarks ---
    print(f"Workers: {NUM_WORKERS}  |  Games per run: {N_GAMES}\n")
    print(f"{'Sims':>6}  {'Avg moves':>10}  {'Avg time':>10}  {'Games/s':>9}  {'Sims/s':>10}")
    print("-" * 55)

    #results = []
    #for sims in SIM_COUNTS:
    #    print(f"  Running {sims} sims x {N_GAMES} games...", flush=True)
    #    r = run_benchmark(state_dict, sims)
    #    results.append(r)
    #    print(f"{r['sims']:>6}  {r['avg_moves']:>10.1f}  "
    #          f"{r['avg_time_s']:>9.1f}s  "
    #          f"{r['games_per_sec']:>9.3f}  "
    #          f"{r['sims_per_sec']:>10.0f}")

    # Linearity check: time should scale with sims if linear
    #print("\nScaling relative to 500 sims (linear = ratio matches sim ratio):")
    #base = results[0]
    #for r in results[1:]:
    #    sim_ratio  = r["sims"] / base["sims"]
    #    time_ratio = r["avg_time_s"] / base["avg_time_s"]
    #    print(f"  {base['sims']} -> {r['sims']}:  "
    #          f"sim ratio={sim_ratio:.2f}x  time ratio={time_ratio:.2f}x  "
    #          f"({'linear' if abs(time_ratio - sim_ratio) < 0.1 * sim_ratio else 'non-linear'})")

    ## Verdict
    #ratio = direct['calls_per_sec'] / results[0]['sims_per_sec']
    #print(f"\nDirect/multi-worker ratio: {ratio:.1f}x  -> "
    #      f"{'IPC/sleep is the ceiling' if ratio > 1.5 else 'GPU batching is contributing'}")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
