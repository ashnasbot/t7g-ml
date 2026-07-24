"""
Shared helpers for device detection and network instantiation.

Used by both training workers (train_mcts.py) and the inference server.
"""
import shutil
import subprocess

import torch
import torch.nn as nn


# Allow TF32 on Ampere+ Tensor Cores for conv/matmul. ~2-3x speedup on
# supported hardware at the cost of ~1e-4 numerical precision, which is
# well below what MCTS / AZ training are sensitive to. No-op on older
# GPUs and on non-CUDA backends.
torch.set_float32_matmul_precision("high")


def get_device() -> torch.device:
    """Return the best available device: CUDA -> DirectML -> CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    try:
        import torch_directml  # type: ignore[import-untyped]
        return torch_directml.device()
    except ImportError:
        return torch.device("cpu")


def load_compiled_network(
    state_dict: dict,
    device: torch.device,
    num_actions: int = 1225,
    compile_net: bool = True,
) -> tuple[nn.Module, nn.Module]:
    """
    Instantiate the right network class from *state_dict*, move it to
    *device*, and optionally wrap it with ``torch.compile``.

    Returns
    -------
    compiled_net : nn.Module
        The (possibly compiled) network used for inference.
    base_net : nn.Module
        The uncompiled network - required for in-place ``load_state_dict``
        updates that preserve CUDA-graph tensor addresses.
    """
    # Dispatch by checkpoint shape so legacy (tanh value), wdl/ownership and
    # t7g-net2 checkpoints all load - the Elo anchor pool keeps old nets
    # playable alongside new-architecture training nets.
    from lib.net2 import build_from_state_dict
    net = build_from_state_dict(state_dict, num_actions=num_actions)
    net.to(device)
    net.eval()
    # channels_last memory format: cuDNN + Tensor Cores operate in NHWC
    # natively. Model weights/buffers are restrided so forward passes skip
    # the implicit NCHW->NHWC conversion cuDNN would otherwise do. Pairs
    # with BF16 autocast; on CUDA only (no-op elsewhere).
    if device.type == "cuda":
        net = net.to(memory_format=torch.channels_last)
    base_net = net
    # Skip torch.compile on ROCm/gfx1151: inductor's Triton conv lowering is
    # a large regression on the Strix Halo iGPU, and eager fp16 already runs
    # faster than the compiled path there. On CUDA, compile stays on (2-3x).
    if compile_net and not torch.version.hip:
        try:
            net = torch.compile(net, mode="reduce-overhead")  # type: ignore[assignment]
        except Exception:
            pass
    return net, base_net


def get_gpu_stats() -> dict[str, float]:
    """
    Return {"util_pct": ..., "temp_c": ...} for the first GPU, via
    nvidia-smi (CUDA) or rocm-smi (ROCm). Empty dict if neither is
    available or the query fails - callers should treat this as
    best-effort telemetry, not a hard dependency.
    """
    if torch.version.hip is not None and shutil.which("rocm-smi"):
        try:
            out = subprocess.check_output(
                ["rocm-smi", "--showuse", "--showtemp", "--json"],
                stderr=subprocess.DEVNULL, timeout=5,
            ).decode()
            import json as _json
            data = _json.loads(out)
            card = next(iter(data.values()))
            return {
                "util_pct": float(card.get("GPU use (%)", card.get("GPU Utilization (%)", "nan"))),
                "temp_c": float(card.get("Temperature (Sensor edge) (C)", card.get("Temperature (Sensor junction) (C)", "nan"))),
            }
        except Exception:
            return {}
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=utilization.gpu,temperature.gpu",
                 "--format=csv,noheader,nounits"],
                stderr=subprocess.DEVNULL, timeout=5,
            ).decode().strip().splitlines()[0]
            util, temp = out.split(",")
            return {"util_pct": float(util), "temp_c": float(temp)}
        except Exception:
            return {}
    return {}
