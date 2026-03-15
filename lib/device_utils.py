"""
Shared helpers for device detection and network instantiation.

Used by both training workers (train_mcts.py) and the inference server.
"""
import torch
import torch.nn as nn

from lib.dual_network import DualHeadNetwork


def get_device() -> torch.device:
    """Return the best available device: CUDA → DirectML → CPU."""
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
) -> tuple[nn.Module, DualHeadNetwork]:
    """
    Instantiate a DualHeadNetwork from *state_dict*, move it to *device*, and
    optionally wrap it with ``torch.compile``.

    Returns
    -------
    compiled_net : nn.Module
        The (possibly compiled) network used for inference.
    base_net : DualHeadNetwork
        The uncompiled network — required for in-place ``load_state_dict``
        updates that preserve CUDA-graph tensor addresses.
    """
    net = DualHeadNetwork(num_actions=num_actions)
    net.load_state_dict(state_dict)
    net.to(device)
    net.eval()
    base_net = net
    if compile_net:
        try:
            net = torch.compile(net, mode="reduce-overhead")  # type: ignore[assignment]
        except Exception:
            pass
    return net, base_net
