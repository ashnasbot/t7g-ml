"""Export a net2 checkpoint to a single-file ONNX for the browser webapp.

Wraps the net to the (policy_logits, value) pair the JS search driver needs,
exports via torch, then re-saves with weights embedded (no external .data
sidecar) so the browser fetches exactly one model file.

Usage:
    python scripts/export_onnx_web.py \
        export/models/run_net2b/promoted_iter0085.pt webapp/spa/models/net2.onnx
"""
import argparse
import pathlib
import sys
import tempfile

import numpy as np
import onnx
import torch

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from lib.net2 import build_from_state_dict  # noqa: E402
from lib.t7g import board_to_obs, new_board  # noqa: E402


class _Wrapper(torch.nn.Module):
    """(B,7,7,4) float obs -> (policy_logits[B,1225], value[B,1]).

    Matches the browser search loop: it softmaxes the logits itself and reads
    value = P(win) - P(loss) straight off the net.
    """

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, obs):
        policy_logits, value, _ = self.net(obs)
        return policy_logits, value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", type=pathlib.Path)
    ap.add_argument("out", type=pathlib.Path)
    args = ap.parse_args()

    blob = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state = blob["network"] if isinstance(blob, dict) and "network" in blob else blob
    net = build_from_state_dict(state).eval()
    wrap = _Wrapper(net).eval()
    n_params = sum(p.numel() for p in net.parameters())

    obs0 = torch.from_numpy(board_to_obs(new_board(), True)).unsqueeze(0)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td) / "m.onnx"
        torch.onnx.export(
            wrap, (obs0,), str(tmp),
            input_names=["obs"], output_names=["policy_logits", "value"],
            dynamic_axes={"obs": {0: "batch"},
                          "policy_logits": {0: "batch"},
                          "value": {0: "batch"}},
            opset_version=17, do_constant_folding=True,
        )
        # Collapse any external-data sidecar into a single self-contained file.
        model = onnx.load(str(tmp), load_external_data=True)
        onnx.save_model(model, str(args.out), save_as_external_data=False)

    # Parity check: onnxruntime CPU vs torch on a few boards.
    import onnxruntime as ort
    sess = ort.InferenceSession(str(args.out), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    from lib.t7g import action_masks, apply_move
    board, turn, max_pe, max_ve = new_board(), True, 0.0, 0.0
    for _ in range(6):
        obs = board_to_obs(board, turn).astype(np.float32)[None]
        with torch.no_grad():
            pl_t, v_t, _ = net(torch.from_numpy(obs))
        pl_o, v_o = sess.run(None, {"obs": obs})
        max_pe = max(max_pe, float(np.abs(pl_t.numpy() - pl_o).max()))
        max_ve = max(max_ve, float(np.abs(v_t.numpy() - v_o).max()))
        legal = np.flatnonzero(action_masks(board, turn))
        if len(legal) == 0:
            break
        board = apply_move(board, int(rng.choice(legal)), turn)
        turn = not turn

    size_mb = args.out.stat().st_size / 1e6
    print(f"exported {type(net).__name__} ({n_params} params) -> {args.out} "
          f"({size_mb:.2f} MB, single file)")
    print(f"onnxruntime parity: policy maxerr={max_pe:.2e}  value maxerr={max_ve:.2e}")
    print("OK" if (max_pe < 1e-3 and max_ve < 1e-4) else "CHECK PARITY")


if __name__ == "__main__":
    main()
