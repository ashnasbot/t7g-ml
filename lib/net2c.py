"""t7g-net2c: net2 with the auxiliary heads re-proportioned to measured signal.

net2 gave every auxiliary target a head sized by intuition.  Two 2026-07-24
measurements say that allocation is wrong (see debug/aux_noise_floor.py and
debug/target_ess.py):

  * MARGIN carries no learnable signal beyond z at all -- regress the final
    material margin on the game outcome and nothing survives.  It was trained at
    MARGIN_COEF 0.4, the second-heaviest term in the loss, to restate z.
    net2c DELETES the margin head.

  * OWNERSHIP carries ~4x z's learnable signal, ~75% of it independent of the
    outcome, spread over ~13 effective spatial directions.  In net2 it was the
    THINNEST head in the network -- a 3x32x1x1 conv, 96 parameters -- and it hung
    off `v_sp`, the value branch's 1x1 conv, whose entire purpose is to feed a
    global-pooled scalar.  A dense 49-cell spatial target read out of a feature
    map built to discard spatial detail is an architectural mismatch.
    net2c gives ownership its OWN branch off the trunk, with a 3x3 conv so each
    cell sees local context, at its own width.

  * The aux SOFT POLICY head was ablated to SOFT_POLICY_COEF 0.0 and left in the
    module.  net2c deletes it; revive from history if it is ever wanted.

  * The SHORT-TERM VALUE heads are gone too.  They existed to sharpen late-game
    value, but the by-phase noise table says the late game was never the problem:
    irreducible var(z) is 0.279 at ply 60-80 and 0.000 at ply 80+, so z already
    teaches the endgame exactly.  The opening is the unlearnable part (0.917),
    which the lambda-return MAIN target addresses directly.  And the heads were
    near-redundant regardless: lambda-returns across horizons 2..200 have a
    participation ratio of 1.26 -- one direction explains 88% of them -- while
    st_value_fc read the SAME 96-dim vector as value_wdl.  Three linear readouts
    of identical features predicting a ~0.95-correlated target.

Net2 keeps its st head, so ST_HORIZONS/ST_LAMBDAS stay in lib/training.py and
self-play still computes st_targets into buffer slot 7 -- historical net2
checkpoints must keep loading (eval DB, ckpt_ladder, the browser export).  For a
net2c run the slot is simply unused; set ST_VALUE_COEF = 0.0.

Value capacity is deliberately NOT increased.  The value target carries only
~4000 independent labels in a 392k-row dataset (one per game, ICC 1.000), and
offline arms overfit it within 3 epochs -- holdout value CE rises while train
value CE falls.  Width belongs with the policy target, whose labels are
per-position and effectively unclustered (ICC ~0.005, n_eff ~60-70x the value
head's).

PHASE OUTPUT BUCKETS (2026-07-25).  The value and ownership readouts are
selected by board occupancy, NNUE-style: K final layers, one chosen by a
discrete non-learned router.  Measured by debug/bucket_probe.py on
convergence_4k.npz + run_net2b/promoted_iter0180.pt, held-out R2 over a single
global readout:

    z          +0.00245 at K=8 (peak), +0.00184 at K=4   base 0.17118
    ownership  +0.00418 at K=12,       +0.00369 at K=4   base 0.07837
    random-bucket control: NEGATIVE at every K, both targets

Three facts from that probe drive the defaults here:

  * A per-bucket INTERCEPT alone gains ~0 (+0.00000..+0.00007 everywhere).  The
    mapping itself differs by phase; phase is not just an output shift.  So a
    phase INPUT plane buys nothing -- do not "fix" the all-zero clock plane in
    board_to_obs expecting value gains.  Buckets are the fix.
  * z's gain PEAKS AT K=8 AND DECLINES BY K=12, exactly the ~4000-independent-
    label ceiling from debug/target_ess.py.  value_buckets defaults to 4, which
    takes 75% of the available gain well clear of that turn.  Do not raise it
    without re-measuring on a fresh buffer.
  * Ownership is still climbing at K=12 (per-cell labels, far higher n_eff), so
    own_buckets defaults to 8.

The router is computed INSIDE forward() from the observation itself, so nothing
outside this file changes: no buffer format, no C-side work, no caller changes,
and value_buckets=own_buckets=1 reproduces the un-bucketed net exactly (which is
also what infer_arch reads back for pre-bucketing checkpoints).

CAVEAT ON SIZING: the probe was a linear readout of FROZEN features.  A net
trained with buckets learns different features, so those numbers say "there is
phase structure the current readout cannot express", NOT a predicted Elo gain.

forward() keeps the (policy_logits, value, margin) 3-tuple contract, with margin
returned as None; callers that read it must handle that (lib/training.py and the
TB histogram in scripts/train_mcts.py do).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.t7g import board_to_obs
from lib.training import _ACT_SRC, _ACT_DST, _ACT_INB, POLICY_MASK_VALUE
from lib.net2 import NestedBottleneckBlock


# Occupancy cut points per bucket count, as equal-population quantiles of board
# fill over models/convergence_4k.npz (392k rows, occupancy 4..48).  Equal
# population is what debug/bucket_probe.py measured, so these reproduce its
# buckets exactly; the coarser sets are strict subsets of the finer ones.
_OCC_BOUNDS: 'dict[int, list[int]]' = {
    1: [],
    2: [29],
    3: [22, 36],
    4: [18, 29, 40],
    6: [15, 22, 29, 36, 43],
    8: [13, 18, 24, 29, 34, 40, 45],
}


class Net2C(nn.Module):
    """t7g-net2c.  See module docstring.

    Args:
        num_actions:    must be 1225 (the attention head is built on the
                        49x25 from-to structure).
        channels:       trunk width C (bottleneck runs at C/2).
        num_blocks:     nested-bottleneck block count.
        gpool_blocks:   indices of blocks that get the gpool bias; None =
                        (num_blocks//4, num_blocks//2).
        att_dim:        source/target vector dimension of the policy head.
        value_channels: value branch 1x1 conv width (feeds the pooled scalar).
        value_hidden:   value branch FC width.
        own_channels:   ownership branch width.  Its own branch off the trunk,
                        not shared with the value path.
        value_buckets:  phase output buckets on the WDL readout.  1 = one global
                        readout (pre-2026-07-25 behaviour).  See the module
                        docstring for why the default is 4 and not 8.
        own_buckets:    phase output buckets on the ownership readout.
    """

    def __init__(self, num_actions: int = 1225, channels: int = 128,
                 num_blocks: int = 6, gpool_blocks: 'tuple | None' = None,
                 att_dim: int = 16, value_channels: int = 32,
                 value_hidden: int = 96, own_channels: int = 48,
                 value_buckets: int = 4, own_buckets: int = 8) -> None:
        super().__init__()
        if gpool_blocks is None:
            gpool_blocks = (num_blocks // 4, num_blocks // 2)
        assert num_actions == 1225, "Net2C policy head is specific to the 49x25 action space"
        for nm, k in (("value_buckets", value_buckets), ("own_buckets", own_buckets)):
            if k not in _OCC_BOUNDS:
                raise ValueError(f"{nm}={k} has no boundary table; "
                                 f"known: {sorted(_OCC_BOUNDS)}")
        self.channels = channels
        self.att_dim = att_dim
        self.value_buckets = value_buckets
        self.own_buckets = own_buckets

        self.input_conv = nn.Conv2d(4, channels, kernel_size=3, padding=1, bias=False)
        nn.init.kaiming_normal_(self.input_conv.weight, nonlinearity='relu')
        self.blocks = nn.Sequential(
            *[NestedBottleneckBlock(channels, gpool=(i in gpool_blocks))
              for i in range(num_blocks)]
        )
        self.trunk_bn = nn.BatchNorm2d(channels)

        # Attention policy: per-square source/target vectors.  No soft head.
        self.policy_src = nn.Conv2d(channels, att_dim, kernel_size=1)
        self.policy_dst = nn.Conv2d(channels, att_dim, kernel_size=1)

        pair_idx = (_ACT_SRC * 49 + np.where(_ACT_INB, _ACT_DST, 0)).astype(np.int64)
        self.register_buffer("_pair_idx", torch.from_numpy(pair_idx), persistent=False)
        self.register_buffer("_oob", torch.from_numpy(~_ACT_INB), persistent=False)

        # Value branch: unchanged from net2, deliberately not widened.
        self.value_conv = nn.Conv2d(channels, value_channels, kernel_size=1)
        self.value_fc1 = nn.Linear(2 * value_channels, value_hidden)
        self.value_wdl = nn.Linear(value_hidden, 3 * value_buckets)

        # Ownership branch: its own read of the trunk, 3x3 so each cell sees
        # local context rather than a per-cell projection of pooled features.
        self.own_conv1 = nn.Conv2d(channels, own_channels, kernel_size=3, padding=1)
        self.own_conv2 = nn.Conv2d(own_channels, 3 * own_buckets, kernel_size=1)

        # Router cut points.  Non-persistent: they are a function of the bucket
        # count, which infer_arch recovers from the readout shapes, so they must
        # not become checkpoint state that could drift out of sync with it.
        for nm, k in (("_v_bounds", value_buckets), ("_o_bounds", own_buckets)):
            self.register_buffer(nm, torch.tensor(_OCC_BOUNDS[k], dtype=torch.float32),
                                 persistent=False)

    @staticmethod
    def is_net2c_state_dict(state_dict: dict) -> bool:
        return any(k.startswith("own_conv1.") for k in state_dict)

    @staticmethod
    def infer_arch(state_dict: dict) -> dict:
        """Constructor kwargs from checkpoint shapes."""
        num_blocks = 1 + max(int(k.split(".")[1]) for k in state_dict
                             if k.startswith("blocks."))
        gpool_blocks = tuple(sorted({int(k.split(".")[1]) for k in state_dict
                                     if ".gpool_fc." in k}))
        return {
            "channels": state_dict["input_conv.weight"].shape[0],
            "num_blocks": num_blocks,
            "gpool_blocks": gpool_blocks,
            "att_dim": state_dict["policy_src.weight"].shape[0],
            "value_channels": state_dict["value_conv.weight"].shape[0],
            "value_hidden": state_dict["value_fc1.weight"].shape[0],
            "own_channels": state_dict["own_conv1.weight"].shape[0],
            # Pre-bucketing checkpoints have 3 outputs -> 1 bucket, which
            # reconstructs the un-bucketed net exactly.
            "value_buckets": state_dict["value_wdl.weight"].shape[0] // 3,
            "own_buckets": state_dict["own_conv2.weight"].shape[0] // 3,
        }

    @staticmethod
    def _pick(logits: torch.Tensor, bucket: torch.Tensor, k: int) -> torch.Tensor:
        """Select each row's bucket from a (B, k*3, ...) readout -> (B, 3, ...).

        Every bucket is evaluated and one is gathered, rather than indexing the
        weight matrix per row.  These readouts are tiny (96x3 and 48x3x49), so
        k times their FLOPs is negligible against the ~1M-param trunk, and this
        form stays a single batched op -- and exports to ONNX cleanly, which
        per-row weight indexing does not (scripts/export_onnx_web.py, opset 17).
        """
        if k == 1:
            return logits
        b = logits.shape[0]
        tail = logits.shape[2:]
        sel = logits.reshape(b, k, 3, *tail)
        idx = bucket.reshape(b, 1, 1, *([1] * len(tail))).expand(b, 1, 3, *tail)
        return sel.gather(1, idx).squeeze(1)

    def _policy_logits(self, x: torch.Tensor) -> torch.Tensor:
        b = x.size(0)
        s = self.policy_src(x).reshape(b, self.att_dim, 49)
        t = self.policy_dst(x).reshape(b, self.att_dim, 49)
        allpairs = torch.einsum("bds,bdt->bst", s, t) / (self.att_dim ** 0.5)
        logits = allpairs.reshape(b, 49 * 49)[:, self._pair_idx]
        return logits.masked_fill(self._oob, POLICY_MASK_VALUE)

    def forward(self, obs: torch.Tensor, full: bool = False):
        """(policy_logits, value, margin=None); full=True appends value_logits,
        ownership_logits, and None for the heads net2c does not have."""
        if obs.dim() == 4 and obs.shape[-1] == 4:
            x = obs.permute(0, 3, 1, 2).contiguous(
                memory_format=torch.channels_last
            ).float()
        else:
            x = obs.float()

        # Phase router, read straight off the observation: after the permute
        # above, channel 0 is the opponent's stones and channel 1 is ours, so
        # their sum over the board is occupancy.  Computing it here is what
        # keeps bucketing invisible to every caller.
        occ = (x[:, 0] + x[:, 1]).sum(dim=(1, 2))
        v_bucket = torch.bucketize(occ, self._v_bounds)
        o_bucket = torch.bucketize(occ, self._o_bounds)

        x = F.relu(self.input_conv(x))
        x = self.blocks(x)
        x = self.trunk_bn(x)

        policy_logits = self._policy_logits(x)

        v_sp = F.relu(self.value_conv(x))
        v = torch.cat([v_sp.mean(dim=(2, 3)), v_sp.amax(dim=(2, 3))], dim=1)
        v = F.relu(self.value_fc1(v))
        value_logits = self._pick(self.value_wdl(v), v_bucket, self.value_buckets)
        probs = F.softmax(value_logits, dim=-1)
        value = (probs[:, 0] - probs[:, 2]).unsqueeze(-1)  # P(win) - P(loss)

        if full:
            ownership_logits = self._pick(
                self.own_conv2(F.relu(self.own_conv1(x))), o_bucket, self.own_buckets)
            return (policy_logits, value, None, value_logits,
                    ownership_logits, None, None)
        return policy_logits, value, None

    def forward_full(self, obs: torch.Tensor) -> dict:
        (policy_logits, value, margin, value_logits,
         ownership_logits, soft_policy_logits, st_values) = self.forward(obs, full=True)
        return {
            "policy_logits": policy_logits,
            "value": value,
            "margin": margin,
            "value_logits": value_logits,
            "ownership_logits": ownership_logits,
            "soft_policy_logits": soft_policy_logits,
            "st_values": st_values,
        }

    @torch.no_grad()
    def predict(self, board: np.ndarray, turn: bool) -> tuple[np.ndarray, float]:
        """Single-state inference for MCTS — same contract as DualHeadNetwork."""
        self.eval()
        obs = board_to_obs(board, turn)
        obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(next(self.parameters()).device)
        policy_logits, value, _ = self.forward(obs_tensor)
        policy_probs = F.softmax(policy_logits, dim=-1).cpu().numpy()[0]
        return policy_probs, value.cpu().item()

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path: str) -> None:
        self.load_state_dict(torch.load(path, weights_only=True))
