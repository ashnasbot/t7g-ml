"""t7g-net3: Lc0-BT-style encoder-only transformer over the 49 squares.

A FRESH architecture, not a graft onto net2/net2c.  The trunk is gone: no convs,
no residual bottleneck blocks, no BatchNorm.  The board is 49 tokens and the
trunk is N pre-norm transformer encoder layers.  Everything net2c *measured* is
kept (see lib/net2c.py): no margin head, no soft-policy head, no short-term value
heads, value capacity deliberately not widened, phase output buckets on the value
and ownership readouts.

Why a transformer is a reasonable bet for 7x7 Ataxx specifically
---------------------------------------------------------------
  * 49 tokens.  Attention is O(T^2 * C) = 2401*C per head-group per layer, which
    at C=128 is *cheaper* than the two 3x3 convs it replaces (49 * 9 * 64 * 64).
    The quadratic term that makes transformers expensive in Lc0 (64 squares) and
    ruinous in LLMs is free at our size.
  * Receptive field.  A jump move reaches distance 2, but Ataxx value is a
    whole-board flipping/parity question: a corner cluster changes the value of
    the opposite corner.  net2 needs 6 blocks plus 2 hand-placed global-pool
    biases to move board-wide information around.  Every encoder layer here is
    globally connected by construction, and the gpool bias becomes unnecessary
    rather than architectural furniture.
  * The policy head is ALREADY an attention head.  net2's 1225 = 49x25 from-to
    readout is Lc0's attention policy: logit(src->dst) = S_src . T_dst / sqrt(D).
    In net2 it is a single attention layer bolted onto a conv trunk; here it is
    the natural final readout of a stack that speaks the same language.

Structure (Lc0 BT lineage, deviations marked)
--------------------------------------------
  embedding   per-square Linear(4 -> C), then MULTIPLY-AND-ADD GATING: learned
              (49, C) scale and bias.  This is Lc0's `ma_gating` and it is the
              positional encoding -- absolute, per-square, per-channel, which
              suits a fixed 7x7 board far better than a sinusoid.

  encoder x N pre-norm:  x = x + MHA(LN(x));  x = x + FFN(LN(x))
              FFN is Linear(C -> ffn_mult*C) -> ReLU -> Linear(back).
              DEVIATION: Lc0 BT is post-norm with DeepNorm alpha-scaled
              residuals.  Pre-norm trains without an LR warmup, and this
              codebase's loop runs a constant LR with no warmup
              (scripts/train_mcts.py), so pre-norm is the compatible choice.
              ReLU rather than Lc0's mish: measured no-op here, and relu is what
              the rest of the codebase is tuned for (project_activation_ab).

  attention   logits = QK^T/sqrt(dh) + relative-position bias + smolgen bias.

              RELATIVE-POSITION BIAS (rpe=True, default).  A learned per-head
              scalar for each of the 13x13 = 169 possible (dy, dx) square
              offsets, gathered into the 49x49 logit matrix.  This is T5-style
              relative bias, i.e. a cheap stand-in for Lc0 BT4's RPE, and it is
              the single most Ataxx-relevant inductive bias available: legality
              itself is a function of offset (|d| = 1 clone, |d| = 2 jump), and
              a plain transformer has to rediscover that from the gating params.
              Costs heads*169 parameters.

              SMOLGEN (smolgen=True, default).  Lc0's dynamic attention bias:
              compress every token to `smol_channels`, flatten all 49, and
              generate a full per-head 49x49 logit bias from that global
              summary.  It lets the position itself decide which squares should
              look at which -- "this is a corner fight, tie those cells
              together" -- instead of only the content-based QK term.  The final
              (smol_gen -> 2401) projection is SHARED across all layers, as in
              Lc0, so N layers cost one gen x 2401 matrix between them.
              SIZED DOWN HARD from Lc0 (channels 32/hidden 256/gen 256 -> 8/64/
              32).  At Lc0's sizes smolgen alone is 5.6M parameters, 5x this
              whole network; the defaults here put the trunk at 1.17M, matched to
              net2c's 1.08M so an A/B is a comparison of architectures and not of
              parameter budgets.

  policy      Q/K attention readout -> 49x49 -> gather to 1225, OOB slots pinned
              to POLICY_MASK_VALUE.  Identical contract to net2/net2c.
  value       per-token Linear(C -> value_channels) -> mean+max pool -> FC ->
              3*value_buckets.  Pooled, NOT flattened-49: net2c's finding that
              z carries only ~4000 independent labels is architecture-
              independent, so the value head must stay narrow here too.
  ownership   per-token Linear(C -> 3*own_buckets).  In net2c this needed its
              own 3x3 conv branch to give each cell local context; a token in
              layer N already has whole-board context, so a linear readout is
              the right shape.

forward() keeps the (policy_logits, value, margin=None) 3-tuple contract, and
forward_full() returns the same dict keys as net2c with None for the heads that
do not exist.  build_from_state_dict dispatches to it by checkpoint shape, so
search, the eval pool and the ONNX export need no changes.

UNMEASURED.  Nothing about the value ceiling changes here: the target noise floor
(project_target_noise_floor) is a property of the data, not the net.  The claim
under test is only that this trunk reaches the same or better fit per parameter
and per second than the conv trunk, with the policy head the most likely place to
see it.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.t7g import board_to_obs
from lib.training import _ACT_SRC, _ACT_DST, _ACT_INB, POLICY_MASK_VALUE
from lib.net2c import _OCC_BOUNDS, Net2C

_pick = Net2C._pick  # phase-bucket gather; identical semantics, see lib/net2c.py


def _offset_index() -> np.ndarray:
    """(49, 49) index into a 13x13 table of (dy, dx) offsets, dy/dx in [-6, 6]."""
    ys, xs = np.divmod(np.arange(49), 7)
    dy = ys[:, None] - ys[None, :] + 6
    dx = xs[:, None] - xs[None, :] + 6
    return (dy * 13 + dx).astype(np.int64)


class Smolgen(nn.Module):
    """Lc0 smolgen: a per-head 49x49 attention-logit bias generated from a
    compressed summary of ALL tokens.

    The expensive final projection (smol_gen -> 49*49) is not owned here -- it is
    passed in, shared by every layer, which is what makes the module affordable.
    """

    def __init__(self, dim: int, heads: int, channels: int = 8,
                 hidden: int = 64, gen: int = 32) -> None:
        super().__init__()
        self.heads, self.gen = heads, gen
        self.compress = nn.Linear(dim, channels, bias=False)
        self.dense1 = nn.Linear(49 * channels, hidden)
        self.ln1 = nn.LayerNorm(hidden)
        self.dense2 = nn.Linear(hidden, heads * gen)
        self.ln2 = nn.LayerNorm(heads * gen)

    def forward(self, x: torch.Tensor, shared: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        h = self.compress(x).reshape(b, -1)
        h = self.ln1(F.relu(self.dense1(h)))
        h = self.ln2(F.relu(self.dense2(h)))
        h = h.reshape(b * self.heads, self.gen) @ shared
        return h.reshape(b, self.heads, 49, 49)


class EncoderLayer(nn.Module):
    """Pre-norm encoder layer: MHA(+rpe/+smolgen bias) then FFN.

    Attention is written out rather than using nn.MultiheadAttention or SDPA so
    the additive 49x49 biases go in as plain tensors -- and so the whole thing
    stays a handful of matmuls for scripts/export_onnx_web.py (opset 17).
    """

    def __init__(self, dim: int, heads: int, ffn_mult: int = 2,
                 rpe: bool = True, smolgen: 'Smolgen | None' = None) -> None:
        super().__init__()
        assert dim % heads == 0, f"dim {dim} not divisible by heads {heads}"
        self.heads, self.dh = heads, dim // heads
        self.ln1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim)
        self.ffn1 = nn.Linear(dim, ffn_mult * dim)
        self.ffn2 = nn.Linear(ffn_mult * dim, dim)
        self.rpe = nn.Parameter(torch.zeros(heads, 169)) if rpe else None
        self.smolgen = smolgen
        # Zero-init the residual exits: at init the layer is the identity, which
        # keeps activation variance flat through an arbitrarily deep stack (the
        # same motive as net2's fixed-variance scaling and the zero-init gpool
        # FC, spelled the way transformers spell it).
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        nn.init.zeros_(self.ffn2.weight)
        nn.init.zeros_(self.ffn2.bias)

    def forward(self, x: torch.Tensor, off_idx: torch.Tensor,
                shared: 'torch.Tensor | None') -> torch.Tensor:
        b = x.shape[0]
        q, k, v = self.qkv(self.ln1(x)).chunk(3, dim=-1)
        q = q.reshape(b, 49, self.heads, self.dh).transpose(1, 2)
        k = k.reshape(b, 49, self.heads, self.dh).transpose(1, 2)
        v = v.reshape(b, 49, self.heads, self.dh).transpose(1, 2)
        logits = q @ k.transpose(-2, -1) / (self.dh ** 0.5)
        if self.rpe is not None:
            logits = logits + self.rpe[:, off_idx].unsqueeze(0)
        if self.smolgen is not None and shared is not None:
            logits = logits + self.smolgen(x, shared)
        att = torch.softmax(logits, dim=-1) @ v
        att = att.transpose(1, 2).reshape(b, 49, -1)
        x = x + self.proj(att)
        return x + self.ffn2(F.relu(self.ffn1(self.ln2(x))))


class Net3(nn.Module):
    """t7g-net3.  See module docstring.

    Args:
        num_actions:    must be 1225 (the policy head is the 49x25 from-to map).
        dim:            token/embedding width C.
        num_layers:     encoder layer count.
        heads:          attention heads per layer (dim must divide by it).
        ffn_mult:       FFN hidden = ffn_mult * dim.
        rpe:            learned per-head relative-position bias over the 169
                        (dy, dx) offsets.
        smolgen:        Lc0 dynamic attention bias; the shared output
                        projection lives on the model.
        smol_channels/smol_hidden/smol_gen: smolgen internals.
        att_dim:        policy Q/K width.
        value_channels: per-token value projection width (then mean+max pooled).
        value_hidden:   value FC width.
        value_buckets/own_buckets: phase output buckets (lib/net2c.py).
    """

    def __init__(self, num_actions: int = 1225, dim: int = 128,
                 num_layers: int = 6, heads: int = 8, ffn_mult: int = 2,
                 rpe: bool = True, smolgen: bool = True,
                 smol_channels: int = 8, smol_hidden: int = 64,
                 smol_gen: int = 32, att_dim: int = 16,
                 value_channels: int = 32, value_hidden: int = 96,
                 value_buckets: int = 4, own_buckets: int = 8) -> None:
        super().__init__()
        assert num_actions == 1225, "Net3 policy head is specific to the 49x25 action space"
        for nm, k in (("value_buckets", value_buckets), ("own_buckets", own_buckets)):
            if k not in _OCC_BOUNDS:
                raise ValueError(f"{nm}={k} has no boundary table; "
                                 f"known: {sorted(_OCC_BOUNDS)}")
        self.dim = dim
        self.att_dim = att_dim
        self.value_buckets = value_buckets
        self.own_buckets = own_buckets

        self.embed = nn.Linear(4, dim)
        # ma_gating: the positional encoding.  scale at 1 / bias at 0 so a fresh
        # net starts as pure content embedding and learns the geometry it needs.
        self.gate_mul = nn.Parameter(torch.ones(49, dim))
        self.gate_add = nn.Parameter(torch.zeros(49, dim))

        if smolgen:
            self.smol_shared = nn.Parameter(torch.empty(smol_gen, 49 * 49))
            nn.init.normal_(self.smol_shared, std=1.0 / smol_gen ** 0.5)
        else:
            self.smol_shared = None
        self.layers = nn.ModuleList([
            EncoderLayer(dim, heads, ffn_mult=ffn_mult, rpe=rpe,
                         smolgen=Smolgen(dim, heads, smol_channels, smol_hidden,
                                         smol_gen) if smolgen else None)
            for _ in range(num_layers)
        ])
        self.ln_out = nn.LayerNorm(dim)

        self.policy_q = nn.Linear(dim, att_dim)
        self.policy_k = nn.Linear(dim, att_dim)
        pair_idx = (_ACT_SRC * 49 + np.where(_ACT_INB, _ACT_DST, 0)).astype(np.int64)
        self.register_buffer("_pair_idx", torch.from_numpy(pair_idx), persistent=False)
        self.register_buffer("_oob", torch.from_numpy(~_ACT_INB), persistent=False)
        self.register_buffer("_off_idx", torch.from_numpy(_offset_index()),
                             persistent=False)

        self.value_proj = nn.Linear(dim, value_channels)
        self.value_fc1 = nn.Linear(2 * value_channels, value_hidden)
        self.value_wdl = nn.Linear(value_hidden, 3 * value_buckets)
        self.own_fc = nn.Linear(dim, 3 * own_buckets)

        for nm, k in (("_v_bounds", value_buckets), ("_o_bounds", own_buckets)):
            self.register_buffer(nm, torch.tensor(_OCC_BOUNDS[k], dtype=torch.float32),
                                 persistent=False)

    @staticmethod
    def is_net3_state_dict(state_dict: dict) -> bool:
        return "gate_mul" in state_dict

    @staticmethod
    def infer_arch(state_dict: dict) -> dict:
        """Constructor kwargs from checkpoint shapes."""
        dim = state_dict["gate_mul"].shape[1]
        num_layers = 1 + max(int(k.split(".")[1]) for k in state_dict
                             if k.startswith("layers."))
        rpe = "layers.0.rpe" in state_dict
        smolgen = "smol_shared" in state_dict
        # Head count leaves a shape footprint only in the rpe table or in
        # smolgen's per-head output; with neither enabled it is unrecoverable
        # (nothing in a plain MHA depends on how dim is split) and defaults to 8.
        if rpe:
            heads = state_dict["layers.0.rpe"].shape[0]
        elif smolgen:
            heads = (state_dict["layers.0.smolgen.dense2.weight"].shape[0]
                     // state_dict["smol_shared"].shape[0])
        else:
            heads = 8
        kw = {
            "dim": dim,
            "num_layers": num_layers,
            "heads": heads,
            "ffn_mult": state_dict["layers.0.ffn1.weight"].shape[0] // dim,
            "rpe": rpe,
            "smolgen": smolgen,
            "att_dim": state_dict["policy_q.weight"].shape[0],
            "value_channels": state_dict["value_proj.weight"].shape[0],
            "value_hidden": state_dict["value_fc1.weight"].shape[0],
            "value_buckets": state_dict["value_wdl.weight"].shape[0] // 3,
            "own_buckets": state_dict["own_fc.weight"].shape[0] // 3,
        }
        if smolgen:
            kw["smol_channels"] = state_dict["layers.0.smolgen.compress.weight"].shape[0]
            kw["smol_hidden"] = state_dict["layers.0.smolgen.dense1.weight"].shape[0]
            kw["smol_gen"] = state_dict["smol_shared"].shape[0]
        return kw

    def _tokens(self, obs: torch.Tensor) -> 'tuple[torch.Tensor, torch.Tensor]':
        """(B,7,7,4) or (B,4,7,7) -> (B, 49, 4) tokens plus board occupancy."""
        if obs.dim() == 4 and obs.shape[-1] == 4:
            planes = obs.float()                              # (B,7,7,4)
        else:
            planes = obs.float().permute(0, 2, 3, 1)          # (B,4,7,7) -> (B,7,7,4)
        tok = planes.reshape(-1, 49, 4)
        occ = tok[:, :, 0].sum(1) + tok[:, :, 1].sum(1)
        return tok, occ

    def forward(self, obs: torch.Tensor, full: bool = False):
        """(policy_logits, value, margin=None); full=True appends value_logits,
        ownership_logits, and None for the heads net3 does not have."""
        tok, occ = self._tokens(obs)
        v_bucket = torch.bucketize(occ, self._v_bounds)
        o_bucket = torch.bucketize(occ, self._o_bounds)

        x = self.embed(tok) * self.gate_mul + self.gate_add
        for layer in self.layers:
            x = layer(x, self._off_idx, self.smol_shared)
        x = self.ln_out(x)

        b = x.shape[0]
        q = self.policy_q(x)
        k = self.policy_k(x)
        allpairs = (q @ k.transpose(1, 2)) / (self.att_dim ** 0.5)
        policy_logits = allpairs.reshape(b, 49 * 49)[:, self._pair_idx]
        policy_logits = policy_logits.masked_fill(self._oob, POLICY_MASK_VALUE)

        v_sp = F.relu(self.value_proj(x))                     # (B, 49, vc)
        v = torch.cat([v_sp.mean(dim=1), v_sp.amax(dim=1)], dim=1)
        v = F.relu(self.value_fc1(v))
        value_logits = _pick(self.value_wdl(v), v_bucket, self.value_buckets)
        probs = F.softmax(value_logits, dim=-1)
        value = (probs[:, 0] - probs[:, 2]).unsqueeze(-1)     # P(win) - P(loss)

        if full:
            # (B, 49, 3k) -> (B, 3k, 7, 7): the ownership loss is a per-cell CE
            # against a (B,7,7) label map, so the spatial layout has to match
            # net2c's conv readout exactly.
            own = self.own_fc(x).transpose(1, 2).reshape(b, -1, 7, 7)
            ownership_logits = _pick(own, o_bucket, self.own_buckets)
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
