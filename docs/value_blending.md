# Value-target blending

## Context

The value head is trained to predict the game outcome from a position. In
pure AlphaZero, the target `z` is the terminal result (±1, or the material
ratio for truncated games). The idea behind blending is that MCTS's root
value estimate `q` — a visit-weighted average of its children's values —
is a *lower-variance* signal than the raw outcome. The outcome is unbiased
but noisy (one sample of a stochastic game). The MCTS Q is biased toward
the current network but averaged over many simulations. Blending trades
some bias for lower variance:

    value_target = α * terminal + (1 - α) * root_q

with α=1 meaning pure AZ and α<1 meaning "trust MCTS a bit."

## Why naive blending collapsed

We previously tried α=0.5 (constant, applied to every example). The value
head collapsed to predicting 0 on early-game positions. Mechanism:

1. In the opening, MCTS is doing broad exploration over near-balanced
   lines. The visit-weighted root Q averages toward ~0 (a near-draw
   position spreads visits across roughly equal branches).
2. With constant α=0.5, the target for *every* opening position is
   ≈ 0.5 * (± 1) + 0.5 * 0 = ±0.5, or effectively "near zero, sign noisy".
3. The network finds a shortcut: predict 0 for anything opening-shaped.
4. Once it predicts 0, its leaf evaluations feed back into MCTS → Q stays
   near 0 → the collapse is self-reinforcing.

The failure is not "blending is wrong"; it is "trusting Q when Q is
uninformative is wrong."

## Current fix: Option A + B (gated blend)

Implemented in `lib.train_workers._blended_value_target`. The per-example
effective α is computed so that the Q weight is suppressed in regimes
where Q is known to be unreliable.

**(A) Phase ramp.** Q weight is zero before `temp_moves` (the
`SELF_PLAY_TEMP_MOVES` / TEMP_THRESHOLD constant), then ramps linearly to
full over `BLEND_RAMP_LEN` moves. Rationale: before TEMP_THRESHOLD, MCTS
selects moves with temperature=1 (visit-proportional sampling), deliberately
adding exploration noise to the root distribution. After TEMP_THRESHOLD,
MCTS commits (argmax) and Q becomes an informative best-line estimate.

Binding `t_warmup = temp_moves` is *principled*, not arbitrary: the same
signal that tells MCTS "start playing seriously" tells us "Q is now worth
trusting."

**(B) Visit-concentration gate.** The concentration of the root visit
distribution `π` is computed as `1 - H(π) / log(|support|)`. A one-hot
distribution → 1.0; a uniform one → 0.0. This gates Q by MCTS's own
confidence: when the search was undecided, Q is noisy and we don't trust
it; when the search converged, we do.

The two gates combine multiplicatively. Both must fire for Q to reach
its maximum weight:

    q_weight     = (1 - blend_alpha) * phase * concentration
    alpha_eff    = 1 - q_weight
    value_target = alpha_eff * terminal + q_weight * root_q

Fast path: when `blend_alpha == 1.0` the function short-circuits to
pure terminal (no gating overhead, no behaviour change vs old code).

## Tuning knobs

- `VALUE_BLEND_ALPHA` in `scripts/train_mcts.py`. Start conservatively
  (0.7 or 0.8) — this means max 20-30% Q weight even when both gates
  fully fire. Move toward 0.5 if training is stable.
- `BLEND_RAMP_LEN` in `lib/train_workers.py` (default 10 moves). Longer
  ramp = smoother handoff.
- TEMP_THRESHOLD itself. `SELF_PLAY_TEMP_MOVES` in the training script.
  Changing it will move the blend ramp with it — they are coupled by
  construction.

## What to watch when turning blending on

- **Sign accuracy** in `sign_acc` loss log. If it drops relative to the
  α=1.0 baseline, the blend is hurting. Terminal-only is the safe fallback.
- **Value loss magnitude**. Should drop modestly (lower-variance target).
  If it drops *a lot* and predictions cluster at 0, the collapse has
  returned — increase `blend_alpha` back toward 1.0.
- **Eval win rate vs MM3 / MM4**. The win rate vs MM4 is the real test —
  blending is meant to improve deep value accuracy. MM3 should stay flat;
  MM4 should rise.

## How to disable / revert to pure AZ

Set `VALUE_BLEND_ALPHA = 1.0` in `scripts/train_mcts.py`. The blending
function short-circuits to the fast path (pure terminal), identical to
pre-change behaviour. No other changes needed.

## Upgrade path: Option D (dual-target loss) for later

A+B addresses the *data-pipeline* symptom of collapse (bad targets). A
stronger fix addresses the *objective* itself: instead of blending into a
single target, train the value head against two targets with separate
loss terms. The network cannot satisfy both objectives by predicting 0,
so the collapse mode is structurally impossible.

### Recipe for implementing D

1. **Stop blending at the data layer.** In `lib/train_workers.py`:
   - Keep `_blended_value_target` but add a flag `return_components=True`
     that returns `(terminal, root_q)` instead of a scalar target.
   - Update `_slot_result` and `self_play_game` to pass both targets
     through so the training example becomes `(obs, policy, z, q)` rather
     than `(obs, policy, v)`.

2. **Add a Q target field to the replay buffer.** In `lib/training.py`:
   - Extend the tuple unpacking at the train loop. Currently:
     `buffer_list[i] = (obs, policy, value)`. Change to
     `(obs, policy, terminal, root_q, move_idx)` so the loss can apply
     per-sample gating. `move_idx` lets you re-derive phase gating here
     rather than at data-generation time, which decouples training-time
     blending policy from game-generation history.

3. **Compute two losses.** In `train_network`:

   ```python
   q_gate = phase_gate(move_idx) * concentration_gate(policy)   # (B,)

   value_loss_z = F.mse_loss(pred_value, terminal_target)
   value_loss_q = F.mse_loss(pred_value * q_gate, root_q * q_gate)

   total_loss = policy_loss + LAMBDA_Z * value_loss_z + LAMBDA_Q * value_loss_q
   ```

   Start with `LAMBDA_Z = 1.0`, `LAMBDA_Q = 0.3`. The terminal target is
   always supervised at full weight (so collapse cannot satisfy the Z
   loss). The Q loss adds a bias-variance-reduced secondary signal, but
   only for examples where the gate fires.

4. **Why D is strictly better than A+B once tuned.** A+B reduces the
   *amplitude* of bad supervision; D removes the *possibility* of
   shortcut solutions. If LAMBDA_Q is too high and training misbehaves,
   you recover by reducing LAMBDA_Q. You cannot get a collapse-to-zero
   shortcut because predicting 0 fails the Z loss loudly.

5. **What to carry over.** The phase + concentration gate logic is
   correct and transfers. The only change is where it's applied (at
   loss time, not at data time) and how it combines with the target
   (as a per-sample mask on the Q loss, not as a blend weight on the Z
   target). The scalar blend `α` is gone; replaced by `LAMBDA_Z` and
   `LAMBDA_Q`.

6. **Architectural variant (optional).** Add a second value head so the
   network has `v_z` (terminal predictor) and `v_q` (Q predictor).
   Inference uses a weighted combination. Downside: more parameters;
   upside: the two predictors can specialise.

If A+B works, don't bother with D. If A+B still shows collapse symptoms
at any `blend_alpha < 1.0`, D is the escalation path.
