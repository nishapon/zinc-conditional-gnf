# Stage 26 — Bond-aware pairwise coupling conditioner (experiment C)

## What the conditioner already had, and what is actually new

`MaskedSelfAttention` (`zinc_gnf/flow.py:270`) already computes a learned pair score:

```
logits_ij  = (W_q p_i) . (W_k p_j) / sqrt(head_dim),   p = input_projection(source)
attended_i = sum_j softmax_j(logits_ij) * (W_v p_j)
```

So "a learned pairwise score fed into the conditioner" is **not new**, and a change that
only added another content-based pair score would largely duplicate it. Three properties
of the new pathway lie outside what that attention can express.

**1. Unnormalised aggregation carries degree; softmax cannot.** `softmax` over `j` makes
`attended_i` a convex combination — the weights sum to 1 however many neighbours are
selected. An attention head can say *which* nodes matter and in what proportion, never
*how many*. The new gate is a per-pair independent sigmoid, so

```
degree_i  = sum_j b_ij,        b_ij = sigmoid(l_ij) in (0, 1)
message_i = sum_j b_ij v_j / (1 + degree_i)
```

and `degree_i` is a soft neighbour **count**, passed to the conditioner as its own
feature. Valence is degree together with bond order, and valence is what decides whether a
decoded molecule parses. This is the one distinction the experiment is betting on, and it
is tested directly (`test_the_soft_degree_counts_neighbours_rather_than_normalising`).

**2. Exactly symmetric.** `l_ij == l_ji` holds by construction; `Q K^T` does not. A bond is
a symmetric relation.

**3. Supervisable.** The score has a ground-truth target. The attention logits never do.

Secondary: `l_ij` contains `w . |h_i − h_j|`, which is not a bilinear form and so is not
representable by any single dot-product head.

## Why the flow mathematics survive

The scorer reads **only** the coupling's pass-through half, the `(N, QED)` condition, and
the node mask — all identical in the forward and inverse directions. The true bond matrix
is never an input; it is only a target for an auxiliary loss. So `scale` and `translation`
remain functions of the pass-through half alone, the map stays a bijection in `x`, and
`log|det J| = Σ scale` over real nodes and channels is unchanged.

This is the distinction that made the design choice. Conditioning the coupling **on**
bonds, as `graph_flow.AtomSubnet` does, would make the model `p(embeddings | N, QED, A)`
and require a bond matrix at sampling time, which `NodeFlow.sample` has no way to supply.

Verified by: exact inversion with and without the node-autoregressive coupling; the
analytic log-determinant against `torch.autograd.functional.jacobian` in float64 (`abs=1e-7`),
with an assertion that the log-det is nonzero so the check cannot pass vacuously; padding
invariance including garbage in the padded columns; and node-permutation equivariance of
both the score matrix and the aggregated features.

## Class imbalance

A molecule with N heavy atoms and R rings has `N−1+R` bonds among `N(N−1)/2` pairs. At
three ring systems:

| N | pairs | bonds | positive rate | all-negative accuracy | pos_weight |
|---|---|---|---|---|---|
| 15 | 105 | 16 | 0.152 | 0.848 | 5.6 |
| 23 | 253 | 25 | 0.099 | 0.901 | 9.1 |
| 30 | 435 | 32 | 0.074 | 0.926 | 12.6 |

The imbalance is worst exactly where validity is worst. Two measures: the loss carries
`pos_weight = negatives/positives` measured per batch and capped at 50; and the logs report
positive-class precision, recall and F1 next to `bond_trivial_accuracy`, the all-negative
predictor's accuracy on the same batch, so `bond_accuracy` can never be read alone. The
trivial predictor has F1 = 0 and `bond_predicted_degree` = 0 by construction, and
`test_predicting_no_bond_everywhere_scores_zero_f1_and_a_worse_loss` asserts all of it.

## Initialisation, and one honest consequence

`pair_to_hidden` is zero-initialised, so the augmented flow reproduces the parent
checkpoint's density **bit for bit** (`torch.equal` on the log-probabilities, not
`allclose`). The parent's own weights all load unchanged; only the new keys are absent.

The consequence, which the logs have to expose: with that bridge at zero, the likelihood
gradient reaching the scorer is zero at step 0, because it carries a factor of the bridge
weight. The bridge's own gradient is nonzero, so it leaves zero on the first step and the
scorer trains from step 1. The bond supervision, by contrast, reaches the scorer at step 0
directly. Hence `bond_bridge_norm` in every log line: **if it stays at 0, arm `pairwise`
never engaged its pathway and its result says nothing about the architecture.**

## Three matched arms

| arm | architecture | bond loss | isolates |
|---|---|---|---|
| `control` | parent, unchanged | — | continued training |
| `pairwise` | pathway on | weight 0 (measured, not optimised) | the architecture |
| `pairwise_supervised` | pathway on, identical | weight 0.1 | the training signal |

`pairwise` and `pairwise_supervised` are architecturally identical — same parameters, same
initialisation, same seed — so `pairwise vs control` is the architecture and
`pairwise_supervised vs pairwise` is the supervision. `BondArmConfig.validate` refuses a
`pairwise` arm with a nonzero weight and a `pairwise_supervised` arm with a zero weight,
because either would silently destroy that separation.

Same parent checkpoint, 3,000 steps, lr 1e-5, warmup 100, batch 128, seed 3, same data
directory, same frozen decoder. Model selection stays on the **validation likelihood** in
all three arms, so the auxiliary objective cannot buy a better checkpoint by changing what
"better" means.

## Measured cost (ae128 parent: 12 blocks, 2 couplings each, half = 64, hidden = 256)

| quantity | value |
|---|---|
| conditioners | 24 |
| new parameters | 144,840 (1,427 scorer + 4,608 bridge each) |
| extra activation memory, batch 128, N=38, pair_dim 16 | ≈ 355 MB |
| …at pair_dim 8 | ≈ 213 MB |
| …at batch 64 | ≈ 177 MB |
| new FLOPs per coupling vs existing attention | ratio ≈ 0.50 |

The dominant term is `|h_i − h_j|` of shape `[B, N, N, pair_dim]`, which autograd keeps for
the backward pass. N=38 is the ZINC maximum; a real batch pads to its widest molecule, so
355 MB is a ceiling. On FLOPs the pathway is about half the existing attention, so expect
roughly 20–50% more wall-clock per step — **measure it with `MODE=smoke` rather than
trusting this estimate.** `--inspect-only` reports all of these from the built flow, plus
the bond positive rate on one real batch, without training.

## How success is judged

On generated molecules, against the matched control's **native QED MAE 0.1922, raw
validity 13.8%, valid target-hit yield 30.4%**:

- native (single-draw) QED MAE;
- valid target-hit yield over **all** attempts, not over valid molecules only;
- raw validity by N at 15 / 23 / 30, with corrected validity reported alongside;
- ring-size distributions against the matched reference, uniqueness, diversity;
- paired sample records and fixed-example grids, with **fresh draws for confirmation**;
- best-of-10 kept separate from native throughout.

Bond precision/recall/F1 is a debugging signal. It cannot substitute for any of the above:
a model can score bonds well and still generate nothing that parses.

## Safe import path for the cluster

The cluster has local changes in `zinc_gnf/trainer_flow.py` and `scripts/04_train_flow.py`.
Do **not** check those out wholesale.

1. New files — safe to check out directly, nothing else references them:
   `zinc_gnf/bond_pairwise.py`, `zinc_gnf/bond_arms.py`, `scripts/45_train_bond_arm.py`,
   `tests/test_bond_pairwise.py`, `tests/test_bond_arms.py`, `configs/experiment_c_*.yaml`,
   `slurm/45*.sh`. Take `bond_pairwise.py` **before or with** `flow.py`, which imports it.
2. `zinc_gnf/flow.py` and `zinc_gnf/experiment_arms.py` — check out if the cluster has no
   local edits in them (`git diff --stat` first). The `flow.py` change is additive: with
   `bond_aware_coupling=False` the model is byte-identical to before, which
   `test_turning_the_pathway_off_leaves_the_model_byte_identical` asserts by comparing
   state dicts built under the same global seed.
3. `zinc_gnf/trainer_flow.py` — **apply the three hunks by hand**, they are ~20 lines:
   add the `_augment_loss` no-op hook after `_make_batcher`; replace
   `loss = parts["loss"]` with the hook call; add `"nll_per_node"` and `**auxiliary` to the
   `window.append` dict. The hook defaults to returning the loss unchanged with an empty
   dict, so experiments A and B and plain `04_train_flow.py` runs are unaffected.
