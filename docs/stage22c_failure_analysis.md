# Stage 22c — root-cause analysis of the graph-flow generation failure

Diagnosis only. No training was run, no checkpoint was modified, and the two code fixes
below do not alter any existing checkpoint's behaviour.

## Summary

| # | finding | class | evidence | needs retraining? |
|---|---|---|---|---|
| 1 | `--max-log-scale` defaulted to **2.0** in the training script while `GraphFlowConfig` said **0.75**; the Stage-22a checkpoint was trained at 2.0 | **verified defect (mine)** | code inspection; measured inverse collapse at 2.0 | yes, for a trustworthy prior-only ablation |
| 2 | Resuming did **not** check the checkpoint's `model_config`; `max_log_scale` is a plain float, so `load_state_dict` accepts a mismatch silently | **verified defect (mine)** | code inspection; now a test | no |
| 3 | The bond conditioner is **provably blind to topology at a fixed degree sequence** | **verified architectural limitation** | exact 0.000e+00 difference between a 6-ring and two 3-rings | yes — it is a model change |
| 4 | Representation, padding, masking, bond symmetry, dequantization, discretisation | **verified CORRECT** | 5/5 exact SMILES round trips on real molecules | — |
| 5 | Training and inverse-sampling representations | **verified CONSISTENT** | code trace; same tensor, same semantics both directions | — |
| 6 | Model undertrained | **confirmed, but insufficient** | `rising_undertrained`; see §3 | — |

**The headline: raw validity 0.0012 is not explained by undertraining alone, and more
training will not fix it.** Finding 3 is the binding constraint.

## 1. Verified defects

### 1.1 The configuration mismatch — mine, and it mattered

I lowered `GraphFlowConfig.max_log_scale` from 2.0 to 0.75 *after measuring* that 2.0
destroys the inverse at depth, and never updated `scripts/23_train_graph_flow.py`, which
kept `default=2.0`. The Stage-22a checkpoint therefore trained at exactly the value I had
already shown to be unsafe.

Fix: the script's defaults are now **read from the dataclass**, so they cannot diverge
again, and `--max-shift` is exposed for the same reason.

### 1.2 Resuming could silently change the model — mine

`max_log_scale`, `max_shift` and `noise_scale` are plain floats, not parameters. Nothing
in `_resume` compared them, and `load_state_dict` raises no shape error when they differ:
resuming with different flags would quietly implement a **different function with the same
weights**. This is also the mechanism by which 1.1 went unnoticed.

Fix: `_resume` now compares the checkpoint's `model_config` against the constructed model
and refuses with the exact flag needed. Your existing checkpoint still resumes — with
`--max-log-scale 2.0` passed explicitly, which is the honest way to continue it.

### 1.3 Answering your question 6 directly

- **Sampling stability:** yes, it mattered. See §4.
- **Checkpoint resumption:** it was a live hazard, now closed. The *evaluation* path was
  never affected: `load_graph_flow` rebuilds the config from the checkpoint, so Stage 22b
  evaluated the model as trained (2.0).

## 2. What is NOT broken

### 2.1 The representation round-trips exactly (your question 3)

Real molecules → padded tensors → dequantize → `argmax` → graph → RDKit, requiring the
canonical SMILES back:

```
N=11 atoms_exact=True bonds_exact=True smiles_match=True   CC(=O)Nc1ccc(O)cc1
N= 6 atoms_exact=True bonds_exact=True smiles_match=True   c1ccccc1
N=15 atoms_exact=True bonds_exact=True smiles_match=True   CC(C)Cc1ccc(C(C)C(=O)O)cc1
N=18 atoms_exact=True bonds_exact=True smiles_match=True   O=C(Nc1ccc(F)cc1)c1cc(Cl)ccc1Br
N=11 atoms_exact=True bonds_exact=True smiles_match=True   CCOC(=O)c1ccccc1
-> 5/5 exact round trips
```

Covering halogens, a Kekulé aromatic ring, and a thiophene. So **atom-type sampling,
padding masks, bond symmetry, dequantization and graph discretisation are all correct**.
With `noise_scale=0.6` the true class lands in `[1, 1.6)` and every other class in
`[0, 0.6)`, so `argmax` is exact by construction — not merely empirically.

### 2.2 Training and sampling use the same representation (your question 5)

The atom block conditions on `unpack_upper(bonds)` in both directions, with the same
dequantized semantics: during training the data's bond tensor, during sampling the
model's own. Padded pairs are zero in both. The diagonal is never modelled and is zero in
both. No inconsistency.

## 3. The real cause: the bond conditioner cannot see topology

`BondSubnet` gives pair *(i,j)* only its own pass-through channels plus the per-atom sums
`r_i + r_j` and `|r_i − r_j|`. **Any two graphs with the same degree sequence are therefore
identical to it.** Measured on a 6-ring versus two 3-rings — both all-degree-2 — comparing
the pairs present in both:

```
pair (0,1): |d scale| = 0.000e+00   |d shift| = 0.000e+00
pair (1,2): |d scale| = 0.000e+00   |d shift| = 0.000e+00
pair (3,4): |d scale| = 0.000e+00   |d shift| = 0.000e+00
pair (4,5): |d scale| = 0.000e+00   |d shift| = 0.000e+00
```

Exactly zero. The model cannot distinguish a benzene skeleton from two cyclopropanes.

**Why this explains the observations.** Such a conditioner can learn *marginals* — how many
bonds an atom has, how often a pair is bonded — but not *which specific* pairs, so it
behaves close to a per-pair-independent model with degree regularisation. Consequences:

- **Raw validity ≈ 0** (your question 1). Validity needs ~253 pair decisions at N=23 to be
  jointly correct. Independent-ish decisions with the right marginal give the right bond
  *count* and the wrong bond *set*.
- **Disconnection.** ~25 edges placed near-independently on 23 nodes is around the random-graph
  connectivity threshold: a giant component plus isolated atoms is the typical outcome.
- **Exact N 1.8 % after correction** (your question 2). `valency_correction` keeps only the
  **largest connected component**, so a fragmented graph loses atoms before anything else
  happens. `largest_component_fraction_of_n` upper-bounds the achievable exact-N rate.
- **Corrected validity 1.0000 with aromatic rings 0.003 and uniqueness 0.581.** These are
  mutually consistent with small fragments: a 2–3 atom fragment is trivially valid and
  connected, has no rings, and collides often — 42 % duplicates.

**This is the part that is not undertraining.** The trend said `rising_undertrained`, and
more steps would improve the likelihood and the conditional gap. But no amount of training
can make a conditioner express a distinction it is structurally blind to, and connectivity
and ring closure are exactly that distinction.

### 3.1 The constraint that caused it was unnecessary

I chose sum aggregates to keep the conditioner symmetric in *(i,j)*. **That was not
needed.** Only the strict upper triangle is modelled and `unpack_upper` mirrors it, so each
pair appears exactly once and matrix symmetry holds *whatever* the conditioner computes. I
paid expressiveness for a guarantee the packing already provided.

For contrast, MoFlow uses 10 coupling layers of 3×3 **convolutions** over the N×N adjacency
grid at 512 hidden units — a genuine receptive field over pair structure — and reports 82 %
validity-without-correction on ZINC250K.

### 3.2 Proposed minimal fix (requires retraining; not implemented here)

Keep everything else and replace only the conditioner: scatter the pass-through channels
into the dense N×N grid, apply a small 2-D convolutional or attention block over it, gather
back to the upper triangle. Invertibility is untouched (it still reads only pass-through
channels), symmetry is still structural, and the model gains the ability to condition on
specific neighbourhoods. This is a model change, so it needs a fresh run — it cannot be
applied to the Stage-22a checkpoint.

## 4. Why the prior-only inverse is unstable (your question 7)

The subnets were trained *with* a context vector. Neutralising it pushes them
off-distribution, the log-scales saturate toward their bound, and the inverse condition
number is ~`exp(blocks × max_log_scale)`. Measured with deliberately perturbed subnets:

| `max_log_scale` | fraction of \|s\| at the bound | round trip, with context | round trip, neutralised |
|---:|---:|---:|---:|
| 0.75 | 0.801 | 1.8e-03 | 1.5e-03 |
| 2.0 | 0.869 | 7.0e+01 | 7.1e+01 |

`exp(6 × 0.75) = 90` versus `exp(6 × 2.0) = 162,755`, against float32's ~1e-7 relative
precision — so 2.0 leaves no margin. That is why your run reported **3.3e-05 for the full
arm** (whose subnets saw their trained context) and **1.4e-02 for the prior-only arm**.

**Consequence: the prior-only column in the Stage-22b report is not trustworthy** and
should not be used to argue about where the conditioning lives. At 0.75 it would be.

## 5. Is the reported QED response meaningful? (your question 8)

**No.** Corrected slope +0.0907 against the shuffled arm's +0.0113 looks like separation,
but it is measured on molecules that are **not the requested molecules**: exact N survives
in 1.8 % of outputs, so the population is mostly small fragments of the requested graph.
A slope over fragment QED is not evidence of control over *p(G | N=23, QED=0.8)*.

Two further reasons to discard it: the full arm's own raw population has 0.12 % validity,
so the corrected population is almost entirely the corrector's product rather than the
model's; and the prior-only comparison that would separate genuine control from a shifted
prior is numerically invalid per §4.

## 6. Verified failure-mode attribution (your question 4)

§3 predicts **disconnection dominant, over-valence secondary**, but the Stage-22b report
does not record which. `scripts/25_graph_flow_diagnostics.py` settles it on the real
checkpoint: it attributes every sample to exactly one of `valid`,
`over_valent_only`, `disconnected_only`, `over_valent_and_disconnected`,
`other_sanitisation_failure`, and reports bond density, bond-type and atom-type histograms
**against the real training molecules at the same N**, component counts, largest-component
size, atoms lost and orders removed by correction, and scale saturation plus round-trip
error with and without context.

It is CPU-only, read-only, needs no training, and takes minutes:

```bash
python scripts/25_graph_flow_diagnostics.py \
    --checkpoint outputs/checkpoints/graph_flow_real/last.pt \
    --output-dir outputs/metrics/graph_flow_diagnostics \
    --target-n 15,23,30 --target-qed 0.8 --temperature 0.85 --count 256
```

**How to read it.** If `bonds_mean` and the atom-type fractions match the real reference
while the graphs stay invalid, the model learned the marginals and not the correlations —
confirming §3 and making further training on this architecture pointless for validity. If
the marginals are *also* wrong, undertraining is contributing and the two must be
separated before a Stage 23 decision.

## 7. Recommendation

1. **Run the diagnostic** (minutes, CPU) before any further training.
2. **Do not extend the Stage-22a run.** It is undertrained *and* architecturally limited;
   more steps spend GPU hours on a model that cannot express connectivity.
3. **If the diagnostic confirms matched marginals with broken topology**, Stage 23 is the
   conditioner replacement of §3.2 at `max_log_scale=0.75`, with the same two-arm protocol.
4. **Keep the acceptance criteria where they are:** native raw validity, exact N, and a QED
   response measured on molecules that actually have the requested N — not corrected
   validity, and not a likelihood improvement.

One honest note on expectations: this analysis explains the failure but does not establish
that the fixed conditioner will reach useful validity. MoFlow's 82 % came from a much
larger model (10 + 38 coupling layers at 512 hidden) trained to convergence. Our PoC budget
buys a test of the *mechanism*, not that number.
