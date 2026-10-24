# Stage 22 — architecture review, recommendation and testable hypothesis

## 0. What Stage 21 actually established

Stage 21 (job `340274`) was a NO-GO, and the *shape* of the failure matters for this
design. The `total` assembler removed every fallback (59.36 % → 0 %) and reached 100 %
strict validity, but its QED response **slope went negative** (+0.0311 → −0.0346) and its
within-N correlation inverted (+0.1181 → −0.1419), while average QED rose.

The mechanism: the degree-repair step raises degrees to satisfy `Σdᵢ ≥ 2(n−1)`, which adds
bonds and rings and lifts mean QED — hence the misleadingly better MAE. More importantly,
making the degree equality hold on **100 %** of molecules instead of ~41 % pinned the bond
set to the degree sequence *everywhere*. The 59 % fallback had been acting as an escape
hatch that let the bond logits — the channel carrying the conditional signal — pick the
adjacency. Removing the discontinuity removed the signal path.

Combined with Stages 17–20 this closes off decoder work: the binding constraint is that a
**separately trained decoder stands between the conditional model and the molecule**, and
the property signal does not survive it. Stage 20 measured the other half: a real-vs-
generated embedding classifier reaches **AUC 0.979**, so the generated embeddings are not
on the manifold the decoder was fitted on.

## 1. Literature review

All five families below are *flow* models with exact or bounded likelihoods. The column
that matters for us is the last one.

| model | atoms / bonds | invertibility | validity | property mechanism |
|---|---|---|---|---|
| **GraphNVP** (Madhawa et al., 2019) | two-step: adjacency tensor, then node attributes; both dequantized one-hot | coupling flows, exact likelihood, one-shot | **no guarantee** | latent-space search |
| **MoFlow** (Zang & Wang, KDD 2020) | Glow over bond tensor `B ∈ ℝ^{4×38×38}`, then *graph conditional flow* for atoms `A ∈ ℝ^{38×10}` given bonds | exact, one-pass, 100 % reconstruction | post-hoc valency correction → 100 %; **82 % without correction** on ZINC250K | **latent gradient ascent with a separately trained MLP regressor** |
| **GraphAF** (Shi et al., ICLR 2020) | autoregressive: one atom or bond at a time, dequantized | autoregressive flow, exact | **68 %** without chemical rules, 100 % with rules (masking during sampling) | RL fine-tuning |
| **GraphDF** (Luo et al., ICML 2021) | autoregressive, **discrete** latents via invertible modulo-shift transforms | discrete flow — no dequantization | validity by masking during sampling | RL fine-tuning |
| **cINN** (Ardizzone et al., 2019) | not molecular | conditioning fed to coupling subnetworks; invertibility unaffected | n/a | **native conditional likelihood training** |

### The decisive finding

**None of the molecular flow papers established native target-property conditional
generation.** MoFlow's QED result is explicitly post-hoc latent optimisation — the paper
encodes `Z = f(M)`, trains an MLP to regress `Z → y`, then ascends `Z' = Z + λ·dy/dZ` and
decodes. Its Table 4 reports the *best discovered* QED (0.948, which is the ZINC250K
maximum), not achievement of a requested target. GraphAF, GCPN and MolecularRNN use RL
reward fine-tuning. GraphDF likewise.

So there is **no published QED-MAE, hit-rate or response-slope for native conditional
generation** in this family to compare against, and I will not invent one. Three
consequences:

1. Our Stage 13–21 numbers have no published counterpart; they are not "behind the state
   of the art", because the state of the art did not attempt this task.
2. The user's warning is exactly right: *adding a QED embedding to MoFlow is not known to
   give controllability.* Nothing in the literature licenses that assumption.
3. The conditioning mechanism must therefore be **tested**, not assumed — which drives the
   proof-of-concept design in §4.

The one place native conditional likelihood training *is* established is the cINN line
(Ardizzone et al.): the condition is preprocessed by an unconstrained feed-forward network
and fed to the coupling subnetworks, leaving invertibility untouched, and all parameters
are trained jointly by maximum likelihood. That is the conditioning mechanism I adopt, and
it is the part of the design with actual precedent.

### Three vocabulary distinctions I will hold to

- **Native conditional generation** — sample `G ~ p(G | N, QED)` in one pass. The condition
  enters the model; nothing is scored or filtered afterwards. *This is the only thing
  Stage 22 measures.*
- **Property optimization / latent search** — MoFlow, GraphNVP: encode, ascend a regressor
  gradient in latent space, decode. Needs a property model at inference.
- **Post-generation reranking / RL** — GraphAF, GCPN, MRNN: generate many, score, keep or
  reinforce the best. Uses an oracle at inference or training.

Only the first answers "give me a molecule with N=23 and QED≈0.8".

## 2. What we reuse, and what we replace

Inspected and **reused unchanged** — this is the main reason a direct graph flow is cheap
for us:

| component | why it transfers |
|---|---|
| `preprocessing_charge_aware` + `splits.pkl` | train/val/test split preserved exactly; test never loaded |
| atom vocabulary (17 `(element, charge)` types) | **richer than MoFlow's 9+1**: formal charge is a first-class atom type, so charged species need no special case |
| Kekulé bonds (`clearAromaticFlags=True`), 4 bond types | identical to MoFlow's ZINC250K setup; aromaticity is *inferred by RDKit at sanitisation*, not modelled |
| canonical SMILES atom ordering (`chemistry_charge_aware.py:143`) | the fixed ordering GraphNVP/MoFlow require, already deterministic |
| `MolecularBatch`: `atom_types [B,N]`, `bond_types [B,N,N]`, `node_mask`, `pair_mask`, `n_node`, `qed`, `condition [B,2]` | **already the MoFlow tensor layout** |
| `graph_to_mol_charge_aware(atom_types, bond_matrix)` | discrete graph → sanitized molecule: exactly a direct flow's decode step |
| `generation._molecule_record` | keeps "valid" meaning what it means in Stages 17–21 (sanitizable **and** single connected fragment) |
| `ConditionScaler`, `standardise_condition` | train-only standardisation of (N, QED), already audited |
| `qed_target_metrics`, `qed_achievement_table`, `decoder_metrics`, `internal_diversity` | the same metric definitions, so numbers stay comparable |
| `JobLogger`, `atomic_write_json`, resumable-trainer pattern | infrastructure |

**Replaced** (not deleted — Stage 1–21 code, checkpoints and results stay):

| replaced | by |
|---|---|
| `GraphAutoencoder` encoder/decoder | nothing: the graph is modelled directly, so there is no embedding to invert |
| `NodeFlow` over node embeddings | `ConditionalGraphFlow` over `(atoms, bonds)` |
| the three bond assemblers (`raw`/`constrained`/`degree`/`total`) | `argmax` of the generated tensors, plus an optional **separately reported** MoFlow-style valency correction |
| degree head and degree-constrained integer programs | nothing: bonds are generated, not reconstructed |

The existing GNF remains functional and is the **baseline** in §5.

## 3. The design

### Representation

Exactly the dataset's own tensors, dequantized:

- atoms `x_a = onehot(X) + s·u`, shape `[B, N, 17]`, `u ~ U[0,1)`
- bonds, **strict upper triangle only**, `x_b = onehot(A_ij) + s·u`, shape `[B, P, 4]` with
  `P = N_max(N_max−1)/2`

Modelling only the upper triangle is what makes symmetry *structural*: the lower triangle
is a deterministic mirror, so it must be excluded from the density or it would be
double-counted. There is **no virtual atom type**: `N` is given by the condition, so the
model generates exactly `N` atoms and the padded rows are excluded from the likelihood
entirely. Exact `N` is therefore satisfied by construction — which, as the repo has
documented since Stage 12, is *structural and never evidence of learned conditioning*.

Dequantization gives a bound, not an exact likelihood:

```
log P(G | c)  ≥  E_u[ log p(x | c) ]  +  D·log s ,      D = 17·N + 4·P_real
```

so the reported NLL is an upper bound on the discrete NLL. It is **not comparable** to the
embedding flow's continuous density — different representation, different base measure.
Per the task constraints I report both and compare neither.

### Invertible transformations

Three mask-aware layer types, composed in blocks, acting on `[B, positions, channels]`:

1. **masked ActNorm** — per-channel `y = x·exp(log s) + t`, data-dependent init from the
   first batch's masked statistics; `logdet = Σ_real Σ_ch log s`.
2. **masked invertible 1×1** — `y = x·W` over the channel axis (4×4 for bonds, 17×17 for
   atoms); `logdet = n_real · slogdet(W)`.
3. **masked conditional affine coupling** — split channels; `(s, t) = subnet(x_pass, ctx)`
   with `s = max_log_scale · tanh(raw)`; `y_act = x_act·exp(s) + t`.

Both bounds are load-bearing, and I measured why rather than assuming it. Stage 18 traced
float32 overflow in the *embedding* flow to an unbounded translation head inside a
sequential recursion. This flow is one-shot, so that recursion does not exist — but
**sampling is the inverse**, so an ill-conditioned inverse silently corrupts generated
molecules while the training loss still looks healthy. A stack of *K* couplings each
bounded by *L* has inverse condition number up to `exp(K·L)` against float32's ~1e-7
relative precision, and error compounds *backwards* through the stack. Measured round-trip
relative error, with perturbed subnets:

| configuration | bond stack | atom stack |
|---|---:|---:|
| `max_log_scale=2.0`, `max_shift=10`, 10 blocks | — | **4.4 (destroyed)** |
| `max_log_scale=0.75`, `max_shift=10`, 6 blocks | 1.6e-05 | **0.18 (destroyed)** |
| `max_log_scale=0.75`, `max_shift=3`, 4 blocks | 1.7e-06 | 8.5e-05 |
| **`max_log_scale=0.75`, `max_shift=3`, 6 blocks** (the PoC) | **3.4e-06** | **5.1e-05** |
| `max_log_scale=0.75`, `max_shift=3`, 12 blocks | 2.9e-05 | 5.4e-02 |

Two conclusions. The shift bound must be sized to the data (which lives in
`[0, 1+noise_scale]`), not left generous — a ±10 shift inflates intermediate magnitudes
block over block and the 17-channel atom stack fails before the 4-channel bond stack does.
And **this architecture has a practical depth limit in float32 of roughly 6–8 blocks per
stack**, which is why the PoC uses 6 and why scaling up means float64 inversion or more
aggressive bounds, not simply more blocks. `sample()` therefore reports
`latent_round_trip_relative_error` on every call, and a test asserts the guard *fires* on
a deliberately ill-conditioned configuration rather than passing silently.

Padded positions pass through untouched and contribute nothing to any log-determinant.

### Keeping bond symmetry exact

Because only the upper triangle is modelled, symmetry cannot be violated. The coupling
*conditioner* is still built to be symmetric in `(i,j)` so the model cannot learn an
orientation: for pair `(i,j)` it sees `[x_pass(i,j), r_i + r_j, |r_i − r_j|]` where `r` is
the per-atom sum over its incident pairs. Both endpoint features are symmetric functions
of the unordered pair. The dequantization noise is drawn once per pair, not per ordered
entry.

### Two blocks, bonds first

Following MoFlow's factorisation, which is the part of MoFlow that is well supported:

```
log p(G | c) = log p_bond(x_b | c) + log p_atom(x_a | A, c)
```

- **bond block** — Glow-style stack over `[B, P, 4]`, conditioned on `c`.
- **atom block** — stack over `[B, N, 17]` whose conditioner additionally performs one
  relational message-passing step over the *generated* bond tensor, so atoms are modelled
  **given** bonds: `m_i = Σ_c Σ_j A_c(i,j) · W_c · x_pass(j)`. This is MoFlow's "graph
  conditional flow" in miniature.

At sampling time bonds come from the model, so atom-block errors compound on bond-block
errors. That is inherent to the factorisation and is reported, not hidden.

### The conditioning mechanism, precisely

This is the part the user warned about, so it is specified and *measured* rather than
assumed.

`c = [N_std, QED_std]` → `h_c = MLP(c)` (a cINN-style unconstrained conditioning network,
trained jointly). `h_c` then enters in **two distinct places**, which are ablated
separately:

1. **inside every coupling** — concatenated to the subnet input, so it modulates the
   transformation itself. Invertibility is untouched: `c` is available in both directions.
2. **in the prior** — `μ(c), log σ(c)` per channel.

These are not interchangeable, and conflating them is how a model appears conditional
while being nothing of the kind:

- A **prior-only** conditional flow shifts the whole latent cloud. It can move average QED
  without reshaping the distribution — which is *exactly the artefact Stage 21's `total`
  decoder produced* (QED up, slope negative). Cheap for the optimiser to find, and nearly
  worthless.
- **Coupling conditioning** reshapes `p(G|c)` itself and is what genuine control requires.

So the evaluation reports the **prior-only response** and the **full response** separately,
by neutralising the coupling conditioning at sampling time on a trained model — the same
zero-training ablation trick `zinc_gnf/warm_start.py::neutralised` introduced in Stage 17.

**Conditioning dropout.** With probability `p_drop` the condition is replaced by a learned
*null* embedding during training, so the model learns `p(G|c)` and `p(G)` in one set of
weights. Two reasons, both diagnostic: it gives an exact within-model unconditional
reference for the response measurement, and it penalises ignoring `c` (when `c` is present
it must earn its keep). **Sampling is native only** — no guidance, no `w > 1`
extrapolation, no interpolation between conditional and unconditional scores. Guidance
would be a different claim and is explicitly out of scope.

### Chemistry: deliberately outside the generative model

A hard valence constraint is not an invertible transformation and cannot be part of a
normalizing flow without destroying the density. The literature splits here, and the split
is informative:

- MoFlow applies **post-hoc correction** (delete one bond order from the largest-order bond
  of an over-valent atom, repeat, keep the largest connected component) — correction
  changes the molecule *after* sampling, so corrected validity is not the model's validity.
  Note it can also change `N`, so exact-`N` must be measured before *and* after.
- GraphAF/GraphDF **mask infeasible actions during sampling** — 100 % validity, but the
  sampling distribution is then no longer the learned density.

Stages 17–21 are a cautionary tale about letting a chemistry-correction stage absorb the
property signal. So Stage 22 keeps the generative model chemistry-free and reports:

- **raw validity** — `argmax` of the generated tensors, no correction. *This is the
  architectural number.*
- **strict validity after correction** — the MoFlow-style corrector, reported separately
  and never merged.

For reference, the only published figures of this kind: MoFlow 82 % and GraphAF 68 %
validity-without-check on ZINC250K. Our embedding pipeline's raw validity is ~14 %
(unconditional Arm B) and ~9 % (conditional). That gap is the second hypothesis.

### Sampling, temperature, diversity

Reduced-temperature sampling `z ~ N(μ(c), (t·σ(c))²)`, as in Glow/GraphNVP/MoFlow (MoFlow
used `t = 0.85`). A small sweep is reported because temperature trades diversity against
validity, and the Stage 20/21 evidence is that temperature also changes the *conditional*
behaviour, not only quality.

## 4. Alternatives considered, and why not

**GraphDF-style discrete flow.** Genuinely removes dequantization bias, which is a real
defect of the recommended design. Rejected *for now* because it is autoregressive (so the
condition must survive a long sequential chain and the likelihood becomes
ordering-dependent) and because invertible modulo-shift transforms on discrete variables
are substantially less expressive than continuous couplings. If the PoC shows the
conditional signal is present but the *dequantization noise* swamps it, this is the next
thing to try.

**GraphAF-style autoregressive flow.** Its validity masking is attractive. Rejected as the
primary because (a) masking during sampling means the samples no longer come from the
learned density, which is precisely the kind of post-hoc intervention Stages 17–21 showed
can absorb the property signal; (b) the condition's influence is diluted across the
sequence and depends on the generation order; (c) one-pass likelihood is what makes the
cheap conditional diagnostics in §6 possible.

**Keep the embedding GNF and add a QED-predictive auxiliary loss to the flow.** This was
Stage 20's option A. Rejected as the *next* step because Stage 20 measured AUC 0.979
between real and generated embeddings: the decoder is being asked to invert points it was
never fitted on, and an auxiliary loss on the flow does not change that. It remains the
fallback if the direct flow fails.

**Diffusion / flow-matching on graphs.** Current best-performing family for graph
generation, but it abandons exact likelihood — and the conditional *likelihood gap* is the
single most useful diagnostic this project has. Out of scope for a bounded PoC.

## 5. The testable hypothesis

> **H1 (primary).** A one-shot conditional graph flow trained by maximum likelihood on
> `(G, N, QED)` produces a **measurable, monotone within-N QED response** in *natively*
> sampled molecules: a positive requested-vs-realised response slope and a within-N
> correlation whose bootstrap CI excludes zero, at N ∈ {15, 23, 30} and QED ∈ {0.4, 0.6,
> 0.8}.
>
> **H0 (the control that makes H1 meaningful).** The *same architecture, budget and seed*
> trained with QED labels **permuted within N** shows no such response. If the real arm
> does not separate from the shuffled arm, the architecture is not using the condition and
> the result is negative regardless of how good the molecules look.
>
> **H2 (secondary).** Raw validity (no correction) substantially exceeds the embedding
> pipeline's ~9–14 %.

The shuffled-condition control is the heart of the design. Every stage from 13 to 21
produced plausible-looking molecules and no QED control; the only way to avoid repeating
that is to make "is the condition being used at all?" a measured quantity with its own null.

### Distinguishing architecture from undertraining

A small PoC **cannot** by itself separate "wrong architecture" from "undertrained", and I
will not claim otherwise. Pre-registered reading, decided before the run:

| observation at end of budget | reading |
|---|---|
| validation conditional likelihood gap (real vs permuted `c`) **still rising** | undertrained — extend the budget, do not conclude |
| gap **plateaued > 0** and sampled response positive, separating from the shuffled arm | H1 supported at PoC scale → proceed to staged scale-up |
| gap **plateaued ≈ 0**, indistinguishable from the shuffled arm | the architecture is not using the condition; a larger run will not fix it |
| gap plateaued > 0 but sampled response ≈ 0 | the density notices `c` while sampling does not follow it — the *same* contradiction Stage 17 hit, now in graph space, and a genuinely new finding |

The last row matters: Stage 17 found a QED-only likelihood gap of ≈ +0.074–0.087 in all
three embedding models with a 4.5× spread in sampled response. If that dissociation
reappears in graph space, the problem is not the representation, and that would redirect
the whole project.

**Staged validation plan.** Stage 22a (this PoC, bounded, both arms) → 22b, if and only if
22a supports H1: single arm, 4–5× the budget, full grid, temperature sweep → 22c: matched
comparison against Stage 17 at equal sampling budget, with the test split still untouched.

## 6. Cost

| | PoC (Stage 22a) |
|---|---|
| model | 6 bond coupling blocks + 6 atom coupling blocks, hidden 256 |
| parameters | ≈ 3–6 M (MoFlow's ZINC250K config used 10 + 38 blocks at 512 wide) |
| steps | 6,000 per arm, batch 32 |
| arms | 2 (real conditions, shuffled-within-N conditions) |
| data | full TRAIN split for fitting; VALIDATION for model selection only; **test never loaded** |
| GPU memory | bond tensor `[32, 703, 4]` + atom tensor `[32, 38, 17]` and activations ≈ **< 4 GB**; 16 GB requested for headroom |
| runtime | ≈ 1.5–3 h per arm, both arms in one job inside an 8 h limit; checkpointed every 500 steps and **resumable** |
| evaluation | CPU, matched against the Stage 17 warm start |

No blind 50k-step run; no full-scale training; nothing claimed about deployment.

## Sources

- MoFlow — Zang & Wang, *MoFlow: An Invertible Flow Model for Generating Molecular Graphs*,
  KDD 2020. [arXiv:2006.10137](https://arxiv.org/abs/2006.10137) ·
  [code](https://github.com/calvin-zcx/moflow)
- GraphNVP — Madhawa et al., *GraphNVP: An Invertible Flow Model for Generating Molecular
  Graphs*. [arXiv:1905.11600](https://arxiv.org/abs/1905.11600) ·
  [code](https://github.com/pfnet-research/graph-nvp)
- GraphAF — Shi et al., ICLR 2020. [arXiv:2001.09382](https://arxiv.org/abs/2001.09382)
- GraphDF — Luo et al., ICML 2021. [arXiv:2102.01189](https://arxiv.org/abs/2102.01189)
- cINN — Ardizzone et al., *Guided Image Generation with Conditional Invertible Neural
  Networks*. [arXiv:1907.02392](https://arxiv.org/abs/1907.02392)
- On ordering/dequantization limitations of this model family:
  [arXiv:2210.06032](https://arxiv.org/abs/2210.06032)

*Content from these sources was read and paraphrased; architectural facts (MoFlow's
tensor shapes, its `U[0,0.6]` dequantization, its validity-correction rule, its
latent-gradient-ascent property procedure, the 82 %/68 % validity-without-check figures,
GraphAF's 68 %/100 %) are reported as stated by the papers. No native-conditional QED
benchmark is quoted, because none was established.*
