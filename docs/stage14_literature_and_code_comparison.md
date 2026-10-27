# Stage 14, Part B — what the reference implementations actually do

Every row below was read in the **source**, not inferred from a paper. Clone points and
file/line references are given so each claim can be rechecked.

| Repository | Commit / state when read |
|---|---|
| `sridharmatta1/graph-normalising-flow` | shallow clone, latest commit `Add condition-ablation experiment for the N-conditioned molecular flow`, 2026-10-21 13:56 +0200 |
| `calvin-zcx/moflow` | shallow clone of `main` |
| GNF (arXiv 1905.13177) | abstract + §1 readable via ar5iv; **the PDF body is not retrievable with the tools available here** (arXiv PDF is rejected as an unsupported content type and ar5iv returned only the front matter). Claims below are limited to what was readable, and that limit is stated rather than papered over. |
| MoFlow (arXiv 2006.10137) | methodology taken from the **code**, which is authoritative for what we would reuse |
| GraphAF / GraphNVP | **not verified in source for this stage.** Both are autoregressive-or-coupling flows on adjacency tensors; GraphAF does property optimisation by RL fine-tuning, GraphNVP is unconditional. Neither offers a mechanism we could drop into a frozen-autoencoder node-feature flow without retraining, so they were deprioritised rather than skimmed and cited. |

## 1. Teammates' QM9 implementation

| Mechanism | What their code does | Where |
|---|---|---|
| Conditional prior | `ConditionalPriorNetwork`: MLP on the N-embedding → `mu`, `pre_sigma`; `sigma = softplus(pre_sigma) + min_sigma` (floor 0.1); **both heads zero-initialised** so `sigma` starts exactly at `min_sigma_init = 1.0` regardless of the random N-embedding | `conditional_prior.py:47-96` |
| Per-node prior draw | one i.i.d. vector per node from that single per-graph `N(mu(N), sigma(N)^2)` | `conditional_prior.py:132-146` |
| Joint N + property | `NPropertyEmbedding`: `concat[log(N+1), (p - mean)/std] → MLP(hidden, embed)`. Property mean/std are **CLI flags with hardcoded defaults** (`mu`: 2.6520 / 1.4035) | `property_conditioning.py:27-73`, `generate_molecules_property_conditioned.py:75-77` |
| FiLM coupling | the embedding feeds FiLM `(gamma, beta)` generators inside each coupling; `NConditionedMolecularGNFBlock` takes an `embedding_module`, so N-only and N+property share one architecture | `molecular_flow.py`, used in both generate scripts |
| Temperature | unconditional script only: `sample_temperature` multiplying the prior draw, explicitly described in-code as matching "MoFlow's reduced-temperature generation" | `generate_molecules.py:104-106, 218` |
| Latent scaling | N-conditioned script only: `sigma_scale` multiplies the **learned** `sigma(N)` at generation time, added to test whether a too-narrow `sigma(N)` caused the N=5 uniqueness regression | `generate_molecules_n_conditioned.py:117-125, 205` |
| Graph reconstruction / validity | `decode_bonds_valence_aware(atom_types, bond_probs)`: greedy, valence-respecting assembly from per-pair probabilities. Their own comment states independent per-pair argmax left **~15% of round trips invalid despite ~97% per-pair accuracy** | `qm9_chem.py:40-52` |
| Conditioning ablation (new, today) | `NullEmbedding` returns a constant zero vector for every molecule, keeping FiLM, width, prior and hyperparameters identical while the network cannot see N — i.e. an ablation that **requires retraining** | `null_conditioning.py` |
| Calibration diagnostic | `latent_stats.py` + `inspect_latent_calibration.py`: per-dimension `zscore_rms`, kurtosis, `E|u|/rms(u)`, bias | `molecular_generation/` |

## 2. MoFlow — how it actually reaches a property value

**It does not condition.** `mflow/optimize_property.py`:

* `MoFlowProp` = the frozen flow's encoder plus `propNN`, a small MLP on the latent vector, trained with MSE on the property (`fit_model`, lines 72-113).
* `optimize_mol` (lines 130-200): encode a **real** molecule, then repeat
  `cur_vec = cur_vec + lr * grad / sqrt(grad*grad)` — a *normalised* gradient step on the predicted property — decode **every visited latent**, run `check_validity`, drop anything below a Tanimoto `sim_cutoff` to the starting molecule, and sort the survivors by the property measured with RDKit.
* `generate.py` exposes `--temperature` (default 1.0) for ordinary sampling.

So MoFlow's answer to "give me a specific property" is **latent search with a property predictor plus RDKit validation**, started from data — a *selection/optimisation* method, reported as such, not evidence of a conditional generative model.

## 3. What we already have, and what is left to reuse

| Mechanism | Us | Status |
|---|---|---|
| `mu(c), sigma(c)` prior with softplus floor and zero-init heads | `zinc_gnf/flow.py:NodePrior` | **already theirs, copied deliberately** |
| FiLM in every coupling, one architecture for N-only and N+QED | `flow.py:CouplingNetwork/NodeAutoregressiveCoupling` | already have, plus per-feature **present flags** they do not have |
| Condition standardisation | train-only `ConditionScaler` persisted to `scaler.json`, re-used at sampling | **stronger than theirs** (they pass mean/std as flags) |
| Sampling temperature | `--temperature`, swept | already have |
| `sigma_scale`-style latent widening | rejected in favour of stage 10's maximum-likelihood prior calibration (same cure, keeps a normalised density) | deliberate difference |
| Valence-aware greedy decode | `decoding_charge_aware.valence_aware_bond_decode_typed` | already have |
| Degree-constrained MILP decode | ours only | **stronger than theirs** |
| Conditioning ablation | permutation controls needing **no retraining** (`condition_sensitivity`, incl. QED-only) | **stronger than `NullEmbedding`** |
| Latent property predictor + target-seeking latent steps | `zinc_gnf/qed_guidance.py` (new) | **the one thing worth reusing from MoFlow** |
| Candidate reranking by RDKit property | `qed_guidance.select_best`, equal-budget accounted | new, labelled as selection |

**Nothing in either reference repository would fix native QED control.** Their joint
conditioning is architecturally the same as ours (and ours is a superset); the only
property-*targeting* machinery that exists in either is MoFlow's latent search, which is
explicitly not conditioning. QM9 numbers do not transfer: QM9 molecules have ≤9 heavy
atoms and their conditioned property is a physical quantity, not a 0-1 composite like QED.

## 4. Why this matters for our numbers

Our native joint model gives a pooled response slope of **0.0729** with a positive
QED-only permutation gap (**+0.0764**): QED *reaches* the density but barely moves the
sample. MoFlow's code says the field's working answer to "hit this property value" is a
latent corrector plus RDKit measurement — so implementing that (separately reported, at
equal candidate budget) is a defensible quick win, while improving native conditioning is
a modelling project, not a fix.

Any guided or reranked result must be presented as what it is. `qed_guidance.METHOD_LABELS`
carries that wording into every report so a selection result can never be printed as
native GNF conditioning.
