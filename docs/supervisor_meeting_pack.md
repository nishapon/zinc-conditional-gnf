# Supervisor meeting pack — unconditional vs conditional graph normalizing flow

Prepared from the code at commit `8751f3f`. **Every number in this document is either
read from code or labelled with its source. Nothing is estimated.**

Read section 0 first. It determines how much of the rest can be claimed tomorrow.

---

## 0. Evidence status, stated before any numbers

### 0.1 This workspace contains no measured results

| Artefact | Status |
|---|---|
| `outputs/checkpoints/` | **empty** (only `.gitkeep`; `outputs/*` is gitignored) |
| `outputs/metrics/`, `outputs/samples/`, `outputs/figures/`, `outputs/logs/` | **empty** |
| `data/` | **absent** — no dataset, no `splits.pkl`, no `scaler.json` |
| Python environment | no `torch`, no `numpy`, no `rdkit` |

Consequence: I could not run, recompute or verify a single generation metric. Sections
2, 3, 4 and the aromatic-ring part of 5 are **PENDING** and are written as empty tables
with the exact command that fills them. This is deliberate, per your instruction that an
unavailable comparison must appear as pending rather than be filled with numbers from a
different architecture.

### 0.2 The only measured numbers committed anywhere

`docs/reported_results.json` — and its own `status` field reads:

> "REPORTED, NOT YET RECOMPUTED from the evaluation artefacts. The brief itself asks for
> recomputation and exact metric definitions; until that is done every derived number
> inherits this caveat."

It contains **conditional (N+QED) numbers only. There is no unconditional row anywhere in
the repository.** So the headline comparison your meeting needs has never been measured.

### 0.3 I could not open your slide deck

The OneDrive link returns **HTTP 403 Forbidden** — it needs a signed-in browser session,
which I have no way to provide. So the improved numbers you want used are not available
to me, and I have deliberately **not** substituted the older `reported_results.json`
values for them.

Three numbers did arrive in your message text, and those I have used, attributed to you:

| From your message | Value |
|---|---|
| matched control, native QED MAE | 0.1922 |
| matched control, raw validity | 13.8% |
| matched control, valid target-hit yield | 30.4% |
| frozen correlated prior, native QED MAE | 0.1885 (validity and yield both down) |
| balanced sampling | no confirmation-MAE improvement |

**To finish this pack I need the deck's numbers as text** — paste the tables, or commit
the deck / its metric JSONs into the repo and I will read them from there.

---

## 1. Architecture comparison

### 1.1 The pipeline, both arms

The flow never sees a molecular graph. It models the density of a **frozen
autoencoder's** per-atom embeddings.

```
molecule graph ──► BondAwareEncoder ──► node embeddings [B,N,D] ──► NodeFlow ──► z
(atom_types,                                                         (density)
 bond_matrix,         FROZEN during flow training
 node_mask)           (trainer_flow.py:259, requires_grad_(False);
                       AE sha256 recorded in every flow checkpoint)

z ──► NodeFlow.inverse ──► embeddings ──► BondAwareDecoder ──► atom + bond logits
                                                            ──► bond assembler ──► molecule
```

| Component | File | What it is |
|---|---|---|
| Encoder | `encoder.py:112` | dense bond-aware self-attention GNN, 6 weight-shared blocks; learned per-bond-type bias on both attention logits and values; attends to self + bonded neighbours only |
| Flow | `flow.py:756` | 12 blocks × (ActNorm → LU invertible 1×1 → 2 affine couplings → optional node-autoregressive coupling) |
| Prior | `flow.py:606` | per-node **independent** Gaussian, `N(mu(c), sigma(c)^2)`, softplus floor, zero-initialised heads |
| Decoder | `decoder.py` | per-node atom MLP (17 charge-aware types) + symmetric pair MLP for bond logits + optional degree head |
| Assembler | `decoding_charge_aware.py` | three variants, always reported separately — see 1.5 |

The encoder **requires the true bond matrix**, so it cannot run at generation time.
Generation feeds the decoder zeros plus masks (`generation.py:85`).

### 1.2 Where N and QED enter — four places, only two learned

The dataset stores `condition = [standardised log1p(N), standardised QED]`, standardised
with a **train-only** scaler (`<data_dir>/scaler.json`). `flow.CONDITION_FEATURES =
("n", "qed")` names the same two columns in the same order; `conditioning.py:57` is the
single place that mapping lives.

| # | Entry point | File | Learned? | Affects |
|---|---|---|---|---|
| 1 | **Node-slot count** — N sets how many node vectors are drawn | `generation.py:275`, `flow.py:node_mask_from_counts` | **No — structural** | N only |
| 2 | **FiLM** `(gamma, beta)` on the hidden layer of every coupling conditioner | `flow.py:312-324` (affine), `flow.py:470-490` (node-autoregressive) | Yes, zero-initialised | N and QED |
| 3 | **Conditional prior** `mu(c), sigma(c)` | `flow.py:606-760` | Yes, zero-initialised heads | N and QED |
| 4 | `ConditionEmbedding` with per-feature present flags | `flow.py:574` | Yes | both |

Entry point 2 is applied as `hidden * (1 + gamma) + beta` **before** the ReLU, once per
coupling conditioner. With 12 blocks × 2 affine couplings (+1 node-autoregressive
coupling per block when enabled) that is 24–36 FiLM sites.

### 1.3 Does atom count determine the node slots before generation? **Yes.**

And this is the single most important point for your meeting. The repo states it in its
own code (`conditioning.py:30-34`):

> "In this pipeline N also sets how many node vectors are drawn, and the decoder emits
> exactly one atom per node vector, so the atom count is satisfied by **CONSTRUCTION**
> even if `mu(N), sigma(N)` and every FiLM layer were dead."

`generation.node_count_agreement` (`generation.py:313`) carries the same warning:
*"structural, not learned: the decoder emits one atom per sampled node vector."*

So **atom-count accuracy is not evidence of conditioning** and must not be presented as
such. There is no insert/delete/stop head anywhere in the decoder.

### 1.4 Does the ORIGINAL unconditional model already require a supplied node count? **Yes.**

`generate_molecules(flow, autoencoder, *, count, node_counts: NodeCountSampler, ...)` —
`node_counts` is a **required** keyword argument, and `sizes` always comes from it or from
`target_n` (`generation.py:275`). `NodeCountSampler` is built from the **training
histogram** of heavy-atom counts (`generation.py:50-82`); p(N) is never learned by the
flow in either arm.

**Therefore the unconditional baseline also receives the atom count.** The conditional
model's advantage on atom count is zero by construction. The only genuinely new
controllable quantity is **QED**. Frame the whole comparison that way and it is
defensible; frame it as "we added N and QED control" and the first question will dismantle
it.

### 1.5 "Corrected validity" means something specific here — and not what MoFlow means

| Assembler | File | What it does |
|---|---|---|
| `raw` | `generation.py:206` | argmax bond matrix, diagonal zeroed, **no chemistry** |
| `constrained` | `decoding_charge_aware.py:36` | accept bonds by descending confidence, **demote** order to fit typed valence caps, then a connectivity rescue pass |
| `degree` | `decoding_charge_aware.py:118` | exact integer program (`scipy.optimize.milp`) maximising summed bonded log-odds subject to `degree(i) == predicted_degree(i)`, then valence demotion, then connectivity; falls back to `constrained` with a recorded reason |

**"Corrected validity" in this project = the strict validity of the `degree` (MILP)
assembler.** That path **deletes nothing**: `_molecule_record` (`generation.py:109`)
requires sanitizable **AND** exactly one fragment **AND** a canonical SMILES. Disconnected
output is invalid, not repaired.

MoFlow-style correction — remove bond orders from over-valent atoms, then **keep the
largest connected component, dropping atoms and changing N** — does exist in this repo
(`graph_flow_generation.py:62 valency_correction`) but belongs to the *discrete graph-flow*
pipeline and **is never called from `generation.py`**. So any slide comparing "our
corrected validity" to a published "corrected validity" is comparing two different
operations. Say so explicitly or drop the comparison.

### 1.6 Which changes are FiLM conditioning and which are separate

| Change | Kind | Status |
|---|---|---|
| FiLM in every coupling conditioner | conditioning | in `flow.py` from the start of the rewrite |
| Conditional prior `mu(c), sigma(c)` | conditioning | same |
| Reading `MolecularBatch.condition` in the trainer; sampler accepting conditional flows; the permutation controls | **plumbing + verification — this is the student-added work** | `conditioning.py` module docstring says the architecture was already conditional and "what was missing is the plumbing" |
| `node_autoregressive` coupling | **separate architectural change** | `uncond.yaml: false` vs `nqed.yaml: true` — **a confound, see 2.3** |
| 64-D → 128-D embeddings | **separate change** (new autoencoder) | committed YAMLs say 64; reported results say 128 |
| `degree_head` + MILP degree-constrained decoder | **separate change** (decoder constraint) | no committed YAML enables it — see 6.4 |
| Bond-aware pairwise conditioner | **separate change**, not yet run | `bond_pairwise.py`, stage 26 |
| Correlated (exchangeable) prior | **separate change** | `coupled_prior.py`, experiment B |
| Balanced N–QED sampling | **training-distribution change** | `balanced_sampler.py`, experiment A |

### 1.7 What each existing result labelled "control" actually represents

Four different things carry that word. They are not interchangeable:

| Label | Where | What it is |
|---|---|---|
| **permuted-condition control** | `conditioning.condition_sensitivity`, and inside `trainer_flow.validate` every eval | same molecules, same smoothing noise, scored against **another molecule's** condition vector. Gap > 0 in nats/node = conditioning is used. **This is the real evidence of conditioning.** |
| **QED-only permutation control** | `condition_sensitivity(permute_features=["qed"])`; `scripts/13` exits 5 if it fails | permutes QED while every molecule keeps its **true N**. Required because N and QED are correlated in ZINC, so permuting both cannot show QED does anything beyond N |
| **arm `control`** (experiments A/B/C) | `experiment_arms.py`, `bond_arms.py` | the **same** conditional checkpoint trained for the same extra steps with no intervention. Absorbs "it trained longer", nothing more |
| **`decoder_control`** | stage 22 scripts | a pairing control for decoder comparisons |

When your supervisor asks "what is the control?", the answer depends on which claim. Use
the table.

### 1.8 Checkpoint inventory — what is specified vs what exists

| Arm | Config | `condition_features` | `node_autoregressive` | emb. dim | AE checkpoint | Steps | Seed |
|---|---|---|---|---|---|---|---|
| Unconditional | `zinc250k_flow_uncond.yaml` | `[]` | **false** | 64 | `ablation/ae_degree/seed_3/last.pt` | 50,000 | 1 |
| N-only | `zinc250k_flow_ncond.yaml` | `["n"]` | true | 64 | same | 50,000 | 1 |
| N+QED | `zinc250k_flow_nqed.yaml` | `["n","qed"]` | **true** | 64 | same | 50,000 | 1 |
| **The one the results came from** | **not committed** | n+qed | ? | **128** | `ablation/ae_degree_128d/seed_3/last.pt` | 50,000 | **3** (per arm scripts) |

**Flagged as missing / inconsistent:**

1. **No config exists for the checkpoint that produced the headline numbers.** Every
   script from stage 30 on defaults to `outputs/checkpoints/flow_nqed_ae128_qedcontrast/seed_3/best.pt`,
   and `configs/` has no file for it. `reported_results.json` says "128-D embeddings,
   50,000 training steps". The committed YAMLs are 64-D. **The configuration of your
   headline model is not in version control.**
2. **No unconditional 128-D checkpoint is referenced anywhere.** A matched baseline for
   the headline model may not exist yet.
3. **Training data:** all configs use `data/processed/zinc250k_charge_aware_moflow_fixed`
   (202,111 train molecules per the nqed config comment). Not present here; unverified.
4. **Seeds:** configs say seed 1, the arm scripts say seed 3. Which seed produced the
   headline numbers is **unknown to me**.
5. **Evaluation settings** differ by script: `scripts/13` uses 500 samples/cell at
   temperature 0.2; `scripts/42` uses 50 samples/cell at temperature 0.2. **Temperature
   0.2 is not the model's own density** — it is a strong narrowing of the prior, and every
   validity and diversity number inherits it. This must be stated on the slide.
6. **Only one seed per architecture.** Every comparison available is a *sampling*
   repetition, not independent training runs. Say "n=1 model" out loud.

---

## 2. Main generation-quality comparison — **PENDING**

### 2.1 Why it does not exist yet, and what I built

`scripts/42_checkpoint_eval.py` is the right protocol (shared noise ids, one metric
implementation, one budget) but its `verify_compatibility` (line 333) **raises unless all
checkpoints agree on `condition_features`** — so it refuses an unconditional-vs-conditional
pair by construction. Hand-merging a `05_generate.py` run with a `13_generate_nqed.py` run
risks mismatched budget, temperature and novelty reference.

I wrote **`scripts/46_uncond_vs_cond.py`**, which runs both arms in one process holding
identical: metric implementation (`metrics_generation.generation_metrics`), sample budget,
novelty reference (train-split canonical SMILES), temperature, decoder checkpoint, and
**the same sampled N vector for both arms**.

Conditional requests are drawn as **whole (N, QED) rows from the training split**, so
their empirical relationship is preserved — sampling the two marginals independently would
hand the conditional model combinations ZINC does not contain and then score it on them.

```bash
python scripts/46_uncond_vs_cond.py \
  --unconditional-checkpoint <uncond best.pt> \
  --conditional-checkpoint   <nqed best.pt> \
  --ae-checkpoint outputs/ablation/ae_degree_128d/seed_3/last.pt \
  --output-dir outputs/metrics/uncond_vs_cond \
  --count 10000 --temperature 1.0 --headline degree --inspect-only   # drop to run
```

`--inspect-only` prints the checkpoint diff and the request joint without generating —
**run that first**, because it tells you whether a clean contrast is even available.

### 2.2 The table to fill

| Metric (denominator) | Unconditional | Conditional (N+QED) |
|---|---|---|
| attempts | PENDING | PENDING |
| raw validity (/ attempts) | PENDING | PENDING |
| corrected = `degree` validity (/ attempts) | PENDING | PENDING |
| uniqueness (/ valid) | PENDING | PENDING |
| novelty vs train (/ unique valid) | PENDING | PENDING |
| **NUV (/ attempts)** | PENDING | PENDING |
| internal diversity | PENDING | PENDING |
| mean achieved heavy atoms | PENDING | PENDING |
| atom-count match (/ attempts) | PENDING | PENDING |

Denominators are exactly as `metrics_generation.decoder_metrics` defines them
(`metrics_generation.py:107`), every rate with a Wilson 95% CI:

- `strict_validity` = (sanitizable ∧ single fragment ∧ SMILES) / **all samples**
- `uniqueness` = distinct canonical SMILES / **strictly valid samples**
- `novelty_vs_train` = unique SMILES not in train / **unique valid SMILES**
- `nuv` = validity × uniqueness × novelty. Because the denominators chain
  (V/A · U/V · Nv/U = Nv/A), **NUV is exactly the fraction of all attempts that are valid,
  unique and novel** — which is the quantity you asked for.

### 2.3 Architecture differences that will remain

If you compare `zinc250k_flow_uncond.yaml` against `zinc250k_flow_nqed.yaml`, the two
differ in **two** things, not one:

- `condition_features`: `[]` → `["n","qed"]` (intended)
- `node_autoregressive`: **false → true** (a confound: an extra coupling per block,
  transforming every channel of node *i* given the full vectors of nodes < *i*)

That comparison therefore measures **conditioning plus a strictly larger flow**. Two
honest options:

1. Train an unconditional checkpoint with `node_autoregressive: true` (and 128-D) —
   the clean baseline. Cost: one training run.
2. Report the comparison as "conditioning + node-autoregressive coupling" and show the
   permuted-condition gap (section 3.3) as the *separate* evidence that the conditioning
   specifically is alive.

`scripts/46` computes this diff automatically and prints
`clean_conditioning_contrast: true/false` with a verdict string, so the caveat cannot be
forgotten.

---

## 3. Deliberate requests, 3×3 grid — **PARTIALLY AVAILABLE, SUPERSEDED**

### 3.1 What exists

`docs/reported_results.json` has achieved QED for the nine cells — but it is second-hand,
flagged not-recomputed, and **you say it is superseded by the deck I cannot read**. I am
therefore *not* reproducing it as the current result. For the record, its internal pattern:

| requested QED → | 0.4 | 0.6 | 0.8 |
|---|---|---|---|
| N=15 | 0.600 | 0.569 | 0.644 |
| N=23 | 0.609 | 0.541 | 0.597 |
| N=30 | 0.495 | 0.491 | 0.505 |

Read the rows: within every atom count, achieved QED is **nearly flat and sometimes
inverted** with respect to the request (N=23: 0.609 for a 0.4 request, 0.541 for a 0.6
request). If the new numbers still look like this, the defensible claim is *"QED
conditioning is measurable in likelihood but does not translate into single-draw QED
control"* — not *"the model follows both requests"*.

### 3.2 The per-cell table to fill

Nine rows, from `scripts/13_generate_nqed.py` (500 samples/cell) or `scripts/42` (50/cell):

| N | QED req | attempts | valid | atom-count enforced? | mean achieved QED | QED MAE | QED sd | valid ∧ on-target (±0.1) / attempts |
|---|---|---|---|---|---|---|---|---|
| 15/23/30 × 0.4/0.6/0.8 | | PENDING | PENDING | **yes, by construction** | PENDING | PENDING | PENDING | PENDING |

Definitions in code: `qed_evaluation.py:49 qed_target_metrics` (MAE over the **valid
molecules of the headline decoder only**, plus signed bias, RMSE, hit rate at
`DEFAULT_QED_TOLERANCE = 0.1`); `scripts/42_checkpoint_eval.py:275
valid_on_target_yield` = on-target scored samples / **all attempted samples in the cell**.

Response slopes within each atom count: `qed_achievement_table`'s
`qed_independent_of_n` is the mean of the **within-N** requested-vs-realised correlations
— the right statistic, because the ZINC N–QED association would otherwise explain a
pooled correlation on its own.

### 3.3 The evidence that conditioning is real, independent of generation

This is cheap, does not need generation, and is the strongest thing you can show:

```bash
# writes conditioning_check.json; also printed every eval in train.log
python scripts/19_conditional_signal_audit.py --flow-checkpoint <nqed best.pt> ...
```

- **permuted-condition gap** in nats/node, with the per-molecule win rate. Gap 0 with
  ties = 1.0 ⇒ conditioning ignored. Gap > 0 ⇒ used.
- **QED-only permutation** (true N kept) ⇒ isolates QED from N.
- **prior response** `mu(c), sigma(c)` across requested N ⇒ a flat curve means the
  conditional prior is dead.

`trainer_flow.validate` already logs `val_condition_gap` and `val_condition_win_rate`
**every evaluation**, so if you still have any `metrics.jsonl` from the nqed run, these
numbers already exist and need no new compute. **Check that file first.**

### 3.4 Targets with little training support

`scripts/46 --inspect-only` reports, per deliberate cell, the count of training molecules
within ±1 atom and ±0.05 QED. Expect (N=30, QED=0.8) and (N=15, QED=0.4) to be thin:
larger molecules have systematically lower QED, so the corners fight the data. Fill from
the run — do not assert it without the count.

---

## 4. Unconditional screening baseline — **PENDING**, design fixed

The honest comparison for "can you hit a joint (N, QED) request?":

- Unconditional arm: generate at a **stated** budget, then filter to the requested N and
  pick the closest QED. **Report the total attempts consumed**, because screening buys
  accuracy with compute and the conditional model must be charged the same budget.
- Count a size mismatch as a **failure** for joint success, not as a discard.
- First-draw vs best-of-ten is only comparable when candidate allocation is identical and
  invalid/wrong-size outputs are handled the same way. The existing best-of-k protocol
  (`scripts/33`) uses **disjoint groups of k** from a 100-candidate pool — that allocation
  must be mirrored exactly on the unconditional side or the comparison is void.

Existing reference point you should keep separate: `reported_results.json` gives native
MAE 0.1848 vs best-of-10 MAE 0.0327 under "100 candidates per condition, 10 disjoint
groups of 10, nearest-to-target kept". That 5.7× gap is **selection**, not conditional
control — `scripts/39_conditioning_diagnostics.py:150` already carries that caveat in
code. Label it "exploratory, selection-based" on the slide.

---

## 5. Why QED, from the data

### 5.1 QED by atom count — **computed by `scripts/46 --inspect-only`**

It reports per-N QED mean/sd/quantiles (5/50/95), plus **Spearman and Pearson** N–QED
correlation over the training split, and the per-cell grid support. The brief's
requirement not to assume independence is met by construction: requests are drawn as whole
training rows.

PENDING — needs the dataset.

### 5.2 Aromatic rings vs reference — **PENDING**

`reported_results.json`: 1.15 aromatic rings generated vs 1.88 in real ZINC. Two problems
to fix before this goes on a slide:

1. **The reference must be size-matched.** Aromatic-ring count rises with heavy-atom
   count, so comparing generated molecules (whose size distribution is whatever was
   requested) against all of ZINC confounds size with chemistry. Compare within each N
   group.
2. **"Does 128-D move it closer?"** cannot be answered from the repo: there is no 64-D vs
   128-D ring comparison in any committed artefact, and the two differ in the
   *autoencoder*, so a difference would be a decoder-capacity effect, not a flow effect.

---

## 6. Slide audit

### 6.1 The 25.1% MAE reduction — **cannot be audited, and does not appear in the repo**

```
grep -rn "25.1" docs/   →  no matches
```

The claim is not in `slides_content.json`, not in `final_results_slides.md`, not in
`reported_results.json`. It must come from the deck I cannot open. **Please paste the two
numbers it is computed from.** The arithmetic to check is which of these it is:

- a **reduction in MAE**: (before − after) / before
- a **reduction toward a floor**: (before − after) / (before − floor)

These differ a lot. The constant-predictor MAE floor recorded in an earlier session was
**0.1333** (flagged PROVISIONAL, uniform weights). Against a native MAE of 0.1848 that
floor is *lower* than the model — i.e. predicting the dataset-mean QED for every request
beat the model by ~28%. If the 25.1% is measured against a baseline that is itself worse
than the constant predictor, the claim needs reframing. **This is the highest-risk number
in the deck.** Your supervisor is likely to ask for the floor.

### 6.2 Metric denominators — code is correct and explicit

Verified in `metrics_generation.py:107-158`: validity per attempt, uniqueness per valid,
novelty per unique, NUV = product with attempt denominator, fallback rate per attempt,
QED/rings over valid only, every rate with a Wilson CI. **The definitions are sound. The
risk is slide text that reports uniqueness or QED without saying the denominator is "valid
molecules", which makes a 97% uniqueness look like 97% of attempts.** Add the denominator
to every caption.

### 6.3 Pictured molecules — selection must be stated

`molecule_grids.best_of_k_gallery` deliberately separates `selection_rate`
(kept groups / drawn) from `target_hit_yield_over_all_drawn` (on-target / drawn), with a
comment at lines 304–306 keeping them apart. If the pictured molecules came from a
best-of-k gallery, the caption must say **"best of k, k = N, selected by nearest QED"**.
A grid of selected molecules presented as typical output is the single easiest thing for an
examiner to catch.

### 6.4 The chemistry gate — defined and pre-registered, outcome **PENDING**

`scripts/42_checkpoint_eval.py:86`, fixed before any evaluation ran:

| Gate | Threshold |
|---|---|
| `macro_raw_validity_min_delta` | ≥ −0.02 vs baseline |
| `macro_atom_count_adherence_min_delta` | ≥ −0.02 vs baseline |
| `macro_valid_on_target_yield_min_delta` | ≥ 0.0 vs baseline |
| `macro_degree_validity_floor` | ≥ 0.95 absolute |
| `macro_aromatic_rings_retention` | ≥ 0.90 × baseline |
| `macro_uniqueness_retention` | ≥ 0.90 × baseline |
| `macro_internal_diversity_retention` | ≥ 0.95 × baseline |

Primary ranking metric: `macro_qed_mae` (equally weighted over the nine cells).
**The outcome requires a `scripts/42` artefact, and none exists in this workspace.** If
the deck states a gate outcome, it came from a run whose JSON is not committed — commit it.

Note also: `degree_validity_floor = 0.95` is a floor on the **MILP-assembled** validity,
which `reported_results.json` puts at 0.97–1.00. The gate is therefore comfortably passed
by a model whose **raw** validity is 0–1% at N=30. That is a real limitation of the gate,
not a reason to change it now — but do not present passing it as evidence of chemical
quality.

### 6.5 Supporting experiments — different checkpoints and budgets

| Experiment | Checkpoint | Budget | Comparable to headline? |
|---|---|---|---|
| Temperature sweep | `scripts/05`, uncond, 64-D | 10,000/temperature | **No** — different arm and embedding width |
| Depth / Arm A vs Arm B | `scripts/11`, uncond 10k vs 50k steps | file re-read only | **No** — unconditional, and 10k vs 50k steps |
| Pairwise conditioner (stage 26) | 128-D, not yet run | 3,000 steps, 50/cell | pending |
| Screening / best-of-k | `scripts/33`, 100 candidates/condition | 100/condition | **No** — selection protocol, 2× the per-cell budget of `scripts/13` |
| Experiments A/B (balanced, correlated prior) | 128-D `qedcontrast`, +3,000 steps | 50/cell selection, fresh draws for confirmation | within-family only |

Keep all of these as **supporting**, each with its own checkpoint and budget printed next
to it. None of them can stand in for the main table.

### 6.6 `docs/final_results_slides.md` has 22 unfilled placeholders

```
grep -c FILL docs/final_results_slides.md   →  22
```

Including the headline corrected-validity cell and the paired ΔQED response. The committed
deck is a template, not a result. Either fill it from committed artefacts or present the
`.pptx` you have and treat the markdown as superseded.

---

## 7. Supported conclusions, and unresolved questions

### 7.1 Supported by code inspection alone (safe to claim tomorrow)

1. The conditional model takes N and QED through a **train-only scaler** into FiLM in every
   coupling conditioner and into a conditional prior `mu(c), sigma(c)`, both
   zero-initialised so an untrained conditional model starts from the unconditional map.
2. **Atom count is structural in both arms.** N sets the node-slot count and the decoder
   emits one atom per slot, so atom-count agreement is plumbing, not conditioning — and
   the unconditional baseline receives the same information. QED is the only genuinely new
   controllable quantity.
3. The verification machinery is sound and unusually strict: permuted-condition likelihood
   gap logged every evaluation, a QED-only permutation that holds N fixed, a prior-response
   sweep, a latent round-trip self-check on the first generated batch, and a pre-registered
   chemistry gate with a fixed primary ranking metric.
4. Metric denominators are explicit and correct, and NUV is genuinely the
   novel-unique-valid fraction **of all attempts**.
5. "Corrected validity" here is MILP degree-constrained assembly that **never deletes
   atoms**, which is stricter than MoFlow-style valency correction.

### 7.2 Unresolved, and the honest wording for each

| Question | Status | What settles it |
|---|---|---|
| Does conditioning improve or cost generation quality? | **unmeasured** — no unconditional result exists | `scripts/46_uncond_vs_cond.py` |
| Does the model follow a **QED** request in a single draw? | reported numbers look flat within N; new numbers unread | the deck's per-cell table + within-N slopes |
| Is the permuted-condition gap > 0 for the headline checkpoint? | **probably already in `metrics.jsonl`** | grep `val_condition_gap` in the nqed run's log |
| Is the 25.1% claim correct? | **cannot audit** — not in the repo | the two input numbers |
| Does the comparison isolate conditioning? | **no** with current configs (`node_autoregressive` differs) | train a matched unconditional 128-D checkpoint |
| Is any result repeated across seeds? | **no** — one model per architecture | ≥2 seeds, or say "n=1" |
| Does 128-D move ring counts toward the reference? | **unmeasured**, and confounded by the autoencoder | size-matched ring comparison |

### 7.3 What to do in the next few hours, in order

1. **`grep val_condition_gap`** in the nqed run's `metrics.jsonl`. Free, already computed,
   and it is your strongest evidence that the conditioning is alive.
2. **Paste the deck's numbers** here (or commit the metric JSONs) so this pack can be
   completed with your current results rather than superseded ones.
3. Run `scripts/46_uncond_vs_cond.py --inspect-only`. It is seconds of CPU and tells you
   whether a clean unconditional baseline exists at 128-D.
4. Put section 1.3/1.4 on a slide **before** any metrics slide. Volunteering that atom
   count is structural converts your most serious vulnerability into evidence of rigour.
5. Mark every pending cell "pending", with the command that fills it. A pack with honest
   gaps is defensible; one with numbers borrowed from another architecture is not.
