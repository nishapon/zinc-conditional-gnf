# Reference protocols, our QED path, and two number audits

Companion to `docs/supervisor_meeting_pack.md`. Everything here was **verified from
primary sources during this session** — paper text, official code, or our own code with
line numbers. Each claim carries a status:

- **verified** — read in the source named
- **reported but unverified** — stated by a person or a non-recomputed file; I could not check it
- **pending** — needs data or compute that does not exist in this workspace

---

## 0. Workspace status (§1)

Re-checked, not assumed. **Nothing has changed since the last session.**

| | Status |
|---|---|
| `outputs/**` (checkpoints, metrics, samples, logs, figures) | **0 files** beyond `.gitkeep`; `outputs/*` gitignored |
| `data/` | **absent** |
| torch / numpy / rdkit / scipy / pyyaml / matplotlib / pandas | **all missing** |
| hardware | 8 CPU cores, 87 GB free, Python 3.11.15, no GPU |
| git | clean, `fsck` clean, HEAD `60aca30` — no repair or re-clone needed |

So every generation, QED-accuracy, chemistry and baseline number remains **pending**. I
did not re-run the earlier audit; I spent the session on things that need no data:
reference verification, code tracing, and two arithmetic audits.

### The precise file list I need

| What | Expected path | Needed for |
|---|---|---|
| dataset + train-only scaler | `data/processed/zinc250k_charge_aware_moflow_fixed/splits.pkl`, `.../scaler.json` | every generation run, QED–N joint support, percentile targets |
| conditional flow checkpoint | `outputs/checkpoints/flow_nqed_ae128_qedcontrast/seed_3/best.pt` | the conditional arm |
| **matched unconditional checkpoint** | does not exist — see §4 | the baseline arm |
| autoencoder | `outputs/ablation/ae_degree_128d/seed_3/last.pt` | decoder for both arms |
| training log of the conditional run | `outputs/checkpoints/flow_nqed_ae128_qedcontrast/seed_3/metrics.jsonl` | `val_condition_gap`, already computed every eval |
| the deck's numbers | — | replacing the superseded `reported_results.json` |

A checkpoint carries its own `flow_config`, `train_config`, AE sha256 and data
fingerprint, so **settings will be recovered from checkpoint metadata, never inferred
from the 64-D YAMLs.** `scripts/46_uncond_vs_cond.py --inspect-only` prints exactly that
and refuses to proceed on a mismatch.

---

## 1. MoFlow's actual generation protocol (§5) — **verified**

Read from the paper via ar5iv (§5.1 Setup) and from `mflow/generate.py` +
`mflow/models/utils.py` in the official repository.

### What the paper says

| Item | Value | Source |
|---|---|---|
| prior | `N(0, (tσ)² I)`, σ **learned**, t = temperature | paper §5.1 |
| temperature | **t = 0.85** for QM9 and ZINC250K generation; **t = 0.6** for the ablation *without* validity correction | paper §5.1 |
| attempts per run | **10,000** latent vectors decoded | paper §5.1 |
| runs | **5**, reporting **mean and standard deviation** | paper §5.1 |

**Your four stated facts are all confirmed.** Two corrections to how they must be
implemented:

1. **Neither code default is 0.85.** `generate.py`'s CLI default is `--temperature 1.0`,
   and the `generate_mols` function signature defaults to `temp=0.7`. 0.85 must be passed
   explicitly.
2. **"10,000 attempts" is the batch size.** `generate_mols(model, batch_size=batch_size,
   ...)` draws exactly `batch_size` latent vectors per run, and `--batch-size` defaults to
   **100**. Reproducing the paper means `--batch-size 10000 --n_experiments 5
   --temperature 0.85`.

### Repeated sampling, not independent training — **verified**

`generate.py`:

```python
for i in range(args.n_experiments):
    adj, x = generate_mols(model, batch_size=batch_size, temp=args.temperature, ...)
```

The model is loaded **once** before the loop. `--n_experiments` defaults to **1**. So the
5 runs are **five sampling repetitions from one trained checkpoint**, and the reported
±sd is *sampling* variability only — it contains **no** training-seed variability. Any
table of ours that puts our ±sd next to theirs must say which kind it is.

### Official denominators — **verified, and they match what you specified**

From `models/utils.py`:

| Metric | Code | Denominator |
|---|---|---|
| Validity | `valid_ratio = len(valid)/n_mols` | **attempted** |
| Uniqueness | `unique_ratio = len(unique_smiles)/len(valid)` | **valid** |
| Novelty | `novel_ratio = novel/len(gen_smiles)` where `gen_smiles` is the *unique* set | **unique valid** |
| N.U.V. | `abs_novel_ratio = novel/n_generated_mols` | **attempted** |

Arithmetic check against the paper's own Table 3 row for GraphNVP: validity 42.6,
uniqueness 94.8, novelty 100 → 0.426 × 0.948 × 1.00 = **0.4038 = the reported N.U.V.
40.38**. The chain is confirmed, and `abs_novel_ratio` computes it directly.

One wording trap: the paper's prose defines Uniqueness as "unique valid molecules in all
the generated molecules", which reads like an attempted denominator. The code and the
table arithmetic both say **per valid**. Use the code.

### Correction rules — **verified, and materially different from ours**

MoFlow with `--correct_validity true` (**the default**) does three things in order:

1. `construct_mol` — argmax atoms and bonds; while adding bonds, if one atom overflows its
   valence by exactly 1 and is N, O or S, it **sets formal charge +1**.
2. `correct_mol` — loop while invalid: take the offending atom's **highest-order** bond and
   **reduce its order by one** (removing it at order 0). Atoms are never deleted.
3. `valid_mol_can_with_seg(..., largest_connected_comp=True)` — if the SMILES contains
   `.`, split and **keep the largest fragment**, chosen by **SMILES string length**
   (`len(s)`), not by atom count.

Consequences for us:

- **Step 3 changes the atom count.** A corrected molecule can have fewer atoms than
  requested. So if we ever report a MoFlow-compatible number, achieved size **and QED must
  be recomputed after correction** — which is exactly what you asked for.
- **Our MILP assembly is not this.** Ours requires a single fragment and
  **deletes nothing**: a disconnected assembly is counted invalid, not repaired
  (`generation.py:109`). Ours is *stricter*. Keep three separate columns: `raw`,
  `native-corrected (MILP)`, `MoFlow-compatible (bond reduction + largest fragment)`.
- Stereochemistry: metrics use `MolToSmiles(..., isomericSmiles=False)` while the training
  reference is built with `isomericSmiles=True`. In practice this is a no-op, because the
  kekulized graph representation carries no stereocentres, so both strings coincide. Not a
  bug — but worth knowing before copying the flags.
- **Novelty reference**: built by round-tripping the dataset's own tensors through
  `construct_mol` (`train_smiles = adj_to_smiles(train_adj, train_x, ...)`), i.e. the
  training split canonicalised **through the same code path as the generated molecules**.
  Ours uses the stored SMILES strings. Same intent, different route; if a ZINC SMILES and
  its graph round-trip disagree, novelty shifts. Worth one spot check.

### Temperature is not transferable — **state this explicitly**

Their `t` multiplies a **learned scalar** σ on a flat latent of size 369 for a
bonds-and-atoms tensor. Our temperature multiplies a **conditional, per-dimension**
`sigma(c)` on per-node embeddings of a frozen autoencoder. **The same number 0.85 does not
describe the same distribution.** Report t = 0.85 as a *comparison setting*, keep our
existing temperature-0.2 results under their own label, and never present matched
temperatures as matched sampling.

### What MoFlow does **not** provide

Its QED results (paper §5.3, Table 4) come from **gradient ascent in latent space on a
trained property regressor**, reporting the **best** scores found (ZINC's own best is
0.948). That is property *discovery*, not target matching. **MoFlow has no target-QED MAE,
no hit rate and no slope.** It cannot be cited as a benchmark for our conditional accuracy.

### If we run it, the honest label

Not "exactly MoFlow". The differences that will remain: different dataset representation
(embedding density vs one-shot graph tensors), different decoder (frozen autoencoder +
assembler vs direct argmax), different correction, a conditional per-dimension prior vs a
learned scalar σ, and 17 charge-aware atom types vs 9. List them next to the table.

---

## 2. The teammate work (§2) — **partly verified**

| Claim | Status | Evidence |
|---|---|---|
| `property_conditioning.py` combines `log(N+1)` with **one standardized scalar** property | **verified** | `NPropertyEmbedding._build`: `n_feat = log(n_node+1)`, `property_feat = (raw - mean)/std`, `concat` → 2-layer MLP |
| it supplies one numerical target repeatedly with different latent noise | **verified** | `run_logp_range.sh` passes `--target_properties` and `--num_molecules_to_generate`; `--random_seed` fixed per run |
| `run_logp_range.sh` sweeps **13 targets, −3…3, 500 attempts each**, no interval bounds | **verified** | `TARGETS="-3,-2.5,…,3"`, `NUM_PER_TARGET=500`, `SCALE=1.0`, `SEED=12345` |
| slides 25–27: five logP targets at training percentiles, 2,000 attempts, five seeds, ±0.25 and ±0.5 | **reported but unverified** | the deck is not fetchable — see below |
| slide 28: the 13-target sweep, one seed | **reported but unverified** | same |
| "Plain A" = reference without logP conditioning | **reported but unverified** | same |
| their QED work is a diagnostic, not a QED-conditioned training experiment | **consistent with the repo** | the only property-conditioned training script is logP (`run_molecular_flow_logp.sh`, `PROP_MEAN=0.3345 PROP_STD=0.9459`); `check_qed.py` is a standalone script |

**I could not open the slides.** The editor URL and the `/export/pdf` URL both redirect to
`googleusercontent.com` (a different origin, which I must not follow blindly), the signed
URL returns HTTP 400, and the fetch tool cannot read `application/pdf` at all. Please
paste slides 25–28 as text if those numbers are to be cited.

Three differences that stop their numbers from being our baseline:

1. **logP on QM9**, not QED on ZINC250K. Different property, different dataset, molecules
   of ~9 heavy atoms rather than 15–38.
2. **TensorFlow/Sonnet**, not PyTorch — a different codebase, so "same method" claims need
   care.
3. Their `sigma_scale` default is **1.0**; ours is **0.2**. Their hit tolerances are
   ±0.25/±0.5 **on logP**, whose training std is 0.9459 — i.e. roughly ±0.26σ and ±0.53σ.
   Our ±0.1 on QED is a *much* tighter ask relative to QED's spread. **Tolerances must be
   compared in standard-deviation units, not raw units.**

One procedure of theirs worth copying, one worth avoiding:
- **Copy**: the requested-vs-achieved table with `within 0.25` / `within 0.5` columns and a
  slope, computed per target, with the valid count shown.
- **Avoid**: their slope is `polyfit` on the **per-target means** (`run_logp_range.sh`).
  A slope of means hides within-target spread entirely. Report a per-molecule regression
  as well, with uncertainty.

---

## 3. Our QED path, traced (§3) — **verified**

| Question | Answer | Evidence |
|---|---|---|
| Are we supplying a **scalar** QED target? | **Yes.** One float per request. | `conditioning.standardise_condition(features, scaler, n, qed)` → `[B, 2]` |
| Is any **CDF / quantile / binning / interval** used? | **No.** | `ConditionScaler` (`preprocessing.py:59`) is plain `(x − mean)/std`, std floored at 1e-8. A repo-wide grep for `quantile|percentile|cdf|digitize` in the condition path finds **one** hit, a diagnostic at `preprocessing.py:344`. `qed_bin_width` exists **only** in `balanced_sampler.py` and `experiment_arms.py` — the balanced sampler's cells, never the conditioning signal. |
| Does inference use the **same** normalization as training? | **Yes, provably.** | Training reads the pre-standardised column from the dataset (`batch_condition`, `conditioning.py:78`). Inference loads `<data_dir>/scaler.json` and `load_condition_scaler` **refuses** it unless `fit_on == "train_only"` and the column names match. |
| Does the condition reach every intended component? | **Yes — three families.** | FiLM in affine couplings (`flow.py:308,344`); FiLM in node-autoregressive couplings (`flow.py:484,509`); the conditional prior (`flow.py:736`: `relu(self.hidden(condition))` → `mu_head`, `sigma_head`). All zero-initialised, so an untrained conditional model equals the unconditional map. |
| Is the prior a distribution over **latent embeddings**, not over QED? | **Yes.** | `parameters_for` returns `mu, sigma` of width `node_embedding_dim`; the prior is `p(z | c)`. It never models `p(QED)`. This is the standard conditional-flow construction of Winkler et al., *Learning Likelihoods with Conditional Normalizing Flows* (arXiv:1912.00042): a base density whose mapping to output space is conditioned on an input. |

### Target vs acceptance interval — the distinction you asked for

| Term | Meaning | Where it lives |
|---|---|---|
| **Requested target** | the scalar we condition on, e.g. 0.4 | model input |
| **Acceptance interval** | the evaluator's tolerance, e.g. [0.39, 0.41] | `qed_evaluation.DEFAULT_QED_TOLERANCE = 0.1` |

**The acceptance interval is never supplied to the model.** It is a post-hoc scoring
window. Conditioning on 0.4 while varying latent noise is a valid conditional-generation
procedure and does **not** mean "QED ≤ 0.4". Scalar conditioning is retained.

I will report **±0.01 (tight)** alongside **±0.1 (existing)**, both labelled as
*evaluator tolerances*. For calibration: QED's training std is roughly 0.1–0.2, so ±0.1 is
about ±0.5–1σ and ±0.01 is about ±0.05–0.1σ — a very demanding window that will likely
read near zero. That is informative, not a failure of the metric.

A genuine interval-conditioned model (conditioning on `[lo, hi]`) would need a different
input representation and training labels, and is **out of scope**.

---

## 4. Structural vs learned conditioning (§4) — **verified**

Three different things, which the slides must not merge:

| Mechanism | Present in the unconditional arm? | Evidence |
|---|---|---|
| **Structural control of output size** — N sets how many node vectors are drawn; the decoder emits exactly one atom per vector | **YES** | `generate_molecules(..., node_counts: NodeCountSampler, ...)` is a **required** argument; sizes come from the training histogram or `target_n` (`generation.py:275`). `node_count_agreement` says in code: *"structural, not learned"* |
| **Learned N dependence** via FiLM + `mu(N), sigma(N)` | no (`condition_features: []`) | `flow.py:736` |
| **Learned QED dependence** via the same pathways | no | — |

So exact atom-count accuracy is **not** evidence that FiLM works, and I will not present
it as such. The evidence for learned conditioning is the permuted-condition likelihood gap
(§7 below), not size agreement.

### The matched baseline does not exist yet

The main QED comparison needs **same architecture, embedding width, coupling depth,
decoder, split, budget and generation settings — differing only in QED conditioning.**

Candidates, and why each falls short:

| Candidate | Problem |
|---|---|
| `zinc250k_flow_uncond.yaml` | 64-D **and** `node_autoregressive: false` vs the conditional config's `true`. Changes conditioning **and** architecture. |
| an N-only 128-D checkpoint | would be the *ideal* baseline (holds learned N conditioning fixed, removes only QED) — **not known to exist** |
| `flow_nqed_ae128_qedcontrast` | the conditional arm itself |

**Minimum additional run**, if a matched baseline must be trained: one flow with
`condition_features: ["n"]`, everything else copied from the conditional checkpoint's own
stored `flow_config`, same AE, same split, same step budget and seed. Per the nqed config
that is **50,000 steps at batch 128** — the same cost as the original conditional run. If
that is too expensive, the fallback is the **+3,000-step screening budget** already used
by experiments A/B/C, warm-started from the conditional checkpoint with QED conditioning
removed — much cheaper, but then it is a *fine-tuned* ablation and must be labelled as one,
not as an independently trained baseline.

Decide this before launching anything. I have not launched anything.

---

## 5. Audit: the 0.1333 "constant-predictor floor" (§7) — **recomputed**

### Where the number comes from

A generator that **ignores the request and always emits QED = c**, scored by macro-MAE
over equally weighted targets {0.4, 0.6, 0.8}:

| c | macro MAE |
|---|---|
| 0.40 | 0.200000 |
| 0.50 | 0.166667 |
| **0.60** | **0.133333** |
| 0.70 | 0.166667 |
| 0.80 | 0.200000 |

Minimum at **c = 0.600, MAE = 0.133333**. So 0.1333 is exactly the macro-MAE of
"always answer 0.6".

### Your correction is right, and here is exactly when it bites

The MAE-optimal constant is the **median** under the evaluation weighting
(`argmin_c E|Y − c|` is a median, not a mean). For this grid the median and mean of
{0.4, 0.6, 0.8} **both equal 0.6**, so the earlier claim was numerically right *for this
symmetric, equally weighted grid* — which is why it went unnoticed. It breaks as soon as
the weighting is uneven, and `qed_achievement_table` already reports a `mae_weighted`
weighted by per-cell **valid counts**, which are very uneven because validity collapses
with N:

| weighting over (0.4, 0.6, 0.8) | optimal constant | MAE | using the *mean* instead |
|---|---|---|---|
| equal (1/3 each) | 0.600 (median = mean) | 0.133333 | same |
| (0.60, 0.30, 0.10) | **0.400** (weighted median) | **0.100000** | 0.500 → 0.120000, **strictly worse** |

So: **recompute the reference with the same weights as the headline metric**, and use the
weighted median. Never reuse 0.1333 for a weighted or asymmetric target set.

### Three further corrections to how it was used

1. **It is not a lower bound.** A perfect model scores **0.0**. 0.1333 is the score of one
   trivial strategy. Calling it a "floor" invites the reading "nothing can do better",
   which is false.
2. **It is not a molecular generator.** A constant score is a scoring reference; it emits
   no molecules, has no validity, no uniqueness and no novelty. It cannot appear in a
   results table as a model row.
3. **It assumes zero variance.** A real constant-*mean* generator has spread, and spread
   raises per-sample MAE:

| achieved QED distribution | macro per-sample MAE |
|---|---|
| exactly 0.60 | 0.133333 |
| N(0.60, 0.05) | 0.146632 |
| N(0.60, 0.10) | 0.161062 |
| N(0.60, 0.15) | 0.181707 |
| N(0.60, 0.20) | 0.208743 |

### What this means for the three MAE figures we have

| Figure | Source | vs 0.1333 (zero-variance) |
|---|---|---|
| 0.1848 | `reported_results.json`, **unverified and superseded** | worse by 0.0515 |
| **0.1922** | your message, matched control | worse by 0.0589 |
| **0.1885** | your message, frozen correlated prior | worse by 0.0552 |

All three are **beaten by "always answer 0.6"** on this grid. But all three sit between
the σ = 0.15 and σ = 0.20 constant-generator rows — so the fair statement is: *"single-draw
QED error is currently in the range a request-ignoring generator with realistic spread
would produce."* That is the sentence to put on the slide. It is survivable, and it is far
better than having it derived in the room.

**Required before this goes in the pack:** the achieved-QED **standard deviation** per
cell, so the right comparison row is chosen rather than guessed. It is pending with the
rest of the generation numbers.

---

## 6. The 25.1% claim (§9) — **still cannot be audited**

Re-checked this session: `grep -rn "25.1" docs/` → **no matches**. It is in neither
`slides_content.json`, `final_results_slides.md`, nor `reported_results.json`. I need the
**two input values**. The two plausible formulas give very different answers, e.g. from
0.1922 to 0.1440:

- plain reduction `(before − after)/before` = 25.1% → after ≈ 0.1440
- reduction toward the 0.1333 reference `(before − after)/(before − 0.1333)` = 25.1% →
  after ≈ 0.1774

The second is only meaningful if 0.1333 is the right reference, which §5 shows depends on
the weighting. **Send the two numbers and the weighting and I will check it in one line.**

---

## 7. QED in plain language (§8) — **verified against the original**

QED is the **quantitative estimate of drug-likeness**, from Bickerton, Paolini, Besnard,
Muresan & Hopkins, *Quantifying the chemical beauty of drugs*, **Nature Chemistry 4,
90–98 (2012)** ([PubMed 22270643](https://pubmed.ncbi.nlm.nih.gov/22270643/),
[open PDF](https://shaker.umh.es/drugbank/PDF/QED-nchem.1243.2012.pdf)).
*Content paraphrased for licensing compliance.*

In one sentence for a supervisor: **QED scores how "drug-like" a molecule's basic physical
properties are, on a 0–1 scale, by combining eight simple descriptors through desirability
curves fitted to 771 marketed oral drugs.**

The eight components:

| | | |
|---|---|---|
| molecular weight | ALOGP (lipophilicity) | hydrogen-bond acceptors |
| hydrogen-bond donors | polar surface area | rotatable bonds |
| **aromatic rings** | structural alerts | |

Three consequences that matter for our claims:

1. **QED is not independent of molecular size.** Molecular weight is one of the eight
   inputs, and weight tracks heavy-atom count closely. Any statement of the form "we
   control QED independently of N" is wrong by construction. This is also why the
   per-request joint support must be measured (§6 of the main pack) rather than assumed.
2. **Aromatic ring count is itself one of the eight inputs.** So a QED-conditioned model is
   *directly* pressured on aromatic rings, and an aromatic-ring comparison is **not** an
   independent chemistry check of QED conditioning — it is partly a check of one of QED's
   own terms. Report it, but do not present it as independent corroboration.
3. **What QED is not:** not a measure of binding affinity, not a prediction of efficacy or
   safety, and not proof that a molecule is a useful drug. It is a property-distribution
   heuristic. There is likewise **no universal ideal aromatic-ring count** — the only
   defensible comparison is against real ZINC molecules, **size-matched** and, where
   support allows, within comparable QED ranges.

---

## 8. What I did not do, deliberately

- No new clone, no repository repair, no destructive command. Git is clean at `60aca30`.
- No training job, no sweep, no cluster submission.
- No number invented, and **no older value substituted for a current one** — the
  superseded `reported_results.json` figures appear only where explicitly labelled.
- No change from scalar to interval conditioning.
- No claim that any of our evaluation is "exactly MoFlow".
