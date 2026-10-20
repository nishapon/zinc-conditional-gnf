# Stage 4-6: unconditional node-feature graph normalizing flow

What this stage is: a density model over the per-atom node embeddings of the
**frozen** learned-degree autoencoder, plus sampling and scoring. The
autoencoder is never retrained, never modified and never overwritten.

## The frozen autoencoder

| | |
|---|---|
| Checkpoint | `outputs/ablation/ae_degree/seed_3/last.pt` |
| Selected on | validation only: exact 93.40%, loss 0.5409, fallback 3.80%, reconstruction validity 99.25% |
| Why `last.pt` | `scripts/03b_*` never passed `--selection-key`, so `best.pt` was selected on the *greedy* decoder's exact match, not the degree decoder's. The reported 93.40% is `result.json["final"]`, which is the last-step model. |
| Integrity | its SHA-256 is written into `config.json` and every flow checkpoint; a resume or a generation run against a different file is refused (exit 3, `--allow-ae-mismatch` to override deliberately) |

Run this once and keep the output with the results:

```
python -c "import hashlib;print(hashlib.sha256(open('outputs/ablation/ae_degree/seed_3/last.pt','rb').read()).hexdigest())"
```

## Why the flow does not simply model the embeddings as they are

Two measured properties of this latent space drive the design:

1. **It is singular.** The encoder is deterministic and permutation
   equivariant, so symmetry-equivalent atoms get bit-identical embeddings
   (99.5% of symmetry classes). An exact tie has probability zero under any
   continuous density, so maximum likelihood on the raw empirical
   distribution is ill-posed and the likelihood is unbounded. The flow
   therefore models the **smoothed** density: `noise_sigma = 0.05` times each
   dimension's training standard deviation is added to every real node during
   training. The chosen checkpoint's own degree decoder tolerates this
   (93.40% exact at σ=0, 92.20% at 0.05, 90.45% at 0.10, 76.15% at 0.20).
2. **It is low rank.** The participation ratio is about 5 of 64 dimensions.
   Hence the fixed diagonal whitening (with a `whiten_std_floor` so a
   near-constant dimension is not amplified) and a learned invertible
   per-node linear layer in every block, which can align the coupling's
   channel split with the directions that actually carry variance.

## Architecture (`zinc_gnf/flow.py`)

```
embeddings -> (+ smoothing noise, train only) -> whitening
           -> 12 x [ ActNorm -> invertible per-node linear
                     -> affine coupling (swap=False) -> affine coupling (swap=True) ]
           -> z ;   log p(x) = log p_prior(z) + log|det J|
```

* s/t networks: one trunk per coupling with two heads, masked multi-head
  self-attention (4 heads) over the real nodes, MLP hidden 256. **No bond
  features and no positions**: at sampling time bonds do not exist yet.
* ActNorm is initialized data-dependently and every coupling's output layer is
  zero-initialized, so the flow starts as the identity with `log|det J| = 0`.
  That is why `max_log_scale` is 1.0 here and not the teammates' 0.25, which
  was a workaround for a zero-initialized ActNorm that never renormalized.
* Prior: per-node i.i.d. `N(mu, sigma^2)`, learned, with a softplus floor.
  The unconditional model's prior does **not** see N, so the later
  N-conditioned model has an honest baseline; N enters generation only through
  how many node vectors are drawn.
* Conditioning is already wired in (`condition_features: ["n"]` or
  `["n", "qed"]` turns on FiLM in every coupling and makes the prior
  conditional) but is off in this stage and no conditional job exists yet.

## Autoencoder trained on 100k, flow trains on 202,111

The autoencoder saw 100,000 of the 202,111 training molecules
(`numpy.random.default_rng(3).permutation(...)[:100000]`, the seed-3 run).
The flow trains on the whole training split by default, because the 93.40%
exact reconstruction was measured on **validation** molecules the encoder had
never seen: its embeddings generalize, so the remaining ~102k training
molecules are legitimate flow training data and more data is strictly better
for density estimation. The alternative is available for a check:
`--ae-seen-only` (with `ae_subset_seed: 3`) reproduces the encoder's own
subset exactly. If the two runs' validation NLL differ noticeably, the
embeddings of unseen molecules are distributed differently and that is worth
knowing; it is not expected.

## Jobs

| Job | Partition | Resources | Walltime | What it does |
|---|---|---|---|---|
| `slurm/04_flow_smoke_gpu.sh` | STUD | 1 GPU, 4 CPU, 12 GB | 40 min | GPU numerics check (round trip + log-det vs autograd Jacobian), 300 steps on 20k molecules, 200 generated molecules scored |
| `slurm/04_flow_uncond_gpu.sh` | STUD | 1 GPU, 4 CPU, 16 GB | 6 h | full training, 50,000 steps, batch 128, resumable |
| `slurm/05_generate_uncond.sh` | CPU | 4 CPU, 16 GB | 3 h | 10,000 molecules per temperature, all metrics, then the across-seed summary |

```
cd ~/zinc-conditional-gnf && git pull origin rewrite/zinc250k-pipeline && git log --oneline -1
sbatch slurm/04_flow_smoke_gpu.sh                                      # first, always
sbatch slurm/04_flow_uncond_gpu.sh                                     # then the real run
sbatch --export=ALL,COUNT=2000,TEMPERATURES="0.6,0.7,0.85,1.0" slurm/05_generate_uncond.sh
sbatch slurm/05_generate_uncond.sh                                     # 10,000 at the chosen T
```

**The walltime is not a promise.** A checkpoint is written every 1,000 steps
and `--resume auto` continues from `last.pt` at the same step, batch position,
RNG state and smoothing-noise stream, so a walltime kill costs at most 1,000
steps. Resubmit the same command until `summary.json` reports
`completed: true`; a finished seed is skipped. Later, the five-seed protocol is
the same script: `sbatch --export=ALL,SEEDS="1 2 3 4 5" slurm/04_flow_uncond_gpu.sh`.

GPU memory: the flow is about 2M parameters and a batch of 128 molecules of up
to 38 atoms needs well under 2 GB, so 8 GB is not a constraint; `--batch-size
256` would also fit if more throughput is wanted. Training is fp32 on purpose
(no AMP): flows are sensitive to reduced precision in `exp(s)` and the
log-determinant.

## Outputs

| Item | Path |
|---|---|
| Training | `outputs/checkpoints/flow_uncond/seed_<n>/{train.log,metrics.jsonl,config.json,summary.json,best.pt,last.pt}` |
| Smoke run | `outputs/checkpoints/flow_smoke/`, `outputs/samples/flow_smoke/T*/` |
| Samples and metrics | `outputs/samples/flow_uncond/seed_<n>/T<temp>/{metrics.json,metrics.txt,samples.jsonl,valid_smiles.txt}` |
| Across seeds | `outputs/samples/flow_uncond/generation_summary.{json,txt}` |
| SLURM logs | `outputs/logs/zinc_flow_<name>_<jobid>.log` / `.err` |

## Reading the numbers

* **Three validity numbers, never mixed.** `raw` = decoder argmax with no
  chemistry; `constrained` = greedy valence-aware assembly; `degree` =
  degree-constrained integer program with a greedy fallback (the headline).
  `fallback_rate` says how often the program was infeasible and the greedy
  decoder took over.
* **Strict validity** = sanitizable **and** a single connected fragment. No
  validity-correction step is applied, unlike MoFlow's headline number.
* **Uniqueness** divides by strictly valid samples; **novelty** divides by
  unique strictly-valid SMILES and references the **training** split only
  (train+val is a secondary number). `N.U.V. = strict_validity x uniqueness x
  novelty`, printed with its formula.
* **93.40% reconstruction does not predict 93.40% generation validity.**
  Reconstruction starts from an embedding the encoder itself produced;
  generation starts from a sample of a learned density, which can land off the
  manifold the decoder was trained on. Expect generation validity to be lower,
  and read it against the σ-robustness row above.
* **Node-count agreement is structural.** The decoder emits one atom per
  sampled node vector, so a generated molecule has the requested N by
  construction. It is logged as a guard against decoding bugs and must never
  be reported as evidence of learned conditioning.
* Pick **one** temperature and report every metric at it. Low temperature buys
  validity and destroys uniqueness; quoting them from different temperatures
  would be meaningless.

## Numerical safety

| Risk | Guard |
|---|---|
| Non-finite loss | training aborts with exit 5 after saving the last finite `last.pt`; the message names `max_log_scale` and the learning rate |
| Runaway log-determinant (the teammates' pre-NaN signature) | `log_det/node` and `z_norm` are logged every `log_every` steps, and a `WARN` fires once the magnitude passes `log_det_warn_threshold` (50 by default) while the run is still finite |
| Exploding gradients | global-norm clipping at 5.0, plus 500 steps of linear LR warm-up |
| Unstable inversion | generation re-encodes the first sampled batch and records `latent_round_trip_relative_error` in `metrics.json`; above 1e-3 the log warns that the metrics describe a different map than the trained one. Measured on the full-size model: 8e-7 at initialization, 2.5e-6 after perturbation, and the analytic log-det matches an autograd Jacobian to 6e-6 |
| Non-finite samples | counted as invalid with reason `nonfinite_embedding` and reported as `nonfinite_embedding_rate`; the decoder receives a sanitized tensor so one bad sample cannot affect the rest of the batch |
| A constant embedding dimension (dead encoder unit) | accepted, not rejected: `whiten_std_floor` bounds both the whitening divisor and the smoothing-noise scale, and `floored_dimensions` is logged and stored in `config.json` |
| Walltime kill / interrupt | `SIGTERM`/`SIGINT` save `last.pt` and exit 130; a checkpoint also lands every 1,000 steps. Resume restores step, batch position, RNG **and** the smoothing-noise stream (tested: 3+3 steps equal 6 bitwise) |
| Missing MILP solver | both the CPU job's preflight and `scripts/05_generate.py` fail loudly, because without `scipy.optimize.milp` every molecule would silently fall back to greedy decoding and the "degree" column would not be the degree decoder |

## Mid-run generation check (pausing at 10,000 steps)

Validation NLL keeps improving by filling in the thin noise directions of the
latent, which says little about generation quality. So the run can be paused
once, sampled from, and then resumed, without giving up the 50,000-step target.

```
# 1. train to step 10,000 and stop there (the run stays UNFINISHED on purpose)
sbatch --export=ALL,PAUSE_AT=10000 slurm/04_flow_uncond_gpu.sh

# 2. sample from the paused checkpoint, into its OWN directory (CPU job)
sbatch --export=ALL,CHECKPOINT_NAME=last.pt,COUNT=2000,SAMPLES_ROOT=outputs/samples/flow_uncond_pause slurm/05_generate_uncond.sh

# 3. resume to 50,000 (leaving PAUSE_AT set is safe: a passed pause point is ignored)
sbatch slurm/04_flow_uncond_gpu.sh
```

How the pause stays safe:

* `--pause-at N` does **not** change `--max-steps`. `summary.json` records
  `paused: true`, `paused_at_step: N`, `completed: false`, `early_stopped:
  false`, and `max_steps: 50000`.
* The finished-seed check therefore reports `paused` and **resumes** instead of
  skipping. Only `completed` or `early_stopped` skips a seed.
* The pause step always gets a validation pass and a checkpoint write, so
  `last.pt` (and `best.pt`, if it improved) are current at the pause.
* `last.pt` carries model, optimizer (Adam moments), batcher position, torch /
  numpy RNG and the smoothing-noise stream. The learning-rate schedule is a
  pure function of the step, so there is no separate scheduler state to lose.
  Resume is bitwise identical to an uninterrupted run (tested).
* A pause point already behind the current step is ignored and logged, so
  resubmitting with `PAUSE_AT` still exported cannot stall the run.
* A pause exits 0: it is a planned stop, not a failure.
* Intermediate samples go to `SAMPLES_ROOT`, never to
  `outputs/samples/flow_uncond`, and the job prints a notice that the run is an
  intermediate check. Nothing is overwritten and the frozen autoencoder is
  untouched (read-only, SHA-256 verified as always).

**What to decide at the pause.** Compare the degree-decoder fallback rate
against the 75-80% seen in the 300-step smoke run. A large drop means the
samples are landing closer to the real-embedding manifold and the run should
continue. A fallback rate still near 75% means more steps will not fix it, and
the target should change instead (raising `noise_sigma` from 0.05 to 0.10, for
which 90.45% reconstruction was measured, or replacing the diagonal whitening
with a full ZCA) - both need approval and neither is implemented here.

## Arm A vs Arm B: the node-autoregressive coupling

**Why.** The shuffled-node control (job 332841) reproduced the flow's failure almost
exactly by keeping the identical multiset of real node embeddings and destroying only
which nodes belong together:

| | Real | Shuffled | Flow T=0.85 |
|---|---:|---:|---:|
| Mean P(no bond) | 0.9039 | 0.9983 | 0.9940 |
| Aromatic rings / molecule | 1.830 | 0.027 | 0.012 |
| Mean QED | 0.7314 | 0.2598 | 0.2376 |
| QED >= 0.7 | 65.95% | 1.45% | 0.62% |
| Degree fallback | 4.4% | 71.9% | 79.1% |

With ~23 atoms a molecule has ~253 pairs and ~24 bonds, so a calibrated bond head should
report 1 - 24/253 = 0.905 — and on real embeddings it reports 0.9039. On shuffled and
flow embeddings it reports 0.994-0.998, i.e. about half a predicted bond per molecule.
Meanwhile the flow's pooled participation ratio already matches the real one (6.27 vs
6.14). The 300-step flow is therefore, to measurement precision, i.i.d. draws from the
correct per-node marginal: the marginal is learned, the joint is not.

**Arm B** adds exactly one mechanism per block:

```
ActNorm -> invertible per-node linear
        -> channel coupling (swap=False)     unchanged
        -> channel coupling (swap=True)      unchanged
        -> node-autoregressive coupling      NEW
```

`y_i = x_i * exp(s_i) + t_i` with `(s_i, t_i) = g(x_1 .. x_{i-1})`: every channel of node
*i* is transformed given the FULL vectors of the preceding nodes. The Jacobian is block
lower triangular, so `log|det J|` is the masked sum of `s` — exact, like the channel
couplings. Density evaluation is one parallel pass; inversion is N sequential passes.
Alternate blocks reverse the real-node order so information travels both ways.

Why this rather than whitening, prior variance, sigma or depth: the flow's per-node
marginal geometry is **already** correct (participation ratio 6.27 vs 6.14) and the prior
is healthy (sigma 1.000 -> 0.997, z-norm 7.8), so those levers address what already
works. An autoregressive conditioner can *copy* an already-placed vector, which is what
symmetry-equivalent atoms need: after smoothing they must sit about
`sigma*sqrt(2*64) = 0.57` whitened units apart while unrelated atoms sit about 11.3
apart — a 20x concentration, not an infinite one.

**Measured cost** (full-size config, CPU, 4 threads; not an estimate):

| | Arm A | Arm B |
|---|---:|---:|
| Parameters | 1,535,360 | 2,573,696 (**+67.6%**) |
| Training step (batch 128 x 38) | 0.359 s | 0.424 s (x1.18) |
| Sampling 128 molecules | 0.09 s | 1.78 s (x19.3) |
| 10,000 samples (flow part only) | ~0.1 min | **~2.3 min** |

Sampling stays well inside the CPU generation job's limit, where the degree-constrained
MILP (~10 ms/molecule, ~2 min per 10,000) dominates anyway. **Arm B has 67.6% more
parameters**, so a positive result is "mechanism + capacity"; if the comparison is
ambiguous the follow-up is a capacity-matched arm (baseline with widened channel
couplings), not a guess.

### Running the comparison

```
cd ~/zinc-conditional-gnf && git pull origin rewrite/zinc250k-pipeline

sbatch --export=ALL,PAUSE_AT=10000 slurm/04_flow_uncond_gpu.sh          # arm A
sbatch --export=ALL,ARM=b,PAUSE_AT=10000 slurm/04_flow_uncond_gpu.sh    # arm B

# mid-run samples go to their OWN roots, never the final evaluation directories
sbatch --export=ALL,CHECKPOINT_NAME=last.pt,COUNT=2000,SAMPLES_ROOT=outputs/samples/flow_uncond_pause slurm/05_generate_uncond.sh
sbatch --export=ALL,ARM=b,CHECKPOINT_NAME=last.pt,COUNT=2000,SAMPLES_ROOT=outputs/samples/flow_uncond_ar_pause slurm/05_generate_uncond.sh
sbatch slurm/08_embedding_populations.sh    # reference populations + both arms' embeddings
```

Arm A writes `outputs/checkpoints/flow_uncond/` and `outputs/samples/flow_uncond/`; Arm B
writes `*_ar/`. Neither touches the frozen autoencoder, the smoke run or any diagnostic
output. Both pause at 10,000 steps with `completed: false`, so nothing continues to
50,000 without a decision.

### Success criteria at the 10,000-step pause

Arm B at initialisation *is* Arm A (zero-initialised heads), the target space, sigma,
whitening and prior are identical, so **validation NLL is directly comparable between
the arms** — but it is a secondary metric. The primary endpoints are generation-side.

| Metric | Real | Shuffled / Flow now | Arm B target |
|---|---:|---:|---:|
| **Mean P(no bond)**, pre-assembly | 0.9039 | 0.998 / 0.994 | **<= 0.93** |
| Expected bonds per atom | ~2.1 | ~0.04 | >= 1.5 |
| Raw strict validity (argmax, no chemistry) | — | 0% | > 5% |
| Aromatic rings / molecule | 1.830 | 0.027 / 0.012 | **>= 1.0** |
| Molecules with an aromatic ring | 93.4% | 2.5% | >= 60% |
| Aromatic / all rings | **0.631** | ~0.02 | >= 0.45 |
| Mean QED | 0.7314 | 0.2598 / 0.2376 | **>= 0.60** |
| **Fraction QED >= 0.7** | 65.95% | 1.45% / 0.62% | **>= 30%** |
| Degree-decoder fallback | 4.4% | 71.9% / 79.1% | **<= 25%** |
| Degree-decoder strict validity | — | — | >= 90% |
| Uniqueness / novelty (10,000 samples) | — | — | >= 95% / >= 95% |
| Internal diversity, scaffolds | ~0.9 | — | no collapse |
| Secondary | — | — | val NLL, `log_det - sum log sigma`, within/between distance ratio and near-identical-pair rate against **real_smoothed** |

The aromatic/all-rings reference is **0.631** as measured on our validation split, not
the ~0.98 quoted in an earlier discussion.

Decision rule: Arm B wins only if P(no bond), aromatic rings and the QED >= 0.7 fraction
all improve materially at equal step count. A better NLL alone does not count. If Arm A
also clears the gates, the architecture was adequate and only training time was missing.

---

## Stage 10 — latent recalibration of a trained flow (new)

**What it fixes.** A flow is sampled from its prior, but training only *pushes* the
latents towards that prior; nothing guarantees they got there. If `z = f(x)` is wider
(or narrower, or off-centre) than the prior per dimension, every sample starts in the
wrong region of latent space, `f^-1` is evaluated where it was never fitted, and the
decoded graphs degrade even though the flow itself is fine.

This is not speculation about our pipeline being affected — it is a *measured* failure
in the team's QM9 work (`molecular_generation/inspect_latent_calibration.py`,
`latent_stats.py` in the teammates' repository): the learned `sigma(N)` underestimated
the empirical latent spread by about **1.5x**, and widening sigma at sampling time
alone recovered uniqueness at N=5. The team's ZINC conditional run compensated the
same way, with sampling temperature **0.2**. Stage 10 measures the ZINC analogue of
that diagnostic for our own checkpoints and, instead of a sampling hack, folds it into
the model.

**Why it is a modelling improvement, not postprocessing.** For a diagonal Gaussian
prior family the maximum-likelihood parameters of the observed latents are exactly
their empirical per-dimension mean and std, and the trained prior is a member of that
family. So on the measurement data the calibrated model's likelihood is **>=** the
trained model's, with equality precisely when the flow was already calibrated — a
no-op for a converged flow. `NodePrior.set_calibration` is read by **both** `log_prob`
and `sample`, so the model stays a normalised density that can be scored; a
generation-time `sigma_scale` would not be. It costs no retraining and applies to the
checkpoints we already have, including Arm B's paused 10,000-step one.

**What it cannot do.** It matches per-dimension location and spread only. It does not
teach the flow inter-node dependence (the shuffled-node control's finding), and it
cannot fix heavy latent tails — the report prints the kurtosis so that is visible. On
the teammates' own ZINC run, calibration plus `T=0.2` still left atom composition and
aromaticity unmatched, so expect validity/QED to move before aromaticity does.
**A gain of 0.0 is a legitimate, informative outcome**: it rules the prior mismatch
out as a cause and points back at the coupling architecture or the training budget.

### Run order

```
cd ~/zinc-conditional-gnf && git pull origin rewrite/zinc250k-pipeline

# 1. measure + write a calibrated copy of arm B's paused checkpoint (CPU, minutes)
sbatch --export=ALL,FLOW_CHECKPOINT=outputs/checkpoints/flow_uncond_ar/seed_1/last.pt \
       slurm/10_calibrate_flow.sh
#    -> outputs/checkpoints/flow_uncond_ar/seed_1/last_calibrated.pt
#       outputs/checkpoints/flow_uncond_ar/seed_1/calibration.{json,txt}
#    read calibration.txt FIRST: zscore_rms is the headline (1.0 = already calibrated)

# 2. sample the calibrated checkpoint over a temperature sweep incl. the team's T=0.2
sbatch --export=ALL,ARM=b,CHECKPOINT_NAME=last_calibrated.pt,COUNT=2000,TEMPERATURES=0.2,0.5,0.7,0.85,1.0,SAMPLES_ROOT=outputs/samples/flow_uncond_ar_calibrated \
       slurm/05_generate_uncond.sh

# 3. the SAME sweep on the UNCALIBRATED checkpoint, so temperature and calibration
#    are not confounded (the existing 10k samples are T=0.85 only)
sbatch --export=ALL,ARM=b,CHECKPOINT_NAME=last.pt,COUNT=2000,TEMPERATURES=0.2,0.5,0.7,0.85,1.0,SAMPLES_ROOT=outputs/samples/flow_uncond_ar_pause_sweep \
       slurm/05_generate_uncond.sh

# 4. in parallel on the GPU: finish arm B's remaining 40,000 steps, since the 10k
#    numbers may simply be an under-trained flow
sbatch --export=ALL,ARM=b slurm/04_flow_uncond_gpu.sh
```

Step 3 is what makes the comparison honest: calibration and temperature both change the
latent spread, so the calibrated run must be read against an uncalibrated run at the
same temperatures, at the same step, from the same checkpoint, with the same decoders.

Compare against the recorded Arm B baseline (10,000 steps, `T=0.85`, 2,000 samples):
mean QED 0.4192, aromatic rings/molecule 0.3891, raw strict validity 1.55%,
degree-decoded validity 98.95%, decoder fallback 64.8%, internal diversity 0.9104.
The gates in the table above are unchanged — calibration has to clear the same bar.
