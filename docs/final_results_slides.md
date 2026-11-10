---
marp: true
title: Conditional Graph Normalizing Flows for N+QED-Controlled Molecular Generation
author: Nisha Ponnambalam
paginate: true
---

<!--
HOW TO USE THIS FILE
--------------------
It is Marp markdown: it renders as slides (VS Code "Marp for VS Code", or
`marp docs/final_results_slides.md --pdf`) and reads normally on GitHub, so Prof. Kaur
can comment on it directly in the browser without installing anything.

Every number still missing is written as  [[FILL: <file> -> <field>]]  with the exact
artefact path, so filling it in is lookup, not recall. Nothing is invented.

Numbers already present are traceable: each table has a Source line.
-->

# Conditional Graph Normalizing Flows for N+QED-Controlled Molecular Generation

ZINC250K · joint (heavy-atom count, QED) conditioning

Nisha Ponnambalam · supervisor: Prof. Simran Kaur

**Status: all experiments complete. This deck is for discussion of results.**

---

## The one slide, if you only read one

**What works.** The autoencoder reconstructs real molecules well. The conditional flow
*does* learn a condition-dependent density — by a large and growing margin.

**What does not.** That conditional signal does not survive decoding into controlled QED.
Asking for QED 0.4 vs 0.8 at fixed molecule size changes the decoded molecules very
little.

**What we did about it.** Four separate interventions, each with a pass/fail criterion
fixed *before* seeing the data. All four failed their criterion.

**What that gives us.** A decomposition of *where* conditional control is lost in a
flow-plus-autoencoder pipeline, with each candidate repair falsified at the cheapest
level that could refute it.

---

## Pipeline, and where the failure sits

```
SMILES ──encode──> node embeddings ──flow f(x,c)──> latent z
                         ▲                              │
                         │                         sample z|c
                   frozen decoder                        │
                         │                         flow f⁻¹(z,c)
                         ▼                              │
  molecule <──argmax + valence-constrained assembly──────┘
                  ↑
         THE CONDITIONAL SIGNAL IS LOST AROUND HERE
```

* Conditioning enters through FiLM blocks and the latent prior.
* The decoder is **frozen** and was trained only for reconstruction.
* The final discrete step (argmax, then valence/degree-constrained assembly) is
  **not differentiable** and was never trained jointly with the flow.

---

## Baselines: what to compare us against

*Prof. Kaur's question. This is the calibration slide.*

| Reference | Validity | QED control reported? | Notes |
|---|---|---|---|
| Real ZINC250K | 100% | — | mean QED ≈ 0.73 |
| GraphNVP | 42.6% | no | ZINC250K |
| MoFlow | 50.3% | no | ZINC250K |
| GraphAF | 68.0% | no | ZINC250K |
| GraphDF | **89.0%** | no | ZINC250K, best of the four |
| Teammates (QM9) | [[FILL: teammates' final slide number]] | no | different dataset, not comparable directly |
| **Ours (corrected)** | [[FILL: outputs/metrics/relabel_evaluation/relabel_evaluation.json -> models.final.by_n.*.cells.*.degree.validity]] | **yes, and it is weak** | ZINC250K |

Source: GraphDF paper, Table 1 (ZINC250K rows).

> **Important caveat for the discussion.** Published validity numbers are not measured
> identically across papers, and most use a correction/sanitisation step. Our corrected
> column is the comparable one. Section "Decoder dependence" shows what correction is
> doing for us.

---

## Nobody above reports conditional QED control

Closest published comparisons for *property control*, not validity:

| Work | Control metric | Value |
|---|---|---|
| DriftingMol | Spearman ρ (requested vs achieved) | 0.493 – 0.515 |
| DriftingMol | slope / MAE | 0.796 / 0.200 |
| DriftingMol | ρ with gradient blocked (ablation) | 0.011 |
| Reward Transport | ρ group-level vs per-molecule | 1.000 vs 0.22 |
| **Ours** | paired ΔQED at fixed N (0.4 → 0.8) | [[FILL: definitive_table.md]] |

**Takeaway for discussion:** even the papers that *target* property control report
per-molecule correlations around 0.5, and one reports per-molecule ρ = 0.22 against a
perfect group-level score. Per-molecule conditional control is weak across the
literature. That is the honest frame for our result.

---

## Result 1 — Autoencoder reconstruction (this part works)

| Metric | Value | Source |
|---|---|---|
| Reconstruction accuracy (real molecules) | [[FILL: reconstruction_analysis.json]] | `scripts/03` |
| Atom-type accuracy | [[FILL]] | " |
| Bond accuracy | [[FILL]] | " |
| Chemistry audit: valence/aromaticity | [[FILL: ae_chemistry_audit.json]] | `scripts/29` |

**Reading:** the representation can carry the chemistry. The bottleneck we found later is
*not* the autoencoder's capacity on real molecules.

**Caveat to state out loud:** good reconstruction of *real* embeddings does not imply good
decoding of *generated* embeddings. That gap is Result 3.

---

## Result 2 — The flow does learn a conditional density

Three-way decomposition of the conditional log-likelihood gap, QED 0.4 vs 0.8,
**nats per node**:

| Checkpoint | Δ latent (z) | Δ prior | Δ log-det | **Total** |
|---|---|---|---|---|
| Baseline | +0.2785 | +0.0067 | −0.0182 | **+0.2670** |
| Early (step 4,000) | +0.6398 | +0.0746 | +0.2022 | **+0.9165** |
| Final (step 50,000) | +2.8366 | −0.0233 | +0.8836 | **+3.6969** |

Source: `qed_gap_decomposition.json`, job 366922. Identity verified to 8.9e-15.

**Reading:** conditioning is emphatically *not* being ignored. The separation grows ~14×
from baseline to final. Whatever is wrong, it is not that the flow failed to learn the
condition.

---

## Result 3 — ...but decoded QED barely responds

| N | Latent alignment with real-QED direction | Delivered / required displacement | Prior-route QED projection |
|---|---|---|---|
| 15 | −0.009 | 0.705 | +0.0202 |
| 23 | +0.568 | 1.526 | −0.0089 |
| 30 | **+0.895** | 1.492 | +0.0360 |

Source: final-checkpoint geometry, job 366922.

**The puzzle, stated plainly:** at N=30 the model moves in almost exactly the right
direction (+0.895) with more than enough magnitude (1.49×) — and decoded QED still hardly
changes.

**Our interpretation (suggested, not proven):** the discrete assembly step collapses a
large volume of embedding space onto the same molecule, so a correct latent displacement
can produce no change in the output graph.

---

## Result 4 — Decoder dependence: corrected vs raw

*Leading with corrected, because that is what the teammates and the published papers
report. Raw is shown alongside because the gap is one of our findings.*

| N | Corrected (degree) | Corrected (constrained) | Raw | Gap |
|---|---|---|---|---|
| 15 | [[FILL]] | [[FILL]] | [[FILL]] | [[FILL]] |
| 23 | [[FILL]] | [[FILL]] | [[FILL]] | [[FILL]] |
| 30 | [[FILL]] | [[FILL]] | [[FILL]] | [[FILL]] |

Source: `qed_support_rerank.json` (`scripts/33`), per-decoder validity by N.

Same generated embedding, decoded three ways — per-embedding QED disagreement:

| Decoder pair | median \|ΔQED\| | fraction > 0.10 | same molecule |
|---|---|---|---|
| constrained vs degree | [[FILL]] | [[FILL]] | [[FILL]] |

**Why this matters:** if the same embedding gets materially different QED from different
assemblers, then "the QED of a generated embedding" is a property of the *postprocessing*,
not of the model. That shapes how any conditional result should be read.

---

## Result 5 — High-QED molecules are present, so support is not the problem

| Source | median QED | IQR | fraction ≥ 0.75 |
|---|---|---|---|
| Real ZINC250K (matched N) | [[FILL: qed_support.json]] | [[FILL]] | [[FILL]] |
| Generated, final ckpt, N=15 | [[FILL: qed_support_rerank.json]] | [[FILL]] | [[FILL]] |
| Generated, final ckpt, N=23 | [[FILL]] | [[FILL]] | [[FILL]] |
| Generated, final ckpt, N=30 | [[FILL]] | [[FILL]] | [[FILL]] |

**Reading:** the model *can* produce high-QED molecules. It cannot produce them **on
request**. That separates a capacity problem from a controllability problem, and it is the
cleanest single argument that the failure is in conditioning rather than in chemistry.

---

## Result 6 — Four interventions, four pre-registered failures

| # | Intervention | Criterion fixed in advance | Outcome |
|---|---|---|---|
| 1 | QED-consistency loss (v1) | finite training | **Numerically failed** — non-finite at ~step 5,400; traced to 4 inverse-only sites |
| 2 | Anchored Conditional Prior | ΔQED under latent shift | **NO-GO at all 3 N** (job 368791) |
| 3 | Decoder-aware guidance | surrogate must transfer to generated embeddings | **Gate failed** → QED arms correctly skipped (job 369665) |
| 4 | Generated-QED relabeling | paired response improvement | **Not established** |

**The methodological point worth making to the committee:** every criterion was fixed in
code *before* the data existed, so none of these is a post-hoc rationalisation. Three of
the four were rejected by a cheap read-only gate before any training was spent.

---

## Intervention 2 in detail — ACP gate

ΔRDKit QED under a +1.0 shift along the real-QED direction:

| N | Real-embedding arm | Generated arm | Native |
|---|---|---|---|
| 15 | +0.1278 | +0.0343 | +0.0336 |
| 23 | +0.1121 | −0.0147 | −0.0601 |
| 30 | −0.0172 | +0.0133 | +0.0268 |

Source: job 368791, 8m09s.

**Reading:** on *real* embeddings a latent shift moves QED (N=15, 23). On *generated*
embeddings it does not. No monotonic response anywhere.

**Honest limitation:** the N=15 real arm had only 14 samples and is underpowered. We do
not lean on it.

---

## Intervention 3 in detail — why the surrogate gate matters

QED surrogate validation, before any guidance was applied:

| Feature set | Real held-out MAE | **Generated MAE** | **Generated correlation** |
|---|---|---|---|
| Moments | 0.1062 | 0.2251 | **0.0204** |
| Structural (+ ring/edge terms) | 0.0891 | 0.1519 | **0.1156** |

Thresholds set in advance: generated MAE ≤ 0.12, correlation ≥ 0.30. **Both failed.**

**Consequence:** the QED-guidance arms were skipped by design rather than run against an
unreliable critic. The manifold-only arm (no learned critic) did run and improved QED MAE
in **2 of 9** conditions.

**The trap we avoided, worth showing:** the manifold arm's apparent response gain
(+0.130 → +0.255 at N=30) came from pushing *low*-QED samples lower — 0.428 → 0.234 at
target 0.4 — not from reaching 0.8. A response number alone would have looked like
success.

---

## Scale dependence: what looked good small, and what held up

*This is the slide I would most like your advice on.*

| Experiment | Scale | Result at that scale | Result after scaling |
|---|---|---|---|
| Decoder-guidance pilot | 8 / cell | [[FILL: guidance_pilot.json]] | Did not hold |
| Best-of-k rerank | 100 / cell | [[FILL: qed_support_rerank.json]] | postprocessing only |
| Relabeling eval | 100 / cell | [[FILL: definitive_table.md]] | Not established |

**Why this slide exists and is not buried:** several promising-looking numbers appeared at
8–10 samples per condition and did not survive at 100. With ~10 samples the confidence
interval on a QED mean is roughly ±0.1, which is the same size as the effect we are
chasing. Presenting a small-sample number as a headline result would not survive the first
question about *n*.

**Discussion question:** is the sample-size sensitivity itself worth a section in the
thesis? It is a real and reusable finding about evaluating conditional generators.

---

## Statistical correction we made during final verification

We found and fixed a flaw in our own evaluation:

**Before:** "improvement" was declared when the fine-tuned model's response interval
excluded **zero**.

**Why that is wrong:** that asks whether the new model responds at all. It does not ask
whether it responds *more than the old model*. Two models can both exclude zero and be
statistically indistinguishable from each other.

**After:** the test is on the **difference between models**, with Holm correction across
the three N.

**Limitation we state rather than hide:** the per-molecule deltas were not saved, so the
exact paired test is unavailable. We report a *conservative* bound (shared noise makes the
models positively correlated, so assuming independence overstates the error). A
non-significant result is therefore **inconclusive**, not proof of equality.

Source: `scripts/37_final_consolidation.py`, `final_consolidation.txt`.

---

## Definitive comparison table

| Checkpoint | Step | N | paired ΔQED | 95% CI | pairs | vs final | CI of difference | Significant |
|---|---|---|---|---|---|---|---|---|
| early | 4,000 | 15/23/30 | [[FILL]] | [[FILL]] | [[FILL]] | — | — | — |
| final | 50,000 | 15/23/30 | [[FILL]] | [[FILL]] | [[FILL]] | ref | — | — |
| relabel fine-tune | +400 | 15/23/30 | [[FILL]] | [[FILL]] | [[FILL]] | [[FILL]] | [[FILL]] | [[FILL]] |

Source: `outputs/metrics/final_consolidation/definitive_table.md` — paste directly.

**Protocol, identical for every row:** one prior-noise draw per N reused across all
checkpoints and all QED requests; temperature 0.2; 100 samples per cell; degree-corrected
decoder as headline; response = per-molecule QED difference between highest and lowest
request at fixed N, over **jointly valid pairs only**.

**Seeds:** the evaluation noise stream is distinct from the relabeling pool stream, and
the code refuses to run if they coincide. Earlier stages used their own seeds, so their
numbers are **not** sample-matched to this table and must not be pooled with it.

---

## The early-vs-final trade-off

| | Early (4,000) | Final (50,000) |
|---|---|---|
| Conditional separation (nats/node) | +0.9165 | **+3.6969** |
| QED responsiveness | **better** | worse |
| Raw validity | worse | **better** |
| Chemical quality | worse | **better** |

**The caveat we must state:** the early checkpoint's responsiveness advantage is
confounded with its lower validity. Low-validity models produce different molecule-size
distributions, and we demonstrated that unpaired scoring inflates QED comparisons by up to
**25 percentage points**. Our paired, jointly-valid protocol is designed to remove exactly
that bias.

**Discussion question:** is the responsiveness/validity trade-off a finding to report, or
a confound to control away? We currently treat it as the former with the caveat attached.

---

## What the work establishes / suggests / leaves open

**Establishes**
* The autoencoder is not the limiting factor on real molecules.
* The conditional flow learns a strongly condition-dependent density (+3.70 nats/node).
* That signal does not translate into controlled decoded QED.
* Four interventions fail pre-registered criteria.
* Decoding depends heavily on correction, increasingly at larger N.

**Suggests (not proven)**
* The loss is concentrated at the discrete assembly step: correct direction (+0.895),
  sufficient magnitude (1.49×), no effect on output.
* Pooled decoder statistics barely predict decoded QED on generated embeddings.

**Open**
* Whether the fine-tuned model differs from the original at all (test inputs not saved).
* Whether any gain is intrinsic or specific to the degree decoder.
* Whether joint decoder training, or direct discrete graph generation, would retain it.

---

## Questions for you

1. **Framing.** Is "where conditional control is lost in a flow+AE pipeline, with four
   falsified repairs" a defensible thesis contribution, or do you want a different frame?

2. **Baselines.** Are the ZINC250K validity numbers (GraphDF Table 1) the right comparison
   set, or should we compare against the teammates' QM9 pipeline despite the dataset
   difference?

3. **Raw vs corrected.** We lead with corrected for comparability. Keep raw as a secondary
   column, or move it to the appendix?

4. **Scale dependence.** Promote to a findings section, or keep as a methodological note?

5. **Scope.** We have stopped running new training interventions. Agreed?

---

## Appendix — artefact index

| Slide | Artefact | Script |
|---|---|---|
| Baselines | GraphDF Table 1 | — |
| AE reconstruction | `reconstruction_analysis.json`, `ae_chemistry_audit.json` | 03, 29 |
| Conditional density | `qed_gap_decomposition.json` | 30 |
| ACP gate | `acp_gate.json` | 31 |
| Guidance | `guidance_pilot.json` | 32 |
| Support / rerank / decoder gap | `qed_support_rerank.json` | 33 |
| Relabeling | `relabel_evaluation.json`, `finetune_report.json` | 34–36 |
| Definitive table | `definitive_table.md`, `final_consolidation.json` | 37 |

**Figures available for the deck**
`qed_distribution.pdf`, `n_qed_heatmap.pdf`, `generated_overlay.pdf` (script 14);
`arm_comparison.pdf`, `arm_b_molecules.png` (script 09); stage-11 final figure.

---

## Appendix — definitions, so numbers are comparable

* **Raw validity** — sanitisable AND single connected component, no repair step.
* **Corrected validity** — after valence/degree-constrained assembly. This is what the
  teammates and MoFlow report (`--correct_validity` is on by default in MoFlow).
* **Paired ΔQED at fixed N** — per-molecule QED difference between the highest and lowest
  QED request, same prior draw, jointly valid pairs only.
* **Condition gap** — validation NLL under permuted conditions minus under own conditions.
  Greater than 0 means conditioning is alive; ≈ 0 means it is ignored.
* **Internal diversity** — 1 − mean pairwise Tanimoto over unique valid molecules.
