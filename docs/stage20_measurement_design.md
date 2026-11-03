# Stage 20 — measurement design, and what it fixes in stage 19

Stage 19 (job `340255`, commit `dc3a3df`) printed one verdict, `H2-direction`. Stage 20
does not dispute that weak QED-predictive conditioning is a leading hypothesis. It
disputes three specific measurements the verdict rested on, and the exclusivity of the
verdict itself.

## 1. The probe's MAE was used as a detection threshold. It is not one.

Stage 19 compared a paired embedding change of about +0.05 to +0.08 against the probe's
held-out MAE of 0.0885 and concluded the change was unresolvable.

Those are different quantities. The MAE is the error of a **single absolute prediction**.
The measured quantity is the **mean of a paired difference** over hundreds of pairs, where
the same molecule appears on both sides, so per-molecule bias cancels, and whose standard
error shrinks roughly as `1/sqrt(pairs)`. Using the former as a threshold for the latter is
biased towards declaring "no signal". `tests/test_qed_pathway.py::
test_a_paired_mean_is_resolved_far_below_the_single_prediction_error` demonstrates this: a
true shift of 0.02 under per-molecule noise of 0.09 is resolved over 400 pairs.

Stage 20 replaces it with two measured quantities, both taken on **real validation
molecules**:

| quantity | what it is | why it is needed |
|---|---|---|
| **gain** | predicted ÷ true difference for real low-QED/high-QED pairs at **fixed N** | the instrument's sensitivity to exactly the contrast the intervention produces. A gain of 0.25 means an observed +0.05 implies a real difference of +0.20 |
| **null** | predicted difference for real pairs in the **same** QED band, where the true difference is ≈ 0 | the actual noise floor of a paired contrast. This, not the MAE, is the detection threshold |

A reported change is `resolved` only when its bootstrap interval separates from the null
interval, and it is then divided by the gain to give an *implied real QED difference*.

The second deficiency was never measured at all: the stage-19 probe was fitted to
**absolute** QED across every N, and most QED variance in ZINC250K is between-N. So a
second read-out is fitted for comparison — `WithinNRidgeProbe`, closed-form ridge on
**within-N centred** QED, trained on train only with `alpha` chosen on an inner split of
train. It is deliberately linear, so a paired prediction difference is exactly
`w · (f_high − f_low)`: a linear functional of the displacement stage 18 measured, with
nothing hidden in a non-linearity (asserted to 1e-12 in the tests).

The stage-19 MLP probe is carried over **unmodified**. `EmbeddingQedProbe.forward` pools to
`cat([mean, std])` and applies `.net`, so applying `.net` to `pooled_features(...)` is the
same function — asserted bit-exactly (`atol=0.0`) so both read-outs are scored on one
identical feature matrix.

## 2. The transfer control ran on training molecules.

94–99 % preservation was measured on molecules the autoencoder was **fitted on**.
`transfer_on_split` repeats it on the **validation** split (the test split is never
loaded), keeping per-molecule values so the preservation ratio carries a bootstrap
interval instead of being a bare ratio of two group means. The train split is measured too,
only so the generalisation gap is visible. Aromaticity, ring counts and raw / constrained /
degree validity sit beside it, and the true QED of the molecules that **survived** decoding
is reported next to the true QED of all selected ones — a decoder that only manages to
build the high-QED molecules would otherwise look like one that preserves the gap.

## 3. H2 structurally masked H3.

The stage-19 ladder tests H2 before H3, so once `H2-direction` fired the H3 branch was
unreachable — which is why the degree-decoder attenuation visible in its own tables could
not appear in the verdict. The mechanisms are not exclusive.

Stage 20 grades four axes **independently**, each with its own status and uncertainty:

- **E1** generated embeddings move weakly, or not along a QED-predictive direction
- **E2** the decoder loses QED information on held-out real embeddings
- **E3** degree-constrained assembly attenuates the response that does arrive
- **E4** generated embeddings are shifted from the real ones the decoder was fitted on

Statuses are `supported`, `not_supported`, `inconclusive` and `needs_further_test`. The
last is reserved for *the instrument being inadequate* and is reported as a fact about the
instrument, never as evidence about the model. That is the guard which prevents "our probe
could not see it" from being recorded again as "the embeddings carry no QED information".

## The identity that makes H3 attributable

`degree_constrained_bond_decode` returns, on **every** fallback path (`bad_degrees`,
`no_solver`, `infeasible_degrees`, `valence_unfixable`, `not_connected`), precisely
`valence_aware_bond_decode_typed(atoms, probs, caps)` — which *is* the `constrained`
decoder. So on a molecule where it fell back under both QED requests, its response cannot
differ from the constrained decoder's.

Both decoders are compared on **one common subset** (valid under both conditions and both
decoders), so a gap can never be an artefact of different survivors. That subset splits
into three strata whose contributions add up exactly, by the law of total expectation:

```
gap = mean(Δ_constrained) − mean(Δ_degree)
    = Σ_strata (stratum share) × (stratum gap)

  fallback_both   degree ≡ constrained by construction → contributes exactly 0
  solved_both     the integer program ran in both conditions → the pure IP effect
  mixed           it ran in one condition only → a decoder-switching effect
```

`fallback_identity_max_deviation` is the measured self-check; if it is non-zero the
decomposition is invalid and E3 returns `needs_further_test` rather than a number.

One further mechanism is measured directly. The integer program constrains the bond **set**
to match the predicted degree sequence. If those predicted degrees do not change when the
request changes, the constraint is *identical* under both requests, the skeleton is frozen,
and the only channel left is the bond **order** chosen inside that fixed set — far narrower
than the constrained decoder, which lets the adjacency itself respond.
`degree_prediction_change_rate`, `bond_set_change_rate` and `bond_order_change_rate`
separate exactly that. Validity selection is isolated as the common-subset minus own-subset
difference. Native raw-argmax validity is carried through unchanged and reported separately.

## A sharper distribution-shift statistic

Stage 19's Mahalanobis mean ratio (0.85–1.05) does not establish that generated embeddings
are on-manifold: two clouds can share a mean distance and still be perfectly separable.
Stage 20 adds a held-out real-versus-generated **AUC** from a cheap ridge classifier on
pooled embeddings. `test_discriminability_separates_clouds_whose_mahalanobis_means_match`
constructs exactly that case — ratio ≈ 1.0, AUC > 0.9 — so the limitation is demonstrated
rather than asserted.

## Scope

Read-only and CPU-only. No flow retraining, no architecture change, no checkpoint written;
writes only under `outputs/metrics/qed_pathway`. Both read-outs are diagnostic: neither
guides generation, reranks a sample, nor is reported as achieved QED. The recommendation
among A (flow) / B (decoder) / C (representation) is a recommendation only, and ships with
the preconditions and the observations that would change it.

## Reading order

`outputs/metrics/qed_pathway.txt`, in this order — the gain/null columns first, because
they decide what every later number *means*:

1. **table B** — gain and null. If neither read-out has a within-N gain whose interval
   excludes zero, the embedding axis is **unmeasured, not cleared**, and no flow-side or
   representation-level decision is justified from this run.
2. **table A** — validation rows, against the train rows. A large gap means stage 19's
   94–99 % was memorised.
3. **table C** — `resolved` plus an implied difference far below the request is the evidence
   for a weak QED-predictive direction. `not_resolved` is a statement about the instrument.
4. **table D** — check `fallback_identity_holds` first, then the strata. Read it **per
   temperature**: stage 19 showed T=1.0 behaving differently from T=0.2.
5. **table E** — the exact denominators. N=15 and N=23 have thin low-QED support (76 and
   236 molecules), which bounds what their cells can show.
