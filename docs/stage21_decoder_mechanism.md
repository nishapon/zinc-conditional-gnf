# Stage 21 — why degree-constrained decoding loses QED, and the one change made

Stage 20 (job `340259`, commit `44ebe76`) decomposed the degree decoder's attenuation at
N=23, T=0.2 for requested QED 0.4 → 0.8:

| quantity | value |
|---|---:|
| constrained decoder response | +0.0557 |
| degree decoder response | −0.0041 |
| attenuation | +0.0598 (CI [+0.0348, +0.0844]) |
| — integer-programming contribution | +0.0212 |
| — solver / fallback switching contribution | **+0.0386** |

The larger half is not the integer program. It is the **discontinuity around it**. That
determined which mechanism Stage 21 implements.

## Step 1 — the failure, read off the code

`degree_constrained_bond_decode` (`zinc_gnf/decoding_charge_aware.py`) has five exits that
abandon its program and return the answer of a *different algorithm*, the greedy
valence-aware decoder: `bad_degrees`, `no_solver`, `infeasible_degrees`,
`valence_unfixable`, `not_connected`.

### Why switching contributes, and why it is the bigger half

The requested degrees are `degree_logits.argmax(-1)`, taken **independently per atom**.
Nothing constrains their sum to be even, and no graph can realise an odd degree sum, so the
decoder tests:

```python
if wanted.sum() % 2 != 0 or (wanted < 0).any() or (wanted > n - 1).any():
    return fallback("infeasible_degrees")
```

Two consequences:

1. **Parity is uncontrolled.** A sum of independent argmaxes is odd roughly half the time.
   Even for a well-trained head the parity is wrong whenever an *odd number* of atoms is
   mispredicted, which is about half of all molecules.
2. **One atom flips it with certainty.** Changing any single atom's degree by ±1 flips the
   parity of the sum. So a QED request that shifts one atom's degree argmax switches the
   whole assembly algorithm.

When that happens, the two QED conditions are decoded by two different algorithms, and the
paired difference measures *greedy minus integer-program* rather than *QED-high minus
QED-low*. That contaminating term has arbitrary sign and is large relative to the genuine
response — which is exactly the +0.0386 Stage 20 attributed to the `mixed` stratum.

A second non-chemical exit: the degree head emits `max_degree + 1` classes, so a predicted
degree of **0** is representable, and it guarantees a disconnected graph for n ≥ 2. The
program is solved and then discarded by `not_connected`.

In both cases the degree head's output is **discarded in one condition and used in the
other**. That is a decoding defect, not a property of the molecules or of the flow.

### Why the program itself also suppresses response

The program maximises the summed log-odds of *bonded* and chooses the bond **set** only.
Orders are then assigned by an independent per-pair argmax and **demoted** until valence is
satisfied:

```python
bond[i, j] = int(np.argmax(symmetric[i, j, 1:])) + 1     # order, chosen independently
...
j = min(partners, key=lambda q: symmetric[i, q, bond[i, q]])
bond[i, j] = bond[j, i] = bond[i, j] - 1                 # demotion
```

Aromaticity lives in the bond-*order* pattern, and aromaticity is a large part of QED. So
the channel that carries most of the QED response is applied after the set is already
fixed and is then damaged by demotion. That is the +0.0212.

### Are the degree sequence and constraints unnecessarily restrictive?

Both too strict and too weak, in different places:

- **Too strict:** rejecting the entire sequence over a parity mismatch. The inconsistency
  is one unit; repairing it costs one atom one step.
- **Too strict:** treating a predicted degree of 0 as a reason to discard the result rather
  than as an obviously invalid value to clip.
- **Too weak:** nothing checks `Σdᵢ ≥ 2(n−1)` — a connected molecule on *n* heavy atoms
  needs at least *n*−1 bonds. A sequence can be perfectly *graphical* and still have **no
  connected realisation**; the decoder solves it, gets a split graph and falls back. (This
  condition was missing from my first implementation too, which made its connectivity
  repair oscillate until a test pinned it.)
- **Not unnecessary:** the degree *equality* itself. It is the symmetry tie-breaker that
  buys validity, and dropping it is just the `constrained` decoder, which has lower degree
  fidelity. So it is kept, and relaxed only as a last-resort ladder step that the
  feasibility argument says should never be reached.

### Do the edge logits contain alternative feasible structures?

Yes, and Stage 20 already showed it indirectly: the `constrained` decoder's adjacency
responds to the QED request (`bond_set_change_rate`) while the degree decoder's does not,
on the *same* logits. Two different chemically valid assemblies of one logit set therefore
exist, and the degree constraint is what suppresses the variation between them. Stage 21
records `set_objective` and `order_objective` per molecule so the log-probability *cost* of
each decoder's choice is directly comparable rather than inferred.

### Could a different feasible assignment preserve the variation?

That is the hypothesis under test. The honest bound: the `constrained` decoder already
delivers +0.0557 with no degree constraint at all, so **that is the practical ceiling** for
a change to degree-constrained assembly. The goal is to approach it while keeping the
degree fidelity and validity the degree constraint buys — not to exceed it.

## Step 2 — the one mechanism implemented

**Make the degree-constrained path total**, so no input can make it switch algorithms
(`zinc_gnf/assembly_total.py`, decoder name `total`). Four steps, none of which can fail,
none of which is a different algorithm:

1. **Repair the degree sequence** to the nearest one a *connected* molecule can realise —
   `1 ≤ dᵢ ≤ min(n−1, capᵢ)`, `Σdᵢ ≥ 2(n−1)`, even sum, Erdős–Gallai — at minimum
   log-probability cost under the degree head's own softmax. A parity problem costs one
   atom one step instead of the whole sequence.
2. **One integer program for the bond set** under those exact degrees. Feasible on the
   complete pair set by the Erdős–Gallai theorem once step 1 holds, so the totality is
   *proved*, not hoped for.
3. **Connectivity by degree-preserving 2-swaps**, no solver: remove `(a,b)` and `(c,d)` in
   different components, add `(a,c)` and `(b,d)`. Every degree is preserved; cross-component
   pairs cannot already be bonds. Terminates because at least one removed edge must be a
   **non-bridge**, and step 1 guarantees ≥ *n*−1 edges, so a split graph always has a
   component with a cycle.
4. **One small integer program for the bond orders** under valence — replacing
   argmax-then-demote. Always feasible, because order 1 everywhere already satisfies
   valence once `dᵢ ≤ capᵢ`.

No fallback, no algorithm switch, no demotion, no failure path.

### What it is not

The target QED **never reaches the decoder**. It conditions the flow, as always; the
assemblers see only the decoder's own bond and degree distributions. There is no property
oracle, no reranking, and no generate-many-and-keep-the-best: one embedding yields exactly
one molecule per decoder, deterministically. Any QED response recovered is response the
conditional flow had already encoded. An assembly that cannot be completed is reported
**failed** and counts as invalid — invalid chemistry is never presented as a molecule.

## Measured behaviour before the cluster run

On synthetic logits in the sandbox (not a substitute for the real run):

| regime | `total` median | baseline `degree` | exact degrees | assembly failures | baseline fallback rate |
|---|---:|---:|---:|---:|---:|
| confident decoder near a real molecule, 15 % degree error | 12–20 ms | 1–7 ms | 20/20 | 0 | **65–90 %** |
| random incoherent logits, all carbon | 11–24 ms | 0.7–10 ms | 20/20 | 0 | 95–100 % |

and on the tiny end-to-end fixture the baseline's fallback rate was 0.33 at T=0.2 and
0.92–0.96 at T=1.0, while `total` reported `fallback 0.00`, `exact_degrees` on every
molecule, and strict validity 1.000 against the baselines' 0.979.

The cost is 2–4× the baseline's single solve — about 14 ms per molecule, ≈2.4 minutes per
10,000 on one core — which is why the runtime criterion is checked explicitly rather than
assumed.

## How to read the result

1. **Table C, the paired table.** The only measurement that isolates the decoder: same
   embeddings, same noise, one common subset valid under both requests for every
   *assembler*. Compare `total − degree` against the `constrained` column, which is the
   ceiling.
2. **Table D, assembly health.** If `fallback_rate` and `requested_sum_was_odd_rate` are
   near zero on real embeddings, the mechanism this stage fixes was not firing, and a small
   improvement is the *expected* result rather than a disappointment.
3. **Table A**, so a QED gain is not bought with validity, aromaticity or diversity — raw
   native validity is its own row.
4. **Table B**, with well-supported and low-support rows read separately.

`raw` is excluded from the paired intersection on purpose: its argmax validity is ~9–14 %
even for the good unconditional run, so intersecting with it would empty the subset and
make the comparison between the three assemblers unanswerable. It is reported separately
and never merged.
