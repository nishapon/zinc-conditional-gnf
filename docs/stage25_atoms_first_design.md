# Stage 25 — atoms-first conditional Graph Normalizing Flow

## The hypothesis, stated so it can fail

Stages 22–24 modelled the joint as

```
bonds_first   log p(G|c) = log p_bond(A|c) + log p_atom(X|A,c)
```

Stage 25 reverses it:

```
atoms_first   log p(G|c) = log p_atom(X|c) + log p_bond(A|X,c)
```

**Both are exact factorisations of the same joint.** Nothing in this stage claims that
bonds-first is *incapable* of representing chemically valid molecules — with unbounded
capacity and unbounded data the two parameterise the same family. The hypothesis is about
**learnability**, inductive bias and sampling quality: that a bond conditioner which can
*see* the chemistry of the two atoms it is joining will place bond orders that are
compatible with them, and that this is easier to learn than recovering compatibility
afterwards by choosing atom types to fit bonds already drawn.

## Why this change and not another

Stage 24 (commit `54b7ed4`) left two facts side by side.

From the code: `ConditionalGraphFlow.encode` ran the bond stack with `extra=None`. The
block choosing bond orders never saw an atom type, while valence is a constraint between
an atom's **type** and the sum of its incident bond orders.

From the cluster run at N=23, QED 0.8, T=1.0:

| measurement | value |
|---|---|
| raw validity | 7.81% |
| bonds per graph | 24.45 vs 24.65 real |
| connected components | 1.20 |
| over-valent atoms per graph | 2.42 |
| best-case relabelling → valence-feasible | 98.4% of graphs, ~10.5% of atoms changed |
| real atoms + generated bonds → raw validity | 2.34% |
| generated atoms + real bonds → raw validity | 14.84% |
| aromatic rings per real N=23 molecule | ~1.848 |
| alternating share of generated 5-/6-rings | 2.64% (real reference 45.6%) |
| representation preserves aromaticity on real molecules | 100% |

Two readings of that evidence point the same way and one complicates it:

* the bond patterns were **realisable** — some atom assignment made 98.4% of them
  valence-feasible — so the bonds were not arbitrary and the atom choice was carrying a
  burden it could not carry against a 73%-carbon marginal;
* the representation is **not** the problem: real molecules keep their aromaticity through
  our own serialise → Kekulise → reconstruct path, so missing aromaticity is the model or
  the repair, not the encoding;
* but substituting the **bonds** helped more than substituting the atoms (14.84% vs
  2.34%). Atom-type marginals are wrong *too*. Reversing the order gives the bond stack
  atom information; it does not by itself fix the atom distribution, and section
  "Limitations" below says so.

Feasibility under the audit's allowed-valence rules is not sanitisation and not achieved
validity — it bounds what a better-coupled model could reach, nothing more.

## What was built

### `AtomChemistryContext` — what the bond conditioner now sees

Per pair `(i, j)`, assembled from the atom tensor the atom stack produced:

* a learned embedding of each endpoint's channels (`atom_context_dim`, default 8);
* the endpoint's **allowed valence**, read from the charge-aware vocabulary and scaled by
  the vocabulary maximum (6, sulphur);
* the endpoint's **formal charge** — `N+1` is not read as neutral nitrogen.

Each is combined symmetrically as `(f_i + f_j, |f_i − f_j|)`, so the context cannot teach
an orientation, and masked so padded atoms contribute exactly zero. Default width: 20
per-pair channels, entering the `TopologyBondSubnet` convolution as extra **input
channels**, so endpoint chemistry and local topology are read together rather than in
separate pathways.

### Continuous versus discrete conditioning — the explicit choice

| mode | what the conditioner reads | consequence |
|---|---|---|
| `dequantized` (default) | the continuous dequantized channels the likelihood itself sees | every operation is linear or smooth; **nothing non-differentiable or non-invertible enters the likelihood path**; training and sampling condition on the same quantity |
| `onehot` (labelled variant) | `argmax` of those channels | `p(x_b | argmax(x_a), c)` is still a normalised density in `x_b` for each fixed `x_a`, so the joint is a valid density and the dequantization bound stands — but the density is **piecewise** in the atom channels |

A test measures exactly this difference: the gradient of the bond term with respect to the
atom channels is non-zero in `dequantized` mode and exactly zero in `onehot` mode. The
default is `dequantized`; `onehot` is offered because it matches decoding, which argmaxes
anyway, and it is never silently selected.

### `AtomSequenceSubnet` — a confound control, not an upgrade

Atoms-first has no bonds when the atom stack runs, so `AtomSubnet`'s relational message
over the bond tensor is unavailable. Replacing it with a per-atom MLP would make `p(X|c)`
a product of **independent** per-atom densities, unable to represent composition
("exactly one sulphur, three nitrogens") — strictly weaker than stage 23, which would
confound the factorisation with a crippled atom model. The replacement therefore keeps
cross-atom structure through two pass-through-only, masked paths:

* a 1-D convolution along the atom **index** axis (canonical SMILES order is a DFS walk,
  so index-adjacent atoms are usually bonded — the same inductive bias, and the same
  stated non-equivariance tradeoff, that the stage-23 bond conditioner already relies on);
* the masked **mean** over real atoms, broadcast back, so each atom's transformation can
  depend on the whole composition.

A test asserts that atom 7's identity reaches atom 0's latent, so this is a measured
property rather than an intention.

### Parameters: fewer, deliberately

| | total | bond stack | atom stack | atom context |
|---|---|---|---|---|
| bonds-first (stage 23 baseline) | 1,309,690 | 319,272 | 978,942 | 0 |
| atoms-first (stage 25 default) | 1,258,778 | 371,112 | 876,030 | 160 |

0.961× the baseline — strictly **fewer**. A win therefore cannot be attributed to
capacity; a loss leaves the question open and owes a capacity-matched rerun
(`ATOM_CONV_HIDDEN=160` overshoots to 1.32×). Both numbers are pinned by a test so a
default change cannot quietly invalidate the comparison.

## What is preserved

Exact log-determinants (checked against the full autograd Jacobian, both orderings);
exact inverses (2.4e-07 at initialisation); the one-shot inverse round-trip guard at
initialisation, at every evaluation and at every generation call; padding inertness in
the latents and in the density; undirected bond symmetry, which is structural because only
the strict upper triangle is modelled; variable heavy-atom counts; the charge-aware
17-type vocabulary; native `(N, QED)` conditioning through both the couplings and the
prior; conditioning dropout and the learned null; and no leakage — every conditioner still
reads only its coupling's pass-through channels, so the inverse recomputes the same scale
and shift.

Stage 23 checkpoints remain loadable, byte-identical in `state_dict`, and **resumable**:
`factorisation` defaults to `bonds_first` in the dataclass and in `from_dict`, a
bonds-first model builds no `atom_context` parameters at all, and the trainer now
normalises both configs through `from_dict` before comparing them, so fields added in
stage 25 do not read as a mismatch. Resuming with a genuinely *different* factorisation is
still refused.

## The controlled experiment

| arm | what it is |
|---|---|
| `atoms_first` | the new factorisation on true `(N, QED)` labels |
| `atoms_first_null` | the same architecture, budget and seed with QED **permuted within N** — the null for every QED number the new architecture reports |
| `bonds_first` | the stage-23 18,000-step checkpoint, **read-only**, under identical requests, seeds and temperatures |

Matched data, matched protocol, matched validation subset for the likelihood numbers.
Training is pure maximum likelihood: **no valence penalty and no correction in the
objective**. An auxiliary valence loss stays a separate experimental variant and is not
part of this stage.

Budget: 9,000 steps per arm with an evaluation every 750 — an **interim** read, because the
baseline has 18,000. `STEPS=18000` resumes both arms to the matched budget. If
`condition_gap_trend.status` is `rising_undertrained`, the budget was too small and no
architectural conclusion is warranted.

## What is reported

`scripts/27_compare_factorisations.py` produces, per arm and per cell: native raw validity
and exact N; valence failure rates by element and formal charge, plus the share of atoms
beyond *any* cap in the vocabulary; connected components; bond counts and bond-order
distributions; atom-type distributions; aromatic rings and Kekulé alternation; ring sizes;
corrected validity with the bond orders it changed and the **atoms it lost**; uniqueness,
novelty against train only, internal diversity; QED MAE, hit rate and within-N response
slope with a bootstrap CI — both unrestricted and **restricted to valid exact-N molecules
with their support counts**; the conditional-likelihood permutation gap; the inverse
round-trip error; and the validation likelihood and discrete bound. The slope estimator is
the same one `scripts/24_evaluate_graph_flow.py` uses, and a test asserts the two agree
exactly so they cannot drift.

## Acceptance criteria

Applied **conjunctively** by the verdict block, which can return `no`:

1. native **raw validity** rises against the bonds-first arm;
2. valence failure rates fall;
3. the atom-type distribution does not degrade — a validity gain bought by turning the
   molecule into carbon is not a gain, and both arms are compared against the real
   distribution;
4. **exact N** is preserved, reported natively and after correction;
5. **aromaticity** improves — a *separate* criterion, read against the measured real
   ceiling (~1.848 rings per real N=23 molecule) and reported beside the alternating-ring
   share, because better valence does not imply learned aromatic chemistry;
6. the QED response on **valid, exact-N** molecules separates from the shuffled-QED arm;
7. the inverse stays sound (round trip ≤ 1e-3), which is decisive whatever else improved.

100% corrected validity is **not** native generation success, and is never the headline.

## Limitations

* **The atom marginals may not move.** Stage 24 found substituting real *bonds* helped
  more than substituting real *atoms* (14.84% vs 2.34% raw validity). Reversing the order
  gives the bond stack atom information; it does not directly fix a wrong atom
  distribution, and criterion 3 exists to catch a trade rather than a fix.
* **Errors now compound the other way.** At sampling time the bond stack conditions on the
  model's *own* atoms, so bond errors compound on atom errors instead of the reverse. This
  is inherent to the factorisation, and the arm's own exact-N and validity numbers are
  where it would show.
* **The atom stack lost its relational view of bonds.** The index convolution plus set
  mean is a different inductive bias, not a strictly stronger one; a test pins what it
  can and cannot see.
* **Index locality, not permutation equivariance.** The density is over canonically
  ordered graphs. That was already true of stages 22–24, MoFlow and GraphNVP, so it adds
  no new limitation — but it is not an invariance claim.
* **A null result is not a refutation of the hypothesis.** At 0.961× capacity and an
  interim 9,000 steps, a loss would need the matched budget and a capacity-matched rerun
  before concluding that atom-conditioned bond generation does not help.
* **Nothing here is a deployment claim.** A positive result licenses a longer run.
