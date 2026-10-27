# Stage 23 — topology-aware bond conditioner

## What Stage 22c established

| N | real bonds | generated | components | largest | exact N after correction |
|---|---:|---:|---:|---:|---:|
| 15 | 15.5 | 9.9 | 5.79 | 8.0/15 | 1.6 % |
| 23 | 24.7 | 17.4 | 7.00 | 13.4/23 | 0.8 % |
| 30 | 33.0 | 25.5 | 7.17 | 20.2/30 | 0.4 % |

At N=23, **74.2 %** of graphs were over-valent *and* disconnected at once, and the carbon
fraction was **0.637 against 0.732** real. Two facts constrain the design:

1. `BondSubnet` gave a pair only `r_i + r_j` and `|r_i − r_j|`, and a six-ring versus two
   three-rings produced a difference of **exactly 0.000e+00** on every shared pair.
2. The **marginals are also wrong** — fewer bonds than real, and too little carbon. Not
   everything can be blamed on topology.

## 1. Architecture: masked 2-D convolution over the adjacency grid

### Why not iterated message passing

The obvious repair — iterate the pair → atom → pair aggregation a few times — **does not
work**, and this is the central design argument. A six-ring and two three-rings are both
2-regular: every atom has degree 2, every neighbour has degree 2, the colouring is stable
at the first iteration. They are the textbook counterexample to 1-WL. So *any*
sum-aggregation scheme over the atom graph assigns identical atom embeddings in both, and
any pair feature derived from them is identical too. More rounds cannot help.

### Why a convolution does work

A convolution over the N×N grid is **not 1-WL-bounded**: it reads the adjacency pattern as
a function of atom *index*. Pair (0,1) sits at grid position (0,1), and the positions where
the two graphs actually differ — `{(2,3),(5,0)}` versus `{(2,0),(5,3)}` — fall inside its
receptive field. This is the mechanism MoFlow used (10 coupling layers of 3×3 convolutions
at 512 hidden; 82 % validity without correction on ZINC250K).

### The design

```
pass-through channels  ->  scatter to a symmetric [B, C, N, N] grid
                       ->  concat a mask channel
                       ->  3x3 conv + FiLM-style condition bias, masked        (stem)
                       ->  2 x (3x3 conv, masked)                              (blocks)
                       ->  concat masked row-mean and column-mean, 1x1 mix      (axial)
                       ->  1x1 head to 2 x active channels, averaged with its transpose
                       ->  gather at the strict upper triangle
```

The **axial aggregation** is deliberate. Dropping the endpoint sums would have thrown away
the degree information Stage 22 *did* carry — it was useful, just insufficient alone. Row
and column means give pair (i,j) a summary of everything incident on atom i and atom j, so
the conditioner has both endpoint-global and local structural information rather than
trading one for the other. The mix is 1×1 because the means are already global along their
axis.

### Tradeoff, stated

Using the index grid means the model is **not permutation equivariant**: its density is
over canonically ordered graphs. That is already true of this project's representation
(`chemistry_charge_aware` fixes the atom order to canonical SMILES order) and of MoFlow and
GraphNVP, so it adds no new limitation. Canonical SMILES order is a DFS walk, so bonded
atoms tend to get nearby indices — which is what makes the inductive bias useful rather
than arbitrary.

### Verified before proposing the run

| conditioner | difference on the 1-WL pair | parameters (N=38) |
|---|---:|---:|
| `pairwise` (Stage 22) | **0.0000e+00** — blind | 1,500,538 |
| `topology` (Stage 23) | **non-zero** — distinguishes | **1,309,690** |

The new conditioner uses **fewer** parameters (0.873×), so an improvement cannot be
attributed to capacity. Both halves of that comparison are asserted by a test.

### Honest bound on the receptive field

With 3 layers of 3×3 the radius is 3 in index space, and the axial means summarise only the
pair's own two atoms. So within a *single* coupling, a rearrangement among atoms sharing
neither an index neighbourhood nor an endpoint with a probe pair can be invisible.
Information still crosses the molecule across the six coupling blocks. A test pins this at
exactly 0.0 so that if the receptive field ever grows, this note must be updated rather
than silently drifting.

### What this is not

A more expressive coupling conditioner is **not** automatically a valid molecular
generator. It removes a proven blindness; whether that translates into connected,
correctly-valent, exact-N molecules is what the experiment measures.

## 2. Preserved, and re-verified for the new conditioner

| property | how |
|---|---|
| invertibility | exact round trip 2.4e-07 at init; tested at several molecule sizes for **both** conditioners |
| log determinant | checked numerically against the true Jacobian |
| no leakage | the conditioner receives **only** pass-through channels — asserted by recording what the subnet is handed, and that forward and inverse see the same input |
| padding masks | latents exactly zero on padding; garbage in padding cannot change the density |
| bond symmetry | structural: only the strict upper triangle is modelled, `unpack_upper` mirrors it. The conv output is also averaged with its transpose so parameters don't depend on pair orientation |
| variable N | per-molecule masks; exact N by construction |
| atom vocabulary | unchanged 17 `(element, charge)` types, Kekulé bonds |
| N and QED conditioning | unchanged: every coupling **and** the prior, with conditioning dropout |
| evaluation | the same molecular metrics, so numbers stay comparable |

## 3. Numerical stability

`max_log_scale = 0.75`, `max_shift = 3.0`, and **they are not restated anywhere**. Every
script derives them from `GraphFlowConfig` — the Stage-22a defect was the script
hard-coding 2.0 while the dataclass said 0.75. A test asserts the script's defaults equal
the dataclass's for all seven numerical fields.

Inversion is now checked at **three** points:

- **at initialisation** — and the run **refuses to start** if the inverse is
  ill-conditioned, rather than discovering it after the GPU hours are spent;
- **at every evaluation** — `inverse_round_trip_worst` is recorded in the history and an
  ERROR is logged above 1e-3;
- **at every generation call** — reported per cell and surfaced in the report.

Resuming with a different configuration is refused with the exact flag needed. Stage-22
checkpoints still load: `from_dict` maps a missing `bond_conditioner` entry to `pairwise`,
so they cannot silently become a different architecture.

Measured, with deliberately over-perturbed subnets (a stress figure, not a training
figure — the trained model measured 2.4e-07):

| `max_log_scale` | pairwise | topology |
|---:|---:|---:|
| 0.75 | 1.5e-03 | 3.5e-03 |
| 2.0 | 7.1e+01 | 2.4e+02 |
| 4.0 | 2.9e+12 | 3.0e+12 |

`exp(6 × 0.75) = 90` against `exp(6 × 2.0) = 162,755` and float32's ~1e-7 precision.

## 4. The experiment

Same ZINC250K split, train for fitting and validation for selection; the test split cannot
be loaded. **Stage 22 weights are never loaded into the new architecture** — the
conditioner's parameters differ, and the config guard refuses it.

Four arms, identical grid, counts and seeds:

| arm | what it is |
|---|---|
| `graph_flow` | the topology conditioner on true `(N, QED)` |
| `graph_flow_null` | the same architecture, budget and seed with **QED permuted within N** |
| `graph_flow_previous` | the Stage-22 pairwise checkpoint, READ-ONLY — the thing to beat |
| `graph_flow_prior` | the real arm with the coupling condition neutralised |
| `baseline` | the Stage-17 embedding GNF, READ-ONLY |

Grid N ∈ {15, 23, 30} × QED ∈ {0.4, 0.6, 0.8} × T ∈ {1.0, 0.85, 0.6}.

**T = 1.0 is listed first deliberately.** Reduced-temperature sampling on a dequantized
one-hot representation biases every variable towards its *majority* class, which for bonds
is "no bond" — so part of Stage 22c's bond deficit at T=0.85 is that bias rather than the
model. T=1.0 is the unbiased reading; the sweep then shows what temperature costs.

Support is reported per cell: N=15 and N=23 have thin low-QED support (76 and 236
molecules), which bounds what those cells can show.

## 5. Success criteria, in order

1. **Fragmentation falls and raw validity rises** against the Stage-22 arm — components
   down from ~7.0, largest component up from 13.4/23, raw validity up from 0.0012.
2. **Not bought by inflating bond density.** `bonds` must move *towards* `real` while
   `over-val` does **not** climb with it. Both columns rising together is a **failure**,
   and the report says so where the table is printed.
3. **QED responsiveness survives on structurally valid, exact-N molecules** — a slope
   measured over fragments is a slope over the wrong molecules, which is why Stage 22b's
   +0.0907 was discarded. It only counts once exact N is no longer ~1 %, and only if it
   separates from the null arm.

Raw validity stays separate from post-correction validity throughout, and because
correction keeps only the largest connected component and can drop atoms, exact N is
reported both before and after.

## 6. Cost

| | Stage 23 PoC |
|---|---|
| parameters | 1,309,690 (**fewer** than Stage 22's 1,500,538) |
| steps | 9,000 per arm, batch 32 |
| arms | 2 trained (topology, shuffled-QED null) + 3 read-only comparisons |
| GPU memory | activations dominated by `[32, 48, 38, 38]` per conv layer ≈ 11 MB; **< 4 GB** total |
| runtime | ≈ 3–4 h per arm (a step costs ~1.5–2× Stage 22a's); both arms plus CPU evaluation inside a 12 h limit |
| resumability | exact next batch, checkpointed every 750 steps; a finished arm is a no-op |

No full-scale training, no deployment claim.

## 7. How to read the result

0. `train.log` → *"inverse at initialisation"*. If the run refused to start, the numerics
   were rejected before any GPU time was spent. Then check `inverse_round_trip_worst` in
   the history: above 1e-3 no generation metric is valid.
1. **Table C, the structure table.** This is what Stage 23 exists to move. Compare
   `graph_flow` against `graph_flow_previous` directly — same grid, same seeds.
2. `raw valid` in the same table, against Stage 22's 0.0012 and the embedding pipeline's
   ~0.09.
3. `condition_gap_trend.status`. If `rising_undertrained`, resubmit with `STEPS=18000` (it
   resumes) and conclude nothing about QED yet.
4. Table A's slope, **against the null arm row**, and only if exact N has improved.
5. The `graph_flow_prior` row — if it matches the full arm, the model is only shifting its
   conditional prior.
6. `diagnostics/graph_flow_diagnostics.json` for the per-cause breakdown and the
   atom/bond marginals. If the carbon gap (0.637 vs 0.732) persists, the **atom** block is
   still under-fit and topology was not the only problem.

One expectation set honestly: MoFlow's 82 % came from a far larger model (10 + 38 coupling
layers at 512 hidden) trained to convergence. This budget buys a test of the *mechanism*.
