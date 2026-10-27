# Stage 24 — the bond counts are right and the molecules are still invalid

Stage 23's topology-aware bond conditioner worked at the level it was aimed at. At N=23,
QED 0.8, T=1.0 the flow now produces 24.45 bonds against 24.65 real and 1.20 connected
components, against stage 22's 17.4 bonds and 7.00 components. Fragmentation is
essentially solved.

Nothing downstream improved in proportion. 229 of 256 graphs (89.5%) carry an over-valent
atom, raw validity is 7.81%, and aromatic rings sit at 0.008 per corrected molecule
against roughly 2 in real ZINC250K.

That combination — correct global counts, broken local structure — is a specific
signature, not a general "needs more training". The model has learned **how many** bonds a
molecule of this size carries. It has not learned **which atom** may carry them.

| cell | real bonds | generated | components | over-valent atoms | over-valent graphs | raw validity |
|---|---|---|---|---|---|---|
| N=23, QED 0.8, T=1.0 | 24.65 | 24.45 | 1.20 | 2.42 | 89.5% | 7.81% |
| N=30, QED 0.8, T=1.0 | 32.99 | 32.98 | — | 3.64 | 96.9% | 1.95% |
| aggregate | — | — | — | — | — | 29.73% |

Corrected exact-N 79.5%, corrected QED slope +0.0818 against a shuffled-QED floor of
+0.0049, aromatic rings 0.008 per corrected molecule.

## Two root causes, both established by reading the code

### H-VALENCE — the flow cannot see valence, by construction

`ConditionalGraphFlow.encode` (`zinc_gnf/graph_flow.py`) factorises the joint as

```
z_bond, logdet_bond = self._run(self.bond_layers, bonds, pair_mask, context, None)
dense = unpack_upper(bonds, self.config.max_nodes)
z_atom, logdet_atom = self._run(self.atom_layers, atoms, node_mask, context, dense)
```

that is, `log p(G|c) = log p_bond(A|c) + log p_atom(X|A,c)`. Bonds come **first** and atoms
are conditioned on them. The bond block's conditioner receives `extra=None`: it never sees
an atom type, at training time or at sampling time.

Valence is not a property of a bond pattern. It is a constraint coupling an atom's **type**
to the **sum of its incident bond orders** — carbon 4, nitrogen 3, oxygen 2, the halogens
1. A bond block that does not know the atom types therefore cannot avoid violating it, no
matter how well trained or how expressive its conditioner is. Stage 23 made that block
better at topology and it remains structurally blind to the only constraint that matters
here.

The whole burden then falls on the atom block, which must choose a type for every atom that
covers a bond-order sum already fixed. It is doing that against a training marginal that is
73% carbon (cap 4), and an atom carrying 5 or 6 orders needs P or S — types that are rare
in the data. Pushing the atom marginal towards sulphur to satisfy valence would wreck the
atom distribution, which is the second thing stage 23 reported as wrong ("the atom
distribution also differs from real data, especially nitrogen, oxygen and halogens"). The
two failures are the same failure: the atom block is being asked to repair a bond pattern
it had no say in.

**Falsifiable prediction.** If this is the mechanism, then for most generated graphs there
*exists* an atom assignment making every atom valence-feasible — the bonds are realisable
and the atom choice is what fails. `best_case_atom_assignment` measures exactly that, and
`infeasible_for_every_type` (bond-order sum above 6, the largest cap in the vocabulary) is
the counter-evidence: those atoms cannot be repaired by any relabelling and would mean the
bond distribution itself is wrong.

### H-AROMATIC — the corrector deletes the double bonds

`valency_correction` (`zinc_gnf/graph_flow_generation.py`) implements MoFlow's rule:

```
target = int(partners[int(np.argmax(bonds[atom, partners]))])
bonds[atom, target] = bonds[target, atom] = bonds[atom, target] - 1
```

It removes one order from the **largest-order** bond of an over-valent atom. The dataset is
a charge-aware **Kekulé** representation (`Chem.Kekulize(kekule, clearAromaticFlags=True)`),
so in an aromatic ring the largest-order bonds are precisely the **double** bonds — and the
alternating single/double pattern is the only thing RDKit can perceive aromaticity from.
The rule is therefore biased towards destroying aromatic rings specifically. At 89.5%
over-valence, nearly every molecule passes through it.

`tests/test_valence_aromaticity.py` demonstrates this on a constructed graph: a benzene with
a double-bonded substituent on atom 0, where the substituent double bond and a ring double
bond tie for largest order. Removing the substituent's order would leave an intact aromatic
ring. The rule takes the ring bond, and the alternation is gone.

This does **not** by itself explain 0.008 aromatic rings per molecule — it explains how a
ring that *was* alternating stops being perceived as aromatic. Whether the model produced
alternating rings at all is a separate question, which is why `correction_damage` measures
aromaticity and alternating-ring counts **before and after** correction on the same graphs,
and why `ring_profile` classifies 5- and 6-rings as `alternating` / `all_single` /
`mixed_with_double`. An `all_single` 6-ring is a cyclohexane, not a failed benzene; counting
rings without the pattern would make "the model builds rings" look like "the model builds
aromatic rings".

Ring detection uses `networkx.minimum_cycle_basis`, not RDKit, because about 92% of graphs
at N=23 cannot be sanitized and that is exactly the population to be measured.

## Reading the audit

`scripts/26_valence_aromaticity_audit.py` (driver: `slurm/24_valence_aromaticity.sh`) is
read-only, CPU-only, trains nothing and selects no molecule by any property. It measures,
over a grid of N ∈ {15, 23, 30} × QED ∈ {0.4, 0.8} × T ∈ {1.0, 0.85}:

| measurement | what it decides |
|---|---|
| real-molecule ceiling (first) | whether the **representation** keeps aromaticity through serialise → Kekulise → reconstruct. If not, nothing else is interpretable |
| `best_case_relabelling.valence_feasible_rate` | high ⇒ bonds realisable, **atom choice** is the failure (supports H-VALENCE); low ⇒ the bond distribution is wrong |
| `infeasible_for_every_type` | atoms beyond cap 6: unfixable by relabelling, counter-evidence to H-VALENCE |
| real atoms + generated bonds | the atom block's isolated contribution |
| generated atoms + real bonds | the bond block's isolated contribution |
| `alternating_rings_destroyed`, aromatic before → after | whether the **repair** is the proximate cause of missing aromaticity, or the model never made the alternation |
| ring size and pattern distributions vs real | whether rings are the wrong size or the right size with the wrong bond orders |
| `argmax_shift` | whether **discretisation** sharpens the atom marginal beyond the density — a decoding fix rather than an architectural one |

The oracle arms substitute real **structures** (atom types, bond matrices) as a diagnostic
reference. They are labelled as such in every output field, are never a generation aid, and
no QED oracle or reranking is used anywhere.

## Remaining uncertainty

* **How much of the gap is undertraining.** H-VALENCE is a structural claim and is
  established; it does not prove that a corrected factorisation trained for 18,000 steps
  would be valid. Stage 23's own `condition_gap_trend` should be re-read before any claim
  that the budget was sufficient.
* **Whether the model ever produces the alternation.** This is what the before/after split
  in `correction_damage` is for. If `aromatic_rings_before_per_graph` is itself ~0, the
  corrector is not the main cause of missing aromaticity and the ring-pattern distribution
  becomes the thing to fix.
* **Whether valence feasibility is sufficient for validity.** A graph can be
  valence-feasible and still fail sanitisation (e.g. a non-Kekulisable ring system). The
  audit reports raw validity beside feasibility rather than inferring one from the other.
* **Atom marginals.** Even with the factorisation reversed, the reported nitrogen, oxygen
  and halogen deficits may persist; they are not explained by the bond ordering alone.
* **The counterfactuals are not a model.** A high best-case feasible rate shows a valid
  assignment *exists*; it does not show that a reversed factorisation would *find* it.

## Recommended Stage 25 change — the smallest one that is justified

### Reverse the factorisation: generate **atoms first**, condition bonds on atom types

```
log p(G|c) = log p_atom(X|c) + log p_bond(A|X,c)
```

Concretely: swap the two stacks in `encode`/`sample` so the atom stack runs with
`extra=None` and the bond stack receives a per-pair atom-type feature (the two endpoints'
types, plus their valence caps) in place of the current `dense` bond grid.

Why this is the minimal change:

* **It is an exact alternative factorisation of the same joint.** Both orderings are
  products of conditionals of `p(X, A | c)`, so the model remains a normalising flow with
  an **exact likelihood** and tractable, exact log-determinants. Nothing about
  invertibility, masking, bond symmetry (upper triangle only), variable N, the charge-aware
  vocabulary, or (N, QED) conditioning through couplings and prior changes. No auxiliary
  objective, no surrogate, no relaxation.
* **It removes the structural blindness.** The block that decides bond orders then knows
  each endpoint's type and cap, so valence becomes a constraint it can represent instead of
  one it cannot see.
* **It costs nothing in capacity.** It reuses `TopologyBondSubnet`; the conditioner gains
  input channels, not layers.

Strengthen it with one cheap addition: feed the bond conditioner **explicit per-atom
valence caps and running bond-order sums** as extra channels on the adjacency grid. The
axial row/column aggregation stage 23 added already computes partial bond-order sums; this
makes the quantity the constraint is written in directly available instead of implicit.

### What this does not fix, stated plainly

Reversing the order makes valence *representable*, not *guaranteed*. A flow places
continuous density over a dequantized one-hot space, and argmax decoding can still land on
an over-valent configuration. Expect improvement in raw validity, not 100%. If
`argmax_shift` shows discretisation is a large part of the marginal error, the dequantisation
noise scale is the next lever, and that is a separate, smaller experiment.

### Alternative considered and rejected

A **valence-aware auxiliary loss** (penalising expected excess valence) would also inject
the constraint and would not require reordering. It is rejected as the primary remedy
because it stops the objective being pure maximum likelihood: the trained density is no
longer the data density, and the exact-likelihood property — the reason this project moved
to a flow at all — is lost. It is kept as a fallback if the reversed factorisation
under-delivers, with that tradeoff stated rather than buried.

### A separate, clearly-labelled reporting fix

The corrector's rule can be improved independently of the model: prefer the **smallest**
bond order that resolves the excess, and break ties towards bonds **not** in a ring. This
would stop the repair destroying alternation it did not need to touch. It must be reported
as what it is — a change to post-hoc correction. It does **not** improve native raw
validity, and corrected-validity numbers before and after such a change are not comparable
to each other.

### Acceptance criteria for Stage 25

Tracked for the real arm against the **shuffled-QED** control, on the same grid and seeds:

1. **native raw validity** (no chemistry applied) — must rise from 29.73% aggregate /
   7.81% at N=23;
2. **exact heavy-atom count** — reported both natively and after correction, because
   correction drops atoms; 100% corrected validity that loses atoms is not a pass;
3. **chemical validity** after sanitisation;
4. **aromaticity** — aromatic rings per molecule, read against the real-molecule ceiling
   measured in this stage, with `alternating_rings_destroyed` reported beside it;
5. **uniqueness**;
6. **QED response** — slope and within-N correlation, which count only if they separate
   from the shuffled-QED arm *and* are computed over exact-N, structurally valid molecules.

All six are already produced by `scripts/24_evaluate_graph_flow.py`. Raw numbers stay
separate from post-correction numbers throughout, and a positive result licenses a longer
run, not a deployment claim.
