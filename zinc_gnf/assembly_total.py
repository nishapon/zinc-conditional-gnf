"""Stage 21: a TOTAL, demotion-free degree-constrained graph assembly.

THE DEFECT THIS REPLACES
------------------------
Stage 20 decomposed the degree decoder's QED attenuation at N=23, T=0.2 into

    integer-programming contribution        +0.0212
    solver / fallback switching             +0.0386   <- the larger half
    total attenuation                       +0.0598   CI [+0.0348, +0.0844]

so the dominant cause is not the integer program but the DISCONTINUITY around it.
`degree_constrained_bond_decode` has five exits that abandon the program and run a
completely different algorithm - the greedy valence-aware decoder - returning its answer
instead:

    bad_degrees  no_solver  infeasible_degrees  valence_unfixable  not_connected

Two of them are reached for reasons that have nothing to do with chemistry:

  PARITY.  The requested degree sequence is `degree_logits.argmax(-1)`, taken
  INDEPENDENTLY per atom. Nothing constrains its sum to be even, no graph can realise an
  odd degree sum, so the decoder tests `wanted.sum() % 2 != 0` and gives up. Decisively:
  changing ONE atom's predicted degree by +-1 flips the parity of the sum with CERTAINTY.
  A QED request that moves a single atom's degree argmax therefore switches the entire
  assembly algorithm, and the measured "QED response" between two conditions is dominated
  by which algorithm ran rather than by what the embedding said.

  ISOLATED ATOMS.  A predicted degree of 0 is representable (`degree_head` emits
  `max_degree + 1` classes) and guarantees a disconnected graph for n >= 2, so the program
  is solved and then thrown away by the `not_connected` exit.

In both cases the degree head's information is DISCARDED in one condition and USED in the
other. That is a decoding defect, not a property of the molecules or of the flow.

THE MECHANISM IMPLEMENTED HERE
------------------------------
ONE mechanism: make the degree-constrained path TOTAL, so that no input can make it
switch algorithms. Four steps, none of which can fail, and none of which is a different
algorithm:

  1. REPAIR, don't reject, the degree sequence. Using the degree head's own softmax, move
     it to the nearest sequence a connected molecule can realise: `1 <= d_i <=
     min(n - 1, cap_i)`, `sum(d) >= 2(n - 1)`, an even sum, and Erdos-Gallai. A parity
     problem then costs one atom one step - chosen by the head's own confidence - instead
     of discarding the sequence and the algorithm with it.

  2. ONE INTEGER PROGRAM for the bond SET, maximising the decoder's summed log-odds of
     "bonded" subject to exact degrees. Because step 1 guarantees the sequence is
     graphical, this program is FEASIBLE ON THE COMPLETE CANDIDATE SET by the
     Erdos-Gallai theorem - the totality here is proved, not hoped for. Valence is
     deliberately NOT a constraint of this program: step 1 gives `d_i <= cap_i`, so
     giving every chosen bond order 1 is always within valence, which is what makes
     step 4 unconditionally feasible too.

  3. CONNECTIVITY by degree-preserving 2-swaps, with no solver. If the solution splits,
     take an edge (a, b) in one component and (c, d) in another, remove both and add
     (a, c) and (b, d). Every degree is preserved exactly, and (a, c), (b, d) cross
     components so they cannot already be bonds. Two conditions make this terminate,
     and both are established rather than assumed:

       * at least one of the two removed edges must be a NON-BRIDGE. If both are
         bridges, each component splits and the halves re-pair into two components
         again, leaving the count unchanged - an earlier version of this code oscillated
         for exactly that reason.
       * step 1 enforces `sum(d) >= 2(n - 1)`, i.e. at least n - 1 bonds. Chemically
         that is just "a connected molecule with n heavy atoms needs n - 1 bonds". A
         degree head that under-predicts gives a perfectly GRAPHICAL sequence with no
         CONNECTED realisation, and then no rewiring can succeed because the bonds are
         not there.

     Given both, a split graph has more than n - 1 edges spread over k >= 2 components,
     so some component carries a cycle and therefore a non-bridge edge: a merging swap
     always exists, and each one reduces the component count by one. The swap applied is
     the one that gives up least log-probability, so the decoder's own distribution still
     governs how the pieces are joined.

  4. ONE SMALL INTEGER PROGRAM for the bond ORDERS on the chosen edges, maximising the
     decoder's own log-probability of each order subject to the valence caps. This is
     what removes the integer-programming half of the attenuation: the existing decoder
     picks orders by an independent argmax and then DEMOTES them until valence is
     satisfied, which silently destroys the order information - and the order
     distribution is where aromaticity, and so much of QED, lives. Here no chosen order
     is ever demoted, and the program is always feasible because order 1 everywhere is.

Taken together: no fallback, no algorithm switch, no demotion, and - by steps 2 and 4 -
no failure path. The degree head's information is used in EVERY condition instead of in
half of them.

What this deliberately does NOT do: it never looks at a target QED, never consults a
property oracle, never ranks candidates by a property and never generates several
molecules to keep the best. It only re-uses the decoder's own bond and degree
distributions, so any QED response it recovers is response the conditional flow had
already encoded. The relaxation ladder below is a safety net that the feasibility
argument says should never fire; if it ever does, that is recorded on the molecule, and a
genuine failure is reported as INVALID rather than presented as a molecule.

Determinism: repair tie-breaks resolve to the lowest atom index, candidate and swap
orderings use stable sorts, and HiGHS is deterministic on identical input, so one
embedding always yields one molecule.
"""

from __future__ import annotations

import time
from typing import Any, Sequence

import numpy as np

from zinc_gnf.constants_charge_aware import ATOM_VALENCE, NUM_BOND_TYPES

RELAXATION_LEVELS = ("exact_degrees", "degrees_within_one", "degrees_free")
PROBABILITY_FLOOR = 1e-9
# Per-atom candidate partners kept in the bond-set program. Comfortably above the largest
# valence cap, so pruning alone can never starve the degree constraint.
CANDIDATE_TOP_K = 10
BOND_PROBABILITY_FLOOR = 1e-3


# --------------------------------------------------------- degree feasibility ---
def is_graphical(degrees: Sequence[int]) -> bool:
    """Erdos-Gallai: can a SIMPLE graph realise this degree sequence?

    Necessary AND sufficient, which is what step 2's feasibility argument rests on. An
    even sum alone is not enough - that is the only condition the existing decoder tests.
    """
    values = sorted((int(d) for d in degrees), reverse=True)
    n = len(values)
    if n == 0:
        return True
    if any(d < 0 or d > n - 1 for d in values):
        return False
    if sum(values) % 2 != 0:
        return False
    prefix = 0
    for k in range(1, n + 1):
        prefix += values[k - 1]
        tail = sum(min(values[i], k) for i in range(k, n))
        if prefix > k * (k - 1) + tail:
            return False
    return True


def repair_degree_sequence(degree_probabilities: np.ndarray,
                           atom_types: Sequence[int] | np.ndarray, *,
                           valence_caps: Sequence[int] = ATOM_VALENCE,
                           requested: Sequence[int] | np.ndarray | None = None
                           ) -> tuple[np.ndarray, dict[str, Any]]:
    """Move the predicted degree sequence to the nearest REALISABLE one, by its own cost.

    The existing decoder discards the sequence when its sum is odd. That is the
    discontinuity: parity flips whenever a single atom's argmax moves by one, so the
    conditioning can decide which algorithm decodes the molecule. Here the same situation
    costs one atom one step, chosen as the cheapest change under the degree head's own
    softmax, and the degree information survives in BOTH conditions.

    `degree_probabilities` is `[n, max_degree + 1]`. The bounds are
    `1 <= d_i <= min(n - 1, cap_i)`: the lower bound is what stops a predicted 0 from
    guaranteeing a disconnected graph and what gives every component an edge for the
    2-swap repair; the upper bound keeps `d_i <= cap_i`, which is what makes the order
    program unconditionally feasible.
    """
    probabilities = np.asarray(degree_probabilities, dtype=np.float64)
    atoms = np.asarray(atom_types, dtype=np.int64).reshape(-1)
    n = int(atoms.shape[0])
    if probabilities.ndim != 2 or probabilities.shape[0] != n:
        raise ValueError(f"degree probabilities {probabilities.shape} do not describe "
                         f"{n} atoms")
    caps = np.asarray([int(valence_caps[int(a)]) for a in atoms], dtype=np.int64)
    classes = probabilities.shape[1]
    base = (np.asarray(requested, dtype=np.int64).reshape(-1) if requested is not None
            else probabilities.argmax(axis=1).astype(np.int64))
    if base.shape[0] != n:
        raise ValueError("the requested degree sequence does not match the atom count")

    upper = np.minimum(np.minimum(caps, max(n - 1, 0)), classes - 1)
    lower = np.minimum(np.ones(n, dtype=np.int64) if n >= 2 else np.zeros(n, dtype=np.int64),
                       upper)
    log_probabilities = np.log(np.clip(probabilities, PROBABILITY_FLOOR, None))

    def cost(atom: int, value: int) -> float:
        """Log-probability given up by assigning this degree to this atom."""
        if value < 0 or value >= classes:
            return float("inf")
        return float(log_probabilities[atom, int(base[atom])]
                     - log_probabilities[atom, int(value)])

    info: dict[str, Any] = {"requested": base.tolist(), "changes": [],
                            "bounds_lower": lower.tolist(), "bounds_upper": upper.tolist()}
    degrees = np.clip(base, lower, upper)
    for atom in np.flatnonzero(degrees != base):
        info["changes"].append({"atom": int(atom), "from": int(base[atom]),
                                "to": int(degrees[atom]), "reason": "bounds"})
    info["clipped_atoms"] = len(info["changes"])
    info["isolated_atoms_requested"] = int((base == 0).sum())
    info["requested_sum_was_odd"] = bool(int(base.sum()) % 2 != 0)

    # A CONNECTED graph on n vertices needs at least n - 1 edges, i.e. sum(d) >= 2(n-1).
    # Chemically: a molecule with n heavy atoms needs at least n - 1 bonds to hold
    # together. A degree head that under-predicts - mostly 1s and 2s - produces a
    # perfectly graphical sequence for which NO connected realisation exists, and then no
    # amount of rewiring can connect the graph because the edges simply are not there.
    # Raising the cheapest degrees is what makes the connectivity guarantee real.
    needed = 2 * (n - 1) if n >= 2 else 0
    info["edges_needed_for_connectivity"] = needed // 2
    info["requested_edges"] = int(base.sum()) // 2
    raised = 0
    while n >= 2 and degrees.sum() < needed:
        candidates = [(cost(a, int(degrees[a]) + 1), a) for a in range(n)
                      if degrees[a] < upper[a]]
        if not candidates:
            break                      # the valence caps forbid a connected molecule
        candidates.sort(key=lambda item: (item[0], item[1]))
        _, atom = candidates[0]
        info["changes"].append({"atom": int(atom), "from": int(degrees[atom]),
                                "to": int(degrees[atom]) + 1, "reason": "connectivity"})
        degrees[atom] += 1
        raised += 1
    info["degrees_raised_for_connectivity"] = raised
    info["connectivity_possible"] = bool(n < 2 or degrees.sum() >= needed)

    # parity: exactly one atom moves one step, the cheapest such move. An increase is
    # preferred so the connectivity bound above cannot be undone by the parity fix.
    if n >= 2 and degrees.sum() % 2 != 0:
        best: tuple[float, int, int] | None = None
        for prefer_increase in (True, False):
            for atom in range(n):
                for delta in ((1,) if prefer_increase else (-1,)):
                    value = int(degrees[atom]) + delta
                    if value < lower[atom] or value > upper[atom]:
                        continue
                    if delta < 0 and degrees.sum() - 1 < needed:
                        continue           # never undo the connectivity bound
                    penalty = cost(atom, value)
                    if best is None or penalty < best[0] - 1e-15:
                        best = (penalty, atom, value)
            if best is not None:
                break
        if best is not None:
            _, atom, value = best
            info["changes"].append({"atom": int(atom), "from": int(degrees[atom]),
                                    "to": int(value), "reason": "parity"})
            degrees[atom] = value
        info["parity_repaired"] = best is not None
    else:
        info["parity_repaired"] = False

    # Erdos-Gallai, in pairs so the sum stays even. Essentially never reached for organic
    # valence caps, but it is the difference between a repair and a wrong answer.
    guard = 0
    while n >= 2 and not is_graphical(degrees) and guard < 2 * n:
        guard += 1
        reducible = [a for a in range(n) if degrees[a] > lower[a]]
        if len(reducible) < 2 or degrees.sum() - 2 < needed:
            break
        reducible.sort(key=lambda a: (-int(degrees[a]), cost(a, int(degrees[a]) - 1), a))
        for atom in reducible[:2]:
            info["changes"].append({"atom": int(atom), "from": int(degrees[atom]),
                                    "to": int(degrees[atom]) - 1,
                                    "reason": "erdos_gallai"})
            degrees[atom] -= 1
    info["graphical"] = bool(is_graphical(degrees))
    info["degrees"] = degrees.tolist()
    info["total_changes"] = len(info["changes"])
    info["atoms_changed"] = len({change["atom"] for change in info["changes"]})
    info["log_probability_cost"] = float(
        sum(cost(int(a), int(degrees[a])) for a in range(n)))
    return degrees, info


# ------------------------------------------------------------------- the solver ---
def _components(bond: np.ndarray) -> list[list[int]]:
    n = bond.shape[0]
    seen = [False] * n
    found = []
    for start in range(n):
        if seen[start]:
            continue
        stack, group = [start], []
        seen[start] = True
        while stack:
            node = stack.pop()
            group.append(node)
            for other in range(n):
                if not seen[other] and bond[node, other] > 0:
                    seen[other] = True
                    stack.append(other)
        found.append(sorted(group))
    return found


def _milp(cost: np.ndarray, triplets: list[tuple[int, int, float]], rows: int,
          lower: np.ndarray, upper: np.ndarray, variables: int,
          time_limit: float | None):
    """One HiGHS call on a sparse constraint matrix."""
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import coo_matrix

    constraints = []
    if rows and triplets:
        data = np.asarray([v for _, _, v in triplets], dtype=np.float64)
        row_index = np.asarray([r for r, _, _ in triplets], dtype=np.int64)
        column_index = np.asarray([c for _, c, _ in triplets], dtype=np.int64)
        matrix = coo_matrix((data, (row_index, column_index)),
                            shape=(rows, variables)).tocsr()
        constraints = [LinearConstraint(matrix, lower, upper)]
    options = {"time_limit": float(time_limit)} if time_limit else None
    return milp(c=cost, constraints=constraints, integrality=np.ones(variables),
                bounds=Bounds(0, 1), options=options)


def _connect_by_swaps(bond: np.ndarray, log_any: np.ndarray
                      ) -> tuple[np.ndarray, dict[str, Any]]:
    """Merge components with degree-preserving 2-swaps. Always succeeds.

    Remove (a, b) from one component and (c, d) from another, add (a, c) and (b, d).
    Degrees are preserved exactly. Every atom has degree >= 1, so every component holds an
    edge; and (a, c), (b, d) span different components, so they cannot already be bonds.
    Each swap merges two components, so (components - 1) swaps connect the graph.

    The swap applied is the one that gives up least `log P(bonded)`, so the decoder's own
    distribution - not an arbitrary choice - decides how the pieces are joined.
    """
    bond = bond.copy()
    record: dict[str, Any] = {"swaps": 0, "log_probability_cost": 0.0,
                              "components_before": len(_components(bond)),
                              "incomplete": False}

    def still_connected_without(component: Sequence[int], u: int, v: int) -> bool:
        """Is (u, v) a NON-bridge inside its component?"""
        inside = set(int(x) for x in component)
        stack, seen = [int(u)], {int(u)}
        while stack:
            node = stack.pop()
            for other in inside:
                if other in seen or bond[node, other] <= 0:
                    continue
                if {node, other} == {int(u), int(v)}:
                    continue                      # the edge under test is removed
                seen.add(other)
                stack.append(other)
        return int(v) in seen

    # Operates on the unit-order bond SET, before orders are assigned, so a 2-swap leaves
    # every DEGREE unchanged and cannot disturb any atom's total bond order.
    #
    # A swap merges two components only if at least one of the two removed edges is a
    # NON-BRIDGE: if both are bridges, each component splits and the two halves re-pair
    # into two components again, leaving the count unchanged (which is what made an
    # earlier version oscillate). With sum(d) >= 2(n-1) guaranteed by the repair there are
    # at least n-1 edges, so whenever the graph is split some component must carry a
    # cycle and therefore a non-bridge edge - so a merging swap always exists.
    for _ in range(bond.shape[0] + 1):
        groups = _components(bond)
        if len(groups) <= 1:
            break
        best: tuple[float, int, int, int, int, int, int, int, int] | None = None
        for index, component in enumerate(groups):
            edges = [(a, b) for a in component for b in component
                     if a < b and bond[a, b] > 0]
            for a, b in edges:
                if not still_connected_without(component, a, b):
                    continue                      # a bridge: cannot anchor the merge
                for other_index, other in enumerate(groups):
                    if other_index == index:
                        continue
                    for c, d in ((c, d) for c in other for d in other
                                 if c < d and bond[c, d] > 0):
                        for (p, q), (r, s) in (((a, c), (b, d)), ((a, d), (b, c))):
                            penalty = float(log_any[a, b] + log_any[c, d]
                                            - log_any[p, q] - log_any[r, s])
                            if best is None or penalty < best[0] - 1e-15:
                                best = (penalty, a, b, c, d, p, q, r, s)
        if best is None:
            # No non-bridge anywhere: there are too few bonds for any connected graph on
            # these atoms, which the repair reports as connectivity_possible = False.
            record["incomplete"] = True
            break
        penalty, a, b, c, d, p, q, r, s = best
        bond[a, b] = bond[b, a] = 0
        bond[c, d] = bond[d, c] = 0
        bond[p, q] = bond[q, p] = 1
        bond[r, s] = bond[s, r] = 1
        record["swaps"] += 1
        record["log_probability_cost"] += penalty
    record["components_after"] = len(_components(bond))
    return bond, record


def total_degree_bond_decode(atom_types: Sequence[int] | np.ndarray,
                             bond_probabilities: np.ndarray,
                             degree_probabilities: np.ndarray, *,
                             valence_caps: Sequence[int] = ATOM_VALENCE,
                             degrees: Sequence[int] | np.ndarray | None = None,
                             candidate_top_k: int | None = CANDIDATE_TOP_K,
                             bond_probability_floor: float = BOND_PROBABILITY_FLOOR,
                             time_limit: float | None = 5.0
                             ) -> tuple[np.ndarray, dict[str, Any]]:
    """Total degree-constrained assembly: repair, set program, 2-swaps, order program.

    Returns `(bond_matrix, info)`. `info["fallback"]` is always False - there is no other
    algorithm to fall back to, which is the whole point - and `info` records the
    relaxation level, the degree repair, the swap count and the solver cost, so every
    claim about this decoder can be checked per molecule.
    """
    atoms = np.asarray(atom_types, dtype=np.int64).reshape(-1)
    probabilities = np.asarray(bond_probabilities, dtype=np.float64)
    n = int(atoms.shape[0])
    started = time.perf_counter()
    if probabilities.shape != (n, n, NUM_BOND_TYPES):
        raise ValueError(f"bond_probabilities shape {probabilities.shape} != "
                         f"({n}, {n}, {NUM_BOND_TYPES})")
    if not np.isfinite(probabilities).all():
        raise ValueError("bond probabilities contain non-finite values")

    caps = np.asarray([int(valence_caps[int(a)]) for a in atoms], dtype=np.int64)
    info: dict[str, Any] = {"fallback": False, "fallback_reason": None, "rescue_edges": 0,
                            "assembly_failed": False, "decoder": "total",
                            "set_solver_calls": 0, "order_solver_calls": 0}
    if n < 2:
        info.update({"connected": True, "valence_used": 0, "swaps": 0,
                     "relaxation_level": RELAXATION_LEVELS[0], "degree_satisfied": True,
                     "degree_absolute_error": 0, "valence_respected": True,
                     "pairs_considered": 0, "pairs_total": 0, "set_objective": 0.0,
                     "order_objective": 0.0, "seconds": time.perf_counter() - started})
        return np.zeros((n, n), dtype=np.int64), info

    wanted, degree_info = repair_degree_sequence(
        degree_probabilities, atoms, valence_caps=valence_caps, requested=degrees)
    info["degree_repair"] = degree_info

    symmetric = np.clip(0.5 * (probabilities + probabilities.transpose(1, 0, 2)),
                        PROBABILITY_FLOOR, None)
    log_probabilities = np.log(symmetric)
    any_bond = np.clip(1.0 - symmetric[:, :, 0], PROBABILITY_FLOOR, 1.0 - PROBABILITY_FLOOR)
    log_any = np.log(any_bond)
    # log-odds of "bonded", the same quantity the existing decoder maximises, so the two
    # bond SETS are chosen on identical grounds and only the surrounding totality differs
    gain = np.log(any_bond) - np.log1p(-any_bond)
    ceiling = np.minimum(caps, n - 1).astype(float)

    def candidate_pairs(top_k: int | None) -> list[tuple[int, int]]:
        """Plausible pairs, with a per-atom coverage guarantee.

        A restriction of the feasible set, so it carries a guarantee and is recorded:
        every atom keeps its `top_k` most probable partners - at least its largest
        admissible degree plus slack - and any pair above `bond_probability_floor`
        survives. If the program is infeasible the candidate set is EXPANDED to the
        complete pair set before any chemical constraint is touched, and on the complete
        set Erdos-Gallai guarantees feasibility.
        """
        if top_k is None:
            return [(i, j) for i in range(n) for j in range(i + 1, n)]
        keep = any_bond >= float(bond_probability_floor)
        reach = int(min(max(int(top_k), int(ceiling.max()) + 1), n - 1))
        for i in range(n):
            row = any_bond[i].copy()
            row[i] = -1.0
            for j in np.argsort(-row, kind="stable")[:reach]:
                keep[i, int(j)] = keep[int(j), i] = True
        np.fill_diagonal(keep, False)
        return [(i, j) for i in range(n) for j in range(i + 1, n) if keep[i, j]]

    def solve_set(pair_list: list[tuple[int, int]], level: str):
        """The bond SET: maximise summed log-odds of bonded subject to the degrees."""
        variables = len(pair_list)
        if variables == 0:
            return None
        cost = -np.asarray([gain[i, j] for i, j in pair_list], dtype=np.float64)
        triplets = []
        for column, (i, j) in enumerate(pair_list):
            triplets.append((i, column, 1.0))
            triplets.append((j, column, 1.0))
        if level == "exact_degrees":
            lower, upper = wanted.astype(float), wanted.astype(float)
        elif level == "degrees_within_one":
            lower = np.maximum(wanted - 1, 1).astype(float)
            upper = np.minimum(wanted + 1, ceiling)
        else:
            lower, upper = np.ones(n), ceiling
        result = _milp(cost, triplets, n, lower, upper, variables, time_limit)
        info["set_solver_calls"] += 1
        if result.x is None:
            return None
        chosen = np.asarray(result.x) > 0.5
        bond = np.zeros((n, n), dtype=np.int64)
        for column, (i, j) in enumerate(pair_list):
            if chosen[column]:
                bond[i, j] = bond[j, i] = 1
        return bond, float(sum(gain[i, j] for (i, j), take
                               in zip(pair_list, chosen) if take)), len(pair_list)

    def solve_orders(bond: np.ndarray):
        """The bond ORDERS on the chosen edges, under valence. Replaces demotion.

        Always feasible: every atom's degree is at most its cap after the repair, so
        order 1 on every edge already satisfies valence. Nothing is ever demoted.
        """
        edges = [(i, j) for i in range(n) for j in range(i + 1, n) if bond[i, j] > 0]
        if not edges:
            return bond, 0.0
        orders = list(range(1, NUM_BOND_TYPES))
        columns = [(k, order) for k in range(len(edges)) for order in orders]
        objective = np.asarray([log_probabilities[edges[k][0], edges[k][1], order]
                                for k, order in columns], dtype=np.float64)
        triplets = []
        for column, (k, _) in enumerate(columns):
            triplets.append((k, column, 1.0))          # exactly one order per edge
        offset = len(edges)
        for column, (k, order) in enumerate(columns):
            i, j = edges[k]
            triplets.append((offset + i, column, float(order)))
            triplets.append((offset + j, column, float(order)))
        lower = np.concatenate([np.ones(len(edges)), np.zeros(n)])
        upper = np.concatenate([np.ones(len(edges)), caps.astype(float)])
        result = _milp(-objective, triplets, offset + n, lower, upper, len(columns),
                       time_limit)
        info["order_solver_calls"] += 1
        if result.x is None:
            return bond, float("nan")
        chosen = np.asarray(result.x) > 0.5
        assembled = np.zeros((n, n), dtype=np.int64)
        for column, (k, order) in enumerate(columns):
            if chosen[column]:
                i, j = edges[k]
                assembled[i, j] = assembled[j, i] = order
        return assembled, float(objective[chosen].sum())

    stages: list[tuple[int | None, str]] = [(candidate_top_k, "exact_degrees")]
    if candidate_top_k is not None:
        stages.append((None, "exact_degrees"))
    stages += [(None, "degrees_within_one"), (None, "degrees_free")]

    cache: dict[Any, list[tuple[int, int]]] = {}
    for top_k, level in stages:
        if top_k not in cache:
            cache[top_k] = candidate_pairs(top_k)
        outcome = solve_set(cache[top_k], level)
        if outcome is None:
            continue
        bond, set_objective, considered = outcome
        bond, swap_record = _connect_by_swaps(bond, log_any)
        bond, order_objective = solve_orders(bond)
        achieved = (bond > 0).sum(axis=1)
        connected = len(_components(bond)) == 1
        info.update({
            "relaxation_level": level, "pairs_considered": considered,
            "pairs_total": n * (n - 1) // 2, "candidate_top_k": top_k,
            "set_objective": set_objective, "order_objective": order_objective,
            "connectivity_repair": swap_record, "swaps": swap_record["swaps"],
            "degree_satisfied": bool(np.array_equal(achieved, wanted)),
            "degree_absolute_error": int(np.abs(achieved - wanted).sum()),
            "connected": bool(connected),
            "valence_used": int(bond.sum()),
            "valence_respected": bool((bond.sum(axis=1) <= caps).all()),
            "seconds": time.perf_counter() - started,
        })
        if not connected or not info["valence_respected"]:
            # The feasibility argument says this is unreachable. If it is ever reached,
            # the molecule is reported as a failed assembly, never dressed up as valid.
            info.update({"assembly_failed": True})
            return np.zeros((n, n), dtype=np.int64), info
        return bond, info

    info.update({"assembly_failed": True, "connected": False, "valence_used": 0,
                 "relaxation_level": None, "degree_satisfied": False, "swaps": 0,
                 "degree_absolute_error": int(wanted.sum()), "valence_respected": True,
                 "set_objective": float("nan"), "order_objective": float("nan"),
                 "pairs_total": n * (n - 1) // 2, "candidate_top_k": candidate_top_k,
                 "seconds": time.perf_counter() - started})
    return np.zeros((n, n), dtype=np.int64), info


__all__ = [
    "BOND_PROBABILITY_FLOOR",
    "CANDIDATE_TOP_K",
    "RELAXATION_LEVELS",
    "is_graphical",
    "repair_degree_sequence",
    "total_degree_bond_decode",
]
