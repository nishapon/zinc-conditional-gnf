"""Phase 0 chemistry diagnostics for ZINC250K.

Purpose: measure, on the real CSV, how the existing neutral-only encoding
rules treat the dataset, WITHOUT changing those rules. Nothing in this module
removes molecules, edits the CSV, or writes to the production processed-data
directory. Every CSV row is counted.

What is measured
----------------
* Total records, parse failures, and per-reason rejection counts. Rejection
  reasons come straight from `zinc_gnf.chemistry.encode_zinc_smiles`, so they
  are exactly what the production preprocessing would report.
* A formal-charge census over every parseable molecule: net-charge classes,
  per-atom charges, charge-typed atom frequencies such as `N+1` / `O-1`,
  charged-atom signatures (e.g. the nitro pattern `N+1|O-1`).
* Charged versus neutral molecule counts and their cross-tabulation against
  rejection reasons (the first failing rule wins, so "charged" and
  "non_neutral" are not identical).
* CSV QED versus RDKit QED discrepancies, for all parseable molecules, for
  charged/uncharged subsets, and for accepted molecules (QED recomputed on the
  reconstructed molecule, as production does).
* All molecules versus accepted neutral-only molecules: QED, heavy-atom count,
  aromatic-atom fraction, logP, SAS, element composition, with KS and
  Wasserstein distances. This quantifies the selection bias of the filter.
* A DIAGNOSTIC-ONLY counterfactual: if formal charges were naively stripped
  (as MoFlow does when it keeps only atomic numbers), how many charged
  molecules would the existing rules then accept, why would the rest still
  fail (e.g. nitro groups need their charge to stay valence-valid), and how
  would the QED label shift? This changes no pipeline behaviour.

Statements about MoFlow in comments refer to its public code
(`data/smile_to_graph.py` keeps `GetAtomicNum()` only).
"""

from __future__ import annotations

import csv
from array import array
from collections import Counter
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import random
import time
from typing import Any, Sequence

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import QED

from zinc_gnf.chemistry import encode_zinc_smiles
from zinc_gnf.constants import (
    ATOM_VOCAB,
    MAX_NODES,
    MIN_NODES,
    REPRESENTATION_VERSION,
)
from zinc_gnf.preprocessing import hash_file
from zinc_gnf.utils.runtime import (
    StopFlag,
    assert_output_outside,
    atomic_write_json,
    collect_environment,
    current_rss_mb,
    format_duration,
    peak_rss_mb,
)


# ---------------------------------------------------------------------------
# Dataset identity (provided by the project owner).
# ---------------------------------------------------------------------------
EXPECTED_ZINC250K_SHA256 = (
    "35e3f1a52b1badc0697e373d73a18ad773f415936ff992f4c6baa2e067b3e6ae"
)
EXPECTED_ZINC250K_ROWS = 249_455
REQUIRED_COLUMNS = ("smiles", "logP", "qed", "SAS")

SCHEMA_VERSION = 1
SKIP_HASH_CHECK = "skip"

# Metrics compared between groups, and the groups themselves.
METRICS = (
    "qed_csv", "qed_rdkit", "n_atoms", "aromatic_fraction", "logp_csv",
    "sas_csv",
)
GROUPS = ("all", "accepted", "rejected", "charged")
DELTA_THRESHOLDS = (1e-4, 1e-3, 1e-2, 1e-1)


class DatasetIdentityError(RuntimeError):
    """The CSV on disk is not the file the diagnostics were designed for."""


# ---------------------------------------------------------------------------
# Small helpers.
# ---------------------------------------------------------------------------
def _to_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number


def atom_type_key(symbol: str, formal_charge: int) -> str:
    """Charge-typed atom label: `C`, `N+1`, `O-1`, `N+2`."""
    if formal_charge == 0:
        return symbol
    return f"{symbol}{formal_charge:+d}"


def deterministic_sample_indices(
    n_rows: int, sample_size: int, seed: int,
) -> list[int]:
    """Sorted, seeded, without-replacement sample of row indices.

    Sorting keeps CSV order, so a sample is a reproducible subsequence of the
    file. If `sample_size >= n_rows` every row is returned.
    """
    if n_rows < 0 or sample_size < 0:
        raise ValueError("n_rows and sample_size must be non-negative")
    if sample_size >= n_rows:
        return list(range(n_rows))
    rng = random.Random(seed)
    return sorted(rng.sample(range(n_rows), sample_size))


def indices_fingerprint(indices: Sequence[int]) -> str:
    """Short hash of a sample's row indices, to prove two runs saw the same rows."""
    digest = hashlib.sha256(
        ",".join(str(int(i)) for i in indices).encode("ascii")
    )
    return digest.hexdigest()[:16]


# ---------------------------------------------------------------------------
# CSV reading that keeps EVERY record.
#
# `preprocessing.read_zinc_csv` silently skips rows with an empty SMILES.
# Diagnostics must account for all records, so this reader keeps them (an
# empty SMILES is later reported by the encoder as `empty_smiles`). It uses
# csv.DictReader because the ZINC250K file stores each SMILES with an
# embedded trailing newline inside quotes (see preprocessing.py).
# ---------------------------------------------------------------------------
@dataclass
class ZincTable:
    smiles: list[str]
    qed: array
    logp: array
    sas: array
    nonnumeric_qed_rows: int
    empty_smiles_rows: int

    def __len__(self) -> int:
        return len(self.smiles)


def read_zinc_table(csv_path: str | Path) -> ZincTable:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"ZINC250K CSV not found at {csv_path}. Compute nodes have no "
            "internet access; download it on the login node first "
            "(see slurm/README_phase0.md)."
        )

    smiles: list[str] = []
    qed_values = array("d")
    logp_values = array("d")
    sas_values = array("d")
    nonnumeric_qed = 0
    empty_smiles = 0

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV at {csv_path} is empty or malformed")
        missing = set(REQUIRED_COLUMNS) - set(reader.fieldnames)
        if missing:
            raise ValueError(
                f"CSV at {csv_path} is missing columns {sorted(missing)}; "
                f"found {reader.fieldnames}"
            )
        for row in reader:
            text = str(row["smiles"] if row["smiles"] is not None else "")
            text = text.strip()
            if not text:
                empty_smiles += 1
            qed_value = _to_float(row["qed"])
            if not math.isfinite(qed_value):
                nonnumeric_qed += 1
            smiles.append(text)
            qed_values.append(qed_value)
            logp_values.append(_to_float(row["logP"]))
            sas_values.append(_to_float(row["SAS"]))

    return ZincTable(
        smiles=smiles, qed=qed_values, logp=logp_values, sas=sas_values,
        nonnumeric_qed_rows=nonnumeric_qed, empty_smiles_rows=empty_smiles,
    )


# ---------------------------------------------------------------------------
# Diagnostic-only charge stripping.
#
# Mirrors the *effect* of MoFlow's atomic-number-only encoding: every formal
# charge is set to zero and explicit hydrogens on charged atoms are dropped so
# RDKit re-infers them from valence ([NH3+] -> NH2, [O-] -> OH). It does NOT
# apply MoFlow's decode-time "+1 charge on over-valent N/O/S" rule, so
# molecules that need their charge to stay valence-valid (nitro, quaternary
# ammonium, N-oxides) fail and are reported as such.
# ---------------------------------------------------------------------------
STRIP_SANITIZE_FAILURE = "strip_sanitize_failure"


def strip_formal_charges(molecule: Chem.Mol) -> Chem.Mol:
    """Return a sanitized copy with all formal charges removed, or raise."""
    editable = Chem.RWMol(molecule)
    for atom in editable.GetAtoms():
        if atom.GetFormalCharge() != 0:
            atom.SetFormalCharge(0)
            atom.SetNumExplicitHs(0)
            atom.SetNoImplicit(False)
            atom.SetNumRadicalElectrons(0)
    stripped = editable.GetMol()
    with rdBase.BlockLogs():
        Chem.SanitizeMol(stripped)
    return stripped


# ---------------------------------------------------------------------------
# Summary statistics.
# ---------------------------------------------------------------------------
def describe(values: Sequence[float] | np.ndarray) -> dict[str, Any]:
    data = np.asarray(values, dtype=np.float64)
    data = data[np.isfinite(data)]
    if data.size == 0:
        return {"n": 0}
    quantiles = np.quantile(data, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    return {
        "n": int(data.size),
        "mean": float(data.mean()),
        "std": float(data.std()),
        "min": float(data.min()),
        "q01": float(quantiles[0]),
        "q05": float(quantiles[1]),
        "q25": float(quantiles[2]),
        "median": float(quantiles[3]),
        "q75": float(quantiles[4]),
        "q95": float(quantiles[5]),
        "q99": float(quantiles[6]),
        "max": float(data.max()),
    }


def delta_statistics(
    csv_values: Sequence[float] | np.ndarray,
    other_values: Sequence[float] | np.ndarray,
) -> dict[str, Any]:
    """Statistics of `other - csv` over rows where both are finite."""
    csv_array = np.asarray(csv_values, dtype=np.float64)
    other_array = np.asarray(other_values, dtype=np.float64)
    mask = np.isfinite(csv_array) & np.isfinite(other_array)
    if not mask.any():
        return {"molecules_compared": 0}
    signed = other_array[mask] - csv_array[mask]
    absolute = np.abs(signed)
    result: dict[str, Any] = {
        "molecules_compared": int(mask.sum()),
        "mean_signed_delta": float(signed.mean()),
        "mean_abs_delta": float(absolute.mean()),
        "median_abs_delta": float(np.median(absolute)),
        "p99_abs_delta": float(np.quantile(absolute, 0.99)),
        "max_abs_delta": float(absolute.max()),
    }
    for threshold in DELTA_THRESHOLDS:
        label = f"{threshold:.0e}".replace("e-0", "e-")
        result[f"within_{label}"] = int((absolute <= threshold).sum())
    result["above_1e-1"] = int((absolute > 0.1).sum())
    return result


def histogram(
    values: Sequence[float] | np.ndarray,
    *,
    low: float,
    high: float,
    bins: int,
) -> dict[str, Any]:
    data = np.asarray(values, dtype=np.float64)
    data = data[np.isfinite(data)]
    counts, edges = np.histogram(data, bins=bins, range=(low, high))
    return {
        "bin_edges": edges.tolist(),
        "counts": counts.astype(int).tolist(),
        "n": int(data.size),
        "outside_range": int(data.size - counts.sum()),
    }


def total_variation_distance(
    left: dict[str, float], right: dict[str, float],
) -> float:
    keys = set(left) | set(right)
    return 0.5 * float(
        sum(abs(left.get(key, 0.0) - right.get(key, 0.0)) for key in keys)
    )


def _fractions(counter: Counter) -> dict[str, float]:
    total = sum(counter.values())
    if total == 0:
        return {}
    return {key: value / total for key, value in counter.most_common()}


# ---------------------------------------------------------------------------
# The census.
# ---------------------------------------------------------------------------
class ChemistryCensus:
    """Accumulates per-row chemistry observations; builds the JSON report.

    `add` performs, per CSV row:
      1. the existing encoder (authoritative rejection reason and accepted
         record),
      2. an independent RDKit parse for the charge/atom census,
      3. for charged molecules only, the diagnostic charge-strip counterfactual.
    """

    def __init__(
        self,
        *,
        min_nodes: int = MIN_NODES,
        max_nodes: int = MAX_NODES,
        counterfactual: bool = True,
    ) -> None:
        self.min_nodes = int(min_nodes)
        self.max_nodes = int(max_nodes)
        self.counterfactual = bool(counterfactual)

        self.total_rows = 0
        self.parse_failures = 0
        self.empty_smiles = 0
        self.rejection_counts: Counter = Counter()
        self.rejected_charged_by_reason: Counter = Counter()
        self.accepted_rows = 0

        # Molecule-level charge census (parseable molecules only).
        self.molecule_class: Counter = Counter()
        self.net_charge: Counter = Counter()
        self.charged_atoms_per_molecule: Counter = Counter()
        self.charge_signature: Counter = Counter()
        self.molecules_with_isotope = 0
        self.molecules_with_radical = 0
        self.multi_fragment_molecules = 0

        # Atom-level census.
        self.atom_charge: Counter = Counter()
        self.atom_type_all: Counter = Counter()
        self.molecules_with_atom_type_all: Counter = Counter()
        self.atom_type_accepted: Counter = Counter()
        self.molecules_with_atom_type_accepted: Counter = Counter()
        self.element_all: Counter = Counter()
        self.element_accepted: Counter = Counter()
        self.out_of_vocab_elements: Counter = Counter()

        # Per-group metric samples.
        self._values: dict[str, dict[str, array]] = {
            group: {metric: array("d") for metric in METRICS}
            for group in GROUPS
        }

        # Paired QED samples for discrepancy statistics and scatter plots.
        self._pairs: dict[str, tuple[array, array]] = {
            name: (array("d"), array("d"))
            for name in (
                "all_parseable", "charged", "uncharged",
                "accepted_reconstructed", "charge_stripped_recovered",
            )
        }

        # Duplicate detection among accepted molecules.
        self._accepted_canonical: set[str] = set()

        # Counterfactual (diagnostic only).
        self.cf_charged = 0
        self.cf_recovered = 0
        self.cf_failures: Counter = Counter()
        self.cf_signature_total: Counter = Counter()
        self.cf_signature_recovered: Counter = Counter()
        self.cf_recovered_nodes = array("d")

        # Time accounting (seconds).
        self.seconds_encode = 0.0
        self.seconds_census = 0.0
        self.seconds_counterfactual = 0.0

    # ------------------------------------------------------------------
    def _push(self, group: str, metric: str, value: float) -> None:
        if math.isfinite(value):
            self._values[group][metric].append(value)

    def _push_pair(self, name: str, csv_value: float, other: float) -> None:
        if math.isfinite(csv_value) and math.isfinite(other):
            csv_store, other_store = self._pairs[name]
            csv_store.append(csv_value)
            other_store.append(other)

    # ------------------------------------------------------------------
    def add(
        self,
        smiles: str,
        csv_qed: float,
        csv_logp: float,
        csv_sas: float,
    ) -> None:
        self.total_rows += 1

        # CSV-derived metrics exist for every row, parseable or not.
        self._push("all", "qed_csv", csv_qed)
        self._push("all", "logp_csv", csv_logp)
        self._push("all", "sas_csv", csv_sas)

        # 1. Authoritative encoding result (existing rules, unchanged).
        started = time.perf_counter()
        record, reason = encode_zinc_smiles(
            smiles, min_nodes=self.min_nodes, max_nodes=self.max_nodes,
        )
        self.seconds_encode += time.perf_counter() - started

        # 2. Independent parse for the census.
        started = time.perf_counter()
        molecule = None
        if smiles:
            with rdBase.BlockLogs():
                molecule = Chem.MolFromSmiles(smiles)

        charged = False
        signature = ""
        n_atoms = 0
        aromatic_fraction = float("nan")
        rdkit_qed_original = float("nan")

        if molecule is None:
            if smiles:
                self.parse_failures += 1
            else:
                self.empty_smiles += 1
        else:
            n_atoms = molecule.GetNumAtoms()
            keys: list[str] = []
            charged_keys: list[str] = []
            net = 0
            aromatic = 0
            has_isotope = False
            has_radical = False
            for atom in molecule.GetAtoms():
                charge = atom.GetFormalCharge()
                symbol = atom.GetSymbol()
                key = atom_type_key(symbol, charge)
                keys.append(key)
                self.atom_charge[charge] += 1
                self.element_all[symbol] += 1
                if symbol not in ATOM_VOCAB:
                    self.out_of_vocab_elements[symbol] += 1
                if charge != 0:
                    charged_keys.append(key)
                    net += charge
                if atom.GetIsAromatic():
                    aromatic += 1
                if atom.GetIsotope() != 0:
                    has_isotope = True
                if atom.GetNumRadicalElectrons() != 0:
                    has_radical = True
            self.atom_type_all.update(keys)
            self.molecules_with_atom_type_all.update(set(keys))
            charged = bool(charged_keys)
            signature = "|".join(sorted(charged_keys))
            aromatic_fraction = aromatic / n_atoms if n_atoms else float("nan")

            if charged:
                if net == 0:
                    self.molecule_class["charged_net_neutral"] += 1
                elif net > 0:
                    self.molecule_class["net_cation"] += 1
                else:
                    self.molecule_class["net_anion"] += 1
                self.charge_signature[signature] += 1
            else:
                self.molecule_class["neutral"] += 1
            self.net_charge[net] += 1
            self.charged_atoms_per_molecule[len(charged_keys)] += 1
            self.molecules_with_isotope += int(has_isotope)
            self.molecules_with_radical += int(has_radical)
            if len(Chem.GetMolFrags(molecule)) != 1:
                self.multi_fragment_molecules += 1

            try:
                rdkit_qed_original = float(QED.qed(molecule))
            except Exception:  # noqa: BLE001 - RDKit raises varied C++ errors.
                rdkit_qed_original = float("nan")

            self._push("all", "qed_rdkit", rdkit_qed_original)
            self._push("all", "n_atoms", float(n_atoms))
            self._push("all", "aromatic_fraction", aromatic_fraction)
            self._push_pair("all_parseable", csv_qed, rdkit_qed_original)
            subset = "charged" if charged else "uncharged"
            self._push_pair(subset, csv_qed, rdkit_qed_original)

            if charged:
                for metric, value in (
                    ("qed_csv", csv_qed), ("logp_csv", csv_logp),
                    ("sas_csv", csv_sas), ("qed_rdkit", rdkit_qed_original),
                    ("n_atoms", float(n_atoms)),
                    ("aromatic_fraction", aromatic_fraction),
                ):
                    self._push("charged", metric, value)
        self.seconds_census += time.perf_counter() - started

        # 3. Bookkeeping that depends on the encoder's verdict.
        if record is None:
            reason = reason or "unknown"
            self.rejection_counts[reason] += 1
            if charged:
                self.rejected_charged_by_reason[reason] += 1
            for metric, value in (
                ("qed_csv", csv_qed), ("logp_csv", csv_logp),
                ("sas_csv", csv_sas),
            ):
                self._push("rejected", metric, value)
            if molecule is not None:
                self._push("rejected", "qed_rdkit", rdkit_qed_original)
                self._push("rejected", "n_atoms", float(n_atoms))
                self._push("rejected", "aromatic_fraction", aromatic_fraction)
        else:
            self.accepted_rows += 1
            self._accepted_canonical.add(record["smiles"])
            self.element_accepted.update(
                ATOM_VOCAB[int(index)] for index in record["atom_types"]
            )
            accepted_keys = [ATOM_VOCAB[int(i)] for i in record["atom_types"]]
            self.atom_type_accepted.update(accepted_keys)
            self.molecules_with_atom_type_accepted.update(set(accepted_keys))
            self._push("accepted", "qed_csv", csv_qed)
            self._push("accepted", "logp_csv", csv_logp)
            self._push("accepted", "sas_csv", csv_sas)
            self._push("accepted", "qed_rdkit", float(record["qed"]))
            self._push("accepted", "n_atoms", float(record["n_node"]))
            self._push("accepted", "aromatic_fraction", aromatic_fraction)
            self._push_pair(
                "accepted_reconstructed", csv_qed, float(record["qed"]),
            )

        # 4. Diagnostic-only counterfactual for charged molecules.
        if self.counterfactual and charged and molecule is not None:
            started = time.perf_counter()
            self._run_counterfactual(molecule, signature, csv_qed)
            self.seconds_counterfactual += time.perf_counter() - started

    # ------------------------------------------------------------------
    def _run_counterfactual(
        self, molecule: Chem.Mol, signature: str, csv_qed: float,
    ) -> None:
        self.cf_charged += 1
        self.cf_signature_total[signature] += 1
        try:
            stripped = strip_formal_charges(molecule)
            stripped_smiles = Chem.MolToSmiles(stripped)
        except Exception:  # noqa: BLE001
            self.cf_failures[STRIP_SANITIZE_FAILURE] += 1
            return

        record, reason = encode_zinc_smiles(
            stripped_smiles, min_nodes=self.min_nodes, max_nodes=self.max_nodes,
        )
        if record is None:
            self.cf_failures[reason or "unknown"] += 1
            return

        self.cf_recovered += 1
        self.cf_signature_recovered[signature] += 1
        self.cf_recovered_nodes.append(float(record["n_node"]))
        self._push_pair(
            "charge_stripped_recovered", csv_qed, float(record["qed"]),
        )

    # ------------------------------------------------------------------
    def qed_pairs(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        csv_store, other_store = self._pairs[name]
        return (
            np.frombuffer(csv_store, dtype=np.float64).copy(),
            np.frombuffer(other_store, dtype=np.float64).copy(),
        )

    def values(self, group: str, metric: str) -> np.ndarray:
        return np.frombuffer(
            self._values[group][metric], dtype=np.float64,
        ).copy()

    # ------------------------------------------------------------------
    def build_report(self) -> dict[str, Any]:
        parseable = self.total_rows - self.parse_failures - self.empty_smiles
        charged_molecules = (
            self.molecule_class["charged_net_neutral"]
            + self.molecule_class["net_cation"]
            + self.molecule_class["net_anion"]
        )
        rejected_rows = self.total_rows - self.accepted_rows
        non_neutral = self.rejection_counts.get("non_neutral", 0)

        def fraction(numerator: float, denominator: float) -> float | None:
            return numerator / denominator if denominator else None

        counts = {
            "total_molecules": self.total_rows,
            "parseable_molecules": parseable,
            "parse_failures": self.parse_failures,
            "empty_smiles": self.empty_smiles,
            "accepted_neutral_only": self.accepted_rows,
            "rejected_total": rejected_rows,
            "accepted_fraction": fraction(self.accepted_rows, self.total_rows),
            "rejected_fraction": fraction(rejected_rows, self.total_rows),
            "unique_canonical_smiles_among_accepted": len(
                self._accepted_canonical
            ),
            "duplicate_canonical_smiles_among_accepted": (
                self.accepted_rows - len(self._accepted_canonical)
            ),
            "charged_molecules": charged_molecules,
            "neutral_molecules": self.molecule_class["neutral"],
            "charged_fraction_of_parseable": fraction(
                charged_molecules, parseable
            ),
            "charged_fraction_of_all_rows": fraction(
                charged_molecules, self.total_rows
            ),
            "rejected_as_non_neutral": non_neutral,
            "non_neutral_fraction_of_all_rows": fraction(
                non_neutral, self.total_rows
            ),
            "multi_fragment_molecules": self.multi_fragment_molecules,
            "molecules_with_isotope": self.molecules_with_isotope,
            "molecules_with_radical": self.molecules_with_radical,
        }

        rejection = {
            reason: {
                "count": count,
                "fraction_of_all_rows": fraction(count, self.total_rows),
                "of_which_charged": self.rejected_charged_by_reason.get(
                    reason, 0
                ),
            }
            for reason, count in sorted(
                self.rejection_counts.items(), key=lambda kv: -kv[1]
            )
        }

        charge_census = {
            "molecule_classes": dict(self.molecule_class),
            "net_charge_distribution": {
                str(key): value
                for key, value in sorted(self.net_charge.items())
            },
            "charged_atoms_per_molecule": {
                str(key): value
                for key, value in sorted(self.charged_atoms_per_molecule.items())
            },
            "atom_formal_charge_distribution": {
                str(key): value
                for key, value in sorted(self.atom_charge.items())
            },
            "top_charge_signatures": [
                {"signature": signature, "molecules": count}
                for signature, count in self.charge_signature.most_common(30)
            ],
        }

        def atom_type_block(
            atoms: Counter, molecules: Counter, denominator: int,
        ) -> dict[str, Any]:
            return {
                "atoms": dict(atoms.most_common()),
                "atom_fractions": _fractions(atoms),
                "molecules_containing": dict(molecules.most_common()),
                "molecule_fractions": {
                    key: fraction(value, denominator)
                    for key, value in molecules.most_common()
                },
            }

        charged_types = sorted(
            (
                {
                    "atom_type": key,
                    "atoms": count,
                    "molecules": self.molecules_with_atom_type_all[key],
                    "fraction_of_parseable_molecules": fraction(
                        self.molecules_with_atom_type_all[key], parseable
                    ),
                }
                for key, count in self.atom_type_all.items()
                if "+" in key or "-" in key
            ),
            key=lambda item: -item["atoms"],
        )
        atom_types = {
            "all_parseable": atom_type_block(
                self.atom_type_all, self.molecules_with_atom_type_all, parseable
            ),
            "accepted_neutral_only": atom_type_block(
                self.atom_type_accepted,
                self.molecules_with_atom_type_accepted,
                self.accepted_rows,
            ),
            "charged_atom_types": charged_types,
            "out_of_vocabulary_elements": dict(
                self.out_of_vocab_elements.most_common()
            ),
        }

        qed = {
            "csv_vs_rdkit_original_molecule": {
                name: delta_statistics(*self.qed_pairs(name))
                for name in ("all_parseable", "charged", "uncharged")
            },
            "csv_vs_rdkit_reconstructed_accepted": delta_statistics(
                *self.qed_pairs("accepted_reconstructed")
            ),
            "csv_vs_rdkit_after_charge_strip_recovered": delta_statistics(
                *self.qed_pairs("charge_stripped_recovered")
            ),
            "note": (
                "delta = RDKit - CSV. 'original_molecule' parses the CSV "
                "SMILES as-is (charges kept). 'reconstructed_accepted' is the "
                "production label path. 'after_charge_strip_recovered' is the "
                "diagnostic counterfactual: how the label would shift if "
                "charges were dropped."
            ),
        }

        comparison_metrics: dict[str, Any] = {}
        for metric in METRICS:
            block: dict[str, Any] = {
                group: describe(self.values(group, metric)) for group in GROUPS
            }
            everyone = self.values("all", metric)
            kept = self.values("accepted", metric)
            if (
                block["all"].get("n", 0) > 1
                and block["accepted"].get("n", 0) > 1
            ):
                from scipy import stats as scipy_stats

                ks = scipy_stats.ks_2samp(everyone, kept)
                block["ks_all_vs_accepted"] = {
                    "statistic": float(ks.statistic),
                    "pvalue": float(ks.pvalue),
                }
                block["wasserstein_all_vs_accepted"] = float(
                    scipy_stats.wasserstein_distance(everyone, kept)
                )
                block["accepted_minus_all_mean"] = (
                    block["accepted"]["mean"] - block["all"]["mean"]
                )
                spread = block["all"].get("std", 0.0)
                block["accepted_minus_all_mean_in_all_std"] = (
                    (block["accepted_minus_all_mean"] / spread)
                    if spread else None
                )
            comparison_metrics[metric] = block

        element_all = _fractions(self.element_all)
        element_accepted = _fractions(self.element_accepted)
        histograms = {}
        for group in ("all", "accepted", "charged"):
            histograms[group] = {
                "qed_csv": histogram(
                    self.values(group, "qed_csv"), low=0.0, high=1.0, bins=40,
                ),
                "n_atoms": histogram(
                    self.values(group, "n_atoms"),
                    low=-0.5, high=self.max_nodes + 20.5,
                    bins=self.max_nodes + 21,
                ),
                "aromatic_fraction": histogram(
                    self.values(group, "aromatic_fraction"),
                    low=0.0, high=1.0, bins=20,
                ),
            }

        comparison = {
            "metrics": comparison_metrics,
            "element_composition": {
                "all_parseable": element_all,
                "accepted_neutral_only": element_accepted,
                "total_variation_distance": total_variation_distance(
                    element_all, element_accepted
                ),
            },
            "histograms": histograms,
            "note": (
                "Group 'all' uses every CSV row for CSV-derived metrics "
                "(qed_csv, logp_csv, sas_csv) and every parseable molecule "
                "for molecule-derived metrics. 'accepted' is the "
                "neutral-only set the production encoder keeps."
            ),
        }

        counterfactual = {
            "performed": self.counterfactual,
            "diagnostic_only": True,
            "description": (
                "Naive formal-charge stripping (explicit H on charged atoms "
                "reset, no decode-time +1 charge rule) followed by the "
                "unchanged encoder. Mirrors the effect of keeping atomic "
                "numbers only; it does NOT change the pipeline."
            ),
            "charged_molecules_tested": self.cf_charged,
            "recovered_by_existing_rules": self.cf_recovered,
            "recovered_fraction_of_charged": fraction(
                self.cf_recovered, self.cf_charged
            ),
            "still_rejected_reasons": dict(self.cf_failures.most_common()),
            "hypothetical_accepted_total": (
                self.accepted_rows + self.cf_recovered
            ),
            "hypothetical_accepted_fraction_of_all_rows": fraction(
                self.accepted_rows + self.cf_recovered, self.total_rows
            ),
            "signatures": [
                {
                    "signature": signature,
                    "molecules": total,
                    "recovered": self.cf_signature_recovered.get(signature, 0),
                }
                for signature, total in self.cf_signature_total.most_common(30)
            ],
            "recovered_n_atoms": describe(
                np.frombuffer(self.cf_recovered_nodes, dtype=np.float64)
            ),
        }

        return {
            "counts": counts,
            "rejection_reasons": rejection,
            "charge_census": charge_census,
            "atom_types": atom_types,
            "qed_discrepancy": qed,
            "distribution_comparison": comparison,
            "charge_strip_counterfactual": counterfactual,
            "timing_seconds": {
                "encoder": self.seconds_encode,
                "census_parse_and_qed": self.seconds_census,
                "counterfactual": self.seconds_counterfactual,
            },
        }


# ---------------------------------------------------------------------------
# Orchestration.
# ---------------------------------------------------------------------------
def verify_dataset_identity(
    csv_path: Path,
    *,
    expected_sha256: str | None,
    logger: Any,
) -> dict[str, Any]:
    """Hash the CSV and compare with the pinned digest (hard failure)."""
    started = time.perf_counter()
    digest = hash_file(csv_path)
    elapsed = time.perf_counter() - started
    info: dict[str, Any] = {
        "csv_path": str(csv_path),
        "csv_bytes": csv_path.stat().st_size,
        "sha256": digest,
        "expected_sha256": expected_sha256,
        "hash_seconds": elapsed,
    }
    if expected_sha256 is None:
        info["sha256_matches"] = None
        logger.warn("SHA-256 check disabled; dataset identity NOT verified")
    elif digest != expected_sha256:
        raise DatasetIdentityError(
            f"CSV SHA-256 mismatch for {csv_path}:\n"
            f"  expected: {expected_sha256}\n"
            f"  actual:   {digest}\n"
            "This is not the dataset the diagnostics were designed for."
        )
    else:
        info["sha256_matches"] = True
        logger.info(f"SHA-256 verified ({format_duration(elapsed)}): {digest}")
    return info


def summary_lines(report: dict[str, Any]) -> list[str]:
    """Compact human-readable summary for the log and a .txt file."""
    counts = report["counts"]
    lines = [
        "ZINC250K chemistry diagnostics summary",
        f"  complete run:            {report['complete']}",
        f"  total molecules:         {counts['total_molecules']:,}",
        f"  parse failures:          {counts['parse_failures']:,}",
        f"  accepted (neutral-only): {counts['accepted_neutral_only']:,} "
        f"({_pct(counts['accepted_fraction'])})",
        f"  rejected:                {counts['rejected_total']:,} "
        f"({_pct(counts['rejected_fraction'])})",
        f"  charged molecules:       {counts['charged_molecules']:,} "
        f"({_pct(counts['charged_fraction_of_parseable'])} of parseable)",
        f"  neutral molecules:       {counts['neutral_molecules']:,}",
        f"  rejected as non_neutral: {counts['rejected_as_non_neutral']:,} "
        f"({_pct(counts['non_neutral_fraction_of_all_rows'])} of all rows)",
        "  rejection reasons:",
    ]
    for reason, block in report["rejection_reasons"].items():
        lines.append(
            f"    {reason:<28} {block['count']:>9,} "
            f"({_pct(block['fraction_of_all_rows'])}, "
            f"charged: {block['of_which_charged']:,})"
        )
    top_types = report["atom_types"]["charged_atom_types"][:8]
    if top_types:
        lines.append("  most frequent charged atom types:")
        for item in top_types:
            lines.append(
                f"    {item['atom_type']:<8} atoms={item['atoms']:>9,} "
                f"molecules={item['molecules']:>8,}"
            )
    cf = report["charge_strip_counterfactual"]
    if cf.get("performed") and cf["charged_molecules_tested"]:
        lines.append(
            "  diagnostic charge-strip counterfactual (changes nothing):"
        )
        lines.append(
            f"    recovered by existing rules: {cf['recovered_by_existing_rules']:,}"
            f" / {cf['charged_molecules_tested']:,} charged "
            f"({_pct(cf['recovered_fraction_of_charged'])})"
        )
        lines.append(
            "    hypothetical accepted fraction of all rows: "
            f"{_pct(cf['hypothetical_accepted_fraction_of_all_rows'])}"
        )
        for reason, count in list(cf["still_rejected_reasons"].items())[:6]:
            lines.append(f"    still rejected: {reason:<28} {count:,}")
    qed_all = report["distribution_comparison"]["metrics"]["qed_csv"]
    if "accepted_minus_all_mean" in qed_all:
        lines.append(
            "  CSV QED mean: all="
            f"{qed_all['all']['mean']:.4f} accepted="
            f"{qed_all['accepted']['mean']:.4f} "
            f"KS={qed_all['ks_all_vs_accepted']['statistic']:.4f}"
        )
    return lines


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100.0 * value:.2f}%"


def run_chemistry_diagnostics(
    *,
    csv_path: str | Path,
    output_dir: str | Path,
    processed_dir: str | Path,
    logger: Any,
    figures_dir: str | Path | None = None,
    min_nodes: int = MIN_NODES,
    max_nodes: int = MAX_NODES,
    expected_sha256: str | None = EXPECTED_ZINC250K_SHA256,
    expected_rows: int | None = EXPECTED_ZINC250K_ROWS,
    max_molecules: int | None = None,
    sample_seed: int = 42,
    progress_every: int = 5000,
    heartbeat_seconds: float = 30.0,
    counterfactual: bool = True,
    make_figures: bool = True,
    repository_dir: str | Path | None = None,
    stop_flag: StopFlag | None = None,
) -> dict[str, Any]:
    """Run the full chemistry diagnostic and write the JSON report.

    Never writes inside `processed_dir`. Returns the report dict (also
    written to `<output_dir>/chemistry_report.json`, or
    `chemistry_report.partial.json` if the run was interrupted).
    """
    csv_path = Path(csv_path)
    output_dir = Path(output_dir)
    assert_output_outside(output_dir, processed_dir)
    if figures_dir is not None:
        assert_output_outside(figures_dir, processed_dir)

    run_started = time.perf_counter()
    logger.header("ZINC250K chemistry diagnostics (Phase 0, CPU only)")
    environment = collect_environment(repository_dir)
    logger.info(
        f"host={environment['hostname']} usable_cpus="
        f"{environment['usable_cpu_count']} "
        f"slurm_job={environment['slurm'].get('SLURM_JOB_ID', 'none')} "
        f"partition={environment['slurm'].get('SLURM_JOB_PARTITION', 'none')}"
    )
    logger.info(f"cpu_model={environment['cpu_model']}")
    logger.info(f"python={environment['python_executable']}")
    logger.info(f"commit={environment['repository_commit']}")

    identity = verify_dataset_identity(
        csv_path, expected_sha256=expected_sha256, logger=logger,
    )

    started = time.perf_counter()
    table = read_zinc_table(csv_path)
    logger.info(
        f"read {len(table):,} CSV records in "
        f"{format_duration(time.perf_counter() - started)} "
        f"(empty_smiles={table.empty_smiles_rows}, "
        f"nonnumeric_qed={table.nonnumeric_qed_rows}); "
        f"rss_mb={current_rss_mb()}"
    )
    row_count_matches = (
        None if expected_rows is None else len(table) == expected_rows
    )
    if row_count_matches is False:
        logger.warn(
            f"record count {len(table):,} differs from expected "
            f"{expected_rows:,}"
        )

    all_indices = list(range(len(table)))
    subsampled = max_molecules is not None and max_molecules < len(table)
    if subsampled:
        indices = deterministic_sample_indices(
            len(table), int(max_molecules), sample_seed,
        )
        logger.warn(
            f"SUBSAMPLE: using {len(indices):,} of {len(table):,} records "
            f"(seed={sample_seed}); results are NOT the full-dataset census"
        )
    else:
        indices = all_indices
    fingerprint = indices_fingerprint(indices)

    census = ChemistryCensus(
        min_nodes=min_nodes, max_nodes=max_nodes,
        counterfactual=counterfactual,
    )
    stop_flag = stop_flag or StopFlag()

    total = len(indices)
    loop_started = time.perf_counter()
    last_heartbeat = loop_started
    processed = 0
    logger.info(f"processing {total:,} molecules (progress every "
                f"{progress_every:,} or {heartbeat_seconds:.0f}s)")
    for position, row_index in enumerate(indices, start=1):
        census.add(
            table.smiles[row_index],
            table.qed[row_index],
            table.logp[row_index],
            table.sas[row_index],
        )
        processed = position
        now = time.perf_counter()
        if (
            position % progress_every == 0
            or now - last_heartbeat >= heartbeat_seconds
            or position == total
        ):
            elapsed = now - loop_started
            rate = position / elapsed if elapsed > 0 else float("nan")
            remaining = (total - position) / rate if rate and rate > 0 else None
            logger.info(
                f"progress {position:,}/{total:,} "
                f"({100.0 * position / total:.1f}%) "
                f"rate={rate:,.0f} mol/s elapsed={format_duration(elapsed)} "
                f"eta={format_duration(remaining)} "
                f"accepted={census.accepted_rows:,} "
                f"charged_cf={census.cf_charged:,} "
                f"rss_mb={current_rss_mb()}"
            )
            last_heartbeat = now
        if stop_flag.requested:
            logger.warn(
                f"stopping early at {position:,}/{total:,} "
                f"({stop_flag.signal_name})"
            )
            break

    complete = processed == total and not stop_flag.requested

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": "chemistry_diagnostics",
        "complete": complete,
        "stopped_by": stop_flag.signal_name,
        "subsampled": subsampled,
        "molecules_processed": processed,
        "records_selected": total,
        "sample_seed": sample_seed if subsampled else None,
        "sample_index_fingerprint": fingerprint,
        "representation_version": REPRESENTATION_VERSION,
        "encoding_rules": {
            "min_nodes": min_nodes,
            "max_nodes": max_nodes,
            "atom_vocab": list(ATOM_VOCAB),
            "neutral_only": True,
            "note": (
                "Rules are those of zinc_gnf.chemistry.encode_zinc_smiles, "
                "unchanged by this diagnostic."
            ),
        },
        "dataset": {
            **identity,
            "records_in_file": len(table),
            "expected_records": expected_rows,
            "record_count_matches_expected": row_count_matches,
            "empty_smiles_rows": table.empty_smiles_rows,
            "nonnumeric_qed_rows": table.nonnumeric_qed_rows,
            "molecules_removed_from_source": 0,
        },
        **census.build_report(),
    }

    # Prove the source file was not touched.
    if csv_path.exists():
        report["dataset"]["sha256_after_run"] = hash_file(csv_path)
        report["dataset"]["source_unchanged"] = (
            report["dataset"]["sha256_after_run"] == identity["sha256"]
        )

    report["environment"] = environment
    report["wall_seconds"] = time.perf_counter() - run_started
    report["peak_rss_mb"] = peak_rss_mb()

    suffix = "chemistry_report.json" if complete else "chemistry_report.partial.json"
    report_path = atomic_write_json(report, output_dir / suffix)
    logger.info(f"wrote report: {report_path}")

    lines = summary_lines(report)
    for line in lines:
        logger.info(line)
    summary_name = (
        "chemistry_summary.txt" if complete else "chemistry_summary.partial.txt"
    )
    (output_dir / summary_name).write_text("\n".join(lines) + "\n", encoding="utf-8")

    if make_figures and figures_dir is not None and processed > 0:
        from zinc_gnf.diagnostic_plots import write_chemistry_figures

        started = time.perf_counter()
        written = write_chemistry_figures(census, report, figures_dir)
        logger.info(
            f"wrote {len(written)} figures to {figures_dir} in "
            f"{format_duration(time.perf_counter() - started)}"
        )
        report["figures"] = [str(path) for path in written]
        atomic_write_json(report, output_dir / suffix)

    logger.info(
        f"done in {format_duration(time.perf_counter() - run_started)}; "
        f"complete={complete}"
    )
    return report


__all__ = [
    "ChemistryCensus",
    "DatasetIdentityError",
    "EXPECTED_ZINC250K_ROWS",
    "EXPECTED_ZINC250K_SHA256",
    "SKIP_HASH_CHECK",
    "ZincTable",
    "atom_type_key",
    "deterministic_sample_indices",
    "delta_statistics",
    "describe",
    "indices_fingerprint",
    "read_zinc_table",
    "run_chemistry_diagnostics",
    "strip_formal_charges",
    "summary_lines",
    "verify_dataset_identity",
]
