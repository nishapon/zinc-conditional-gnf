"""Shared fixtures for the Phase 0 diagnostic tests.

Everything is synthetic and tiny: no dataset, no network, no cluster.
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import pytest
from rdkit import Chem, rdBase
from rdkit.Chem import QED

from zinc_gnf.utils.logging import JobLogger

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

# (smiles, expected encoder verdict under the EXISTING neutral-only rules)
SYNTHETIC_MOLECULES = [
    ("CCO", "accepted"),
    ("c1ccccc1", "accepted"),
    ("CC(=O)Nc1ccc(O)cc1", "accepted"),
    ("C[NH3+]", "non_neutral"),             # strip -> CN        (recovered)
    ("CC(=O)[O-]", "non_neutral"),          # strip -> CC(=O)O   (recovered)
    ("C[N+](=O)[O-]", "non_neutral"),       # nitro: strip fails (valence)
    ("C[N+](C)(C)C", "non_neutral"),        # quaternary: strip fails (valence)
    ("C[SiH3]", "unsupported_element"),
    ("[2H]CCO", "isotope"),
    ("CC.O", "multiple_fragments"),
    ("not a smiles", "parse_failure"),
    ("C", "node_count_below_min"),
    ("", "empty_smiles"),
]

# Extra neutral molecules so the profiler has >= 10 accepted records.
EXTRA_ACCEPTED = [
    "CCN", "CCCO", "c1ccncc1", "Fc1ccccc1", "Clc1ccccc1", "CC(=O)O",
    "c1ccsc1", "c1ccoc1", "C=CC#N", "O=C(Nc1ccc(F)cc1)c1cc(Cl)ccc1Br",
    "CC(C)Cc1ccc(C(C)C(=O)O)cc1", "c1cnc[nH]1",
]

# CCO gets a deliberately wrong CSV QED so discrepancy statistics are testable.
QED_OFFSET = {"CCO": 0.2}


def write_zinc_csv(path: Path, smiles_list: list[str]) -> Path:
    """Write a ZINC250K-style CSV (each SMILES has an embedded newline)."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["smiles", "logP", "qed", "SAS"])
        for index, smiles in enumerate(smiles_list):
            with rdBase.BlockLogs():
                molecule = Chem.MolFromSmiles(smiles) if smiles else None
            if molecule is not None:
                qed_value = QED.qed(molecule) + QED_OFFSET.get(smiles, 0.0)
            else:
                qed_value = 0.5
            writer.writerow(
                [smiles + "\n", round(1.0 + 0.01 * index, 4),
                 round(qed_value, 6), 2.0]
            )
    return path


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def synthetic_csv(tmp_path: Path) -> Path:
    return write_zinc_csv(
        tmp_path / "zinc_synthetic.csv",
        [smiles for smiles, _ in SYNTHETIC_MOLECULES],
    )


@pytest.fixture
def profiling_csv(tmp_path: Path) -> Path:
    smiles = [s for s, _ in SYNTHETIC_MOLECULES] + EXTRA_ACCEPTED
    return write_zinc_csv(tmp_path / "zinc_profiling.csv", smiles)


@pytest.fixture
def logger(tmp_path: Path) -> JobLogger:
    return JobLogger("test_phase0", tmp_path / "logs")


@pytest.fixture
def repository_root() -> Path:
    return REPOSITORY_ROOT
