"""Tests for the ZINC preprocessing protocol."""

from __future__ import annotations

import json
import pickle

import numpy as np
import pandas as pd

from zinc_gnf.data.preprocessing import (
    ConditionScaler,
    murcko_scaffold,
    prepare_zinc_dataset,
    scaffold_split,
)


# Six distinct scaffolds, with two molecules per scaffold.
MOLECULES = [
    "c1ccccc1",
    "Cc1ccccc1",
    "C1CCCCC1",
    "CC1CCCCC1",
    "n1ccccc1",
    "Cc1ccccn1",
    "c1ccoc1",
    "Cc1ccoc1",
    "c1ccsc1",
    "Cc1ccsc1",
    "c1ncc[nH]1",
    "Cc1ncc[nH]1",
]


def test_condition_scaler_training_statistics() -> None:
    records = [
        {
            "qed": 0.20 + 0.05 * index,
            "n_node": 5 + index,
        }
        for index in range(8)
    ]

    scaler = ConditionScaler.fit(records)

    transformed = np.stack(
        [
            scaler.transform(
                record["qed"],
                record["n_node"],
            )
            for record in records
        ]
    )

    np.testing.assert_allclose(
        transformed.mean(axis=0),
        0.0,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        transformed.std(axis=0),
        1.0,
        atol=1e-6,
    )


def test_condition_scaler_round_trip() -> None:
    records = [
        {"qed": 0.30, "n_node": 8},
        {"qed": 0.50, "n_node": 12},
        {"qed": 0.70, "n_node": 18},
        {"qed": 0.90, "n_node": 25},
    ]

    scaler = ConditionScaler.fit(records)

    for record in records:
        condition = scaler.transform(
            record["qed"],
            record["n_node"],
        )
        restored_qed, restored_n = (
            scaler.inverse_transform(condition)
        )

        np.testing.assert_allclose(
            restored_qed,
            record["qed"],
            atol=1e-5,
        )
        np.testing.assert_allclose(
            restored_n,
            record["n_node"],
            atol=1e-5,
        )


def test_scaffold_split_has_no_leakage() -> None:
    records = [
        {"smiles": smiles}
        for smiles in MOLECULES
    ]

    splits, metadata = scaffold_split(
        records,
        seed=7,
        train_fraction=0.5,
        validation_fraction=0.25,
    )

    scaffold_sets = {
        split_name: {
            murcko_scaffold(record["smiles"])
            for record in subset
        }
        for split_name, subset in splits.items()
    }

    assert scaffold_sets["train"].isdisjoint(
        scaffold_sets["val"]
    )
    assert scaffold_sets["train"].isdisjoint(
        scaffold_sets["test"]
    )
    assert scaffold_sets["val"].isdisjoint(
        scaffold_sets["test"]
    )

    assert sum(
        len(subset)
        for subset in splits.values()
    ) == len(records)

    assert metadata["largest_scaffold_group"] == 2


def test_scaffold_split_is_reproducible() -> None:
    records = [
        {"smiles": smiles}
        for smiles in MOLECULES
    ]

    first, _ = scaffold_split(
        records,
        seed=11,
        train_fraction=0.5,
        validation_fraction=0.25,
    )
    second, _ = scaffold_split(
        records,
        seed=11,
        train_fraction=0.5,
        validation_fraction=0.25,
    )

    for split_name in ("train", "val", "test"):
        assert [
            record["smiles"]
            for record in first[split_name]
        ] == [
            record["smiles"]
            for record in second[split_name]
        ]


def test_prepare_small_dataset(tmp_path) -> None:
    csv_path = tmp_path / "zinc.csv"

    pd.DataFrame(
        {"smiles": MOLECULES}
    ).to_csv(
        csv_path,
        index=False,
    )

    output_dir = tmp_path / "processed"

    split_path, scaler_path, report = (
        prepare_zinc_dataset(
            csv_path,
            output_dir,
            target_molecules=12,
            candidate_count=12,
            seed=7,
            train_fraction=0.5,
            validation_fraction=0.25,
        )
    )

    assert split_path.exists()
    assert scaler_path.exists()
    assert (
        output_dir / "preprocessing_report.json"
    ).exists()

    assert report.selected_molecules == 12

    with split_path.open("rb") as handle:
        payload = pickle.load(handle)

    assert sum(
        len(payload[split_name])
        for split_name in ("train", "val", "test")
    ) == 12

    training_conditions = np.stack(
        [
            record["condition"]
            for record in payload["train"]
        ]
    )

    np.testing.assert_allclose(
        training_conditions.mean(axis=0),
        0.0,
        atol=1e-5,
    )

    scaler_payload = json.loads(
        scaler_path.read_text(encoding="utf-8")
    )

    assert scaler_payload["fit_on"] == "train_only"
