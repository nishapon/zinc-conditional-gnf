"""Slice 0 smoke tests: verify the foundation modules import and have
their expected shape, so we can confirm `pip install -e .` is healthy
before any real work starts.
"""

from __future__ import annotations

import pytest


def test_package_importable_and_versioned() -> None:
    import zinc_gnf

    assert hasattr(zinc_gnf, "__version__")
    assert isinstance(zinc_gnf.__version__, str)
    assert zinc_gnf.__version__.count(".") == 2  # X.Y.Z


def test_constants_module_shape() -> None:
    from zinc_gnf import constants

    # Vocab size must match ZINC250K / MoFlow Table 1.
    assert constants.NUM_ATOM_TYPES == 9
    assert constants.NODE_FEATURE_DIM == 9
    assert constants.NUM_BOND_TYPES == 4
    assert constants.MAX_NODES == 38
    assert constants.MIN_NODES == 2

    # Vocab and valence stay in lockstep.
    assert len(constants.ATOM_VOCAB) == constants.NUM_ATOM_TYPES
    assert len(constants.ATOM_VALENCE) == constants.NUM_ATOM_TYPES
    assert set(constants.VALENCE.keys()) == set(constants.ATOM_VOCAB)

    # Textbook valences (sanity).
    assert constants.VALENCE["C"] == 4
    assert constants.VALENCE["N"] == 3
    assert constants.VALENCE["O"] == 2
    assert constants.VALENCE["F"] == 1
    assert constants.VALENCE["S"] == 6

    # Representation version is a non-empty string so checkpoints can
    # record which encoding they were trained against.
    assert isinstance(constants.REPRESENTATION_VERSION, str)
    assert constants.REPRESENTATION_VERSION


def test_bond_class_mapping_is_bijective() -> None:
    from zinc_gnf import constants

    assert len(constants.BOND_TO_INDEX) == 3  # single, double, triple
    assert set(constants.BOND_TO_INDEX.values()) == {1, 2, 3}
    # INDEX_TO_BOND is the inverse of BOND_TO_INDEX.
    for bond_type, index in constants.BOND_TO_INDEX.items():
        assert constants.INDEX_TO_BOND[index] == bond_type


def test_utils_import_cleanly() -> None:
    # Each util module must be importable in isolation.
    from zinc_gnf.utils import config
    from zinc_gnf.utils import seed
    from zinc_gnf.utils import checkpointing
    from zinc_gnf.utils import logging as job_logging

    assert hasattr(config, "load_config")
    assert hasattr(seed, "set_seed")
    assert hasattr(checkpointing, "save_checkpoint")
    assert hasattr(checkpointing, "load_checkpoint")
    assert hasattr(job_logging, "JobLogger")


def test_config_loads(tmp_path) -> None:
    """The real zinc250k.yaml config must load, so a trainer that calls
    load_config on it can rely on its presence."""
    from pathlib import Path

    from zinc_gnf.utils import config

    # Project root is 2 levels up from this test file.
    repo_root = Path(__file__).resolve().parent.parent
    config_path = repo_root / "configs" / "zinc250k.yaml"

    assert config_path.exists(), f"Missing config: {config_path}"

    cfg = config.load_config(config_path)

    # The config is the single source of truth for a few key constants —
    # cross-check against zinc_gnf.constants so the two can't silently
    # drift.
    from zinc_gnf import constants

    assert cfg["data"]["atom_vocab"] == list(constants.ATOM_VOCAB)
    assert cfg["data"]["bond_classes"] == constants.NUM_BOND_TYPES
    assert cfg["data"]["max_nodes"] == constants.MAX_NODES
    assert cfg["autoencoder"]["node_feature_dim"] == constants.NODE_FEATURE_DIM

    # Flow dimensionality has to be even (coupling layer splits in half).
    assert cfg["flow"]["embedding_dim"] % 2 == 0

    # Sigma clamp value — professor's whiteboard check. The config should
    # NOT silently default to a looser value.
    assert cfg["flow"]["max_log_scale"] <= 0.5


def test_job_logger_writes_to_file(tmp_path) -> None:
    from zinc_gnf.utils.logging import JobLogger

    logger = JobLogger(name="test", output_dir=tmp_path)
    logger.info("hello")
    logger.metrics(step=0, metrics={"loss": 1.234, "accuracy": 0.567})
    logger.shapes({"x": _FakeTensor(shape=(2, 3), dtype="float32")})

    log_path = tmp_path / "run.log"
    assert log_path.exists()
    content = log_path.read_text(encoding="utf-8")
    assert "hello" in content
    assert "loss=1.234" in content
    assert "x=(2, 3):float32" in content


class _FakeTensor:
    def __init__(self, shape, dtype) -> None:
        self.shape = shape
        self.dtype = dtype
