"""Registry of molecular representations, for model-code compatibility.

The graph autoencoder and conditional flow (future stages) should read their
atom-vocabulary size, valence table and version from a `RepresentationSpec`
instead of importing `zinc_gnf.constants` directly. That keeps them agnostic
to whether the data is the neutral-only baseline (v1) or the charge-aware
representation (v2):

    spec = get_representation("zinc250k_charge_aware_kekule_v2")
    node_feature_dim = spec.num_atom_types
    caps = spec.valence

Both specs describe the same graph interface: `atom_types` (int64 [N]) indexing
`spec.atom_vocab`, and `bond_matrix` (int64 [N, N]) with `spec.num_bond_types`
classes. Only the atom vocabulary differs.
"""

from __future__ import annotations

from dataclasses import dataclass

from zinc_gnf import constants as v1
from zinc_gnf import constants_charge_aware as v2


@dataclass(frozen=True)
class RepresentationSpec:
    version: str
    atom_vocab: tuple[str, ...]
    elements: tuple[str, ...]
    charges: tuple[int, ...]
    valence: tuple[int, ...]
    num_bond_types: int
    min_nodes: int
    max_nodes: int
    charge_aware: bool

    @property
    def num_atom_types(self) -> int:
        return len(self.atom_vocab)

    @property
    def node_feature_dim(self) -> int:
        return len(self.atom_vocab)


NEUTRAL_V1 = RepresentationSpec(
    version=v1.REPRESENTATION_VERSION,
    atom_vocab=tuple(v1.ATOM_VOCAB),
    elements=tuple(v1.ATOM_VOCAB),
    charges=tuple(0 for _ in v1.ATOM_VOCAB),
    valence=tuple(v1.ATOM_VALENCE),
    num_bond_types=v1.NUM_BOND_TYPES,
    min_nodes=v1.MIN_NODES,
    max_nodes=v1.MAX_NODES,
    charge_aware=False,
)

CHARGE_AWARE_V2 = RepresentationSpec(
    version=v2.REPRESENTATION_VERSION_CHARGE_AWARE,
    atom_vocab=tuple(v2.ATOM_VOCAB),
    elements=tuple(v2.ELEMENT_OF_TYPE),
    charges=tuple(v2.CHARGE_OF_TYPE),
    valence=tuple(v2.ATOM_VALENCE),
    num_bond_types=v2.NUM_BOND_TYPES,
    min_nodes=v2.MIN_NODES,
    max_nodes=v2.MAX_NODES,
    charge_aware=True,
)

REPRESENTATIONS: dict[str, RepresentationSpec] = {
    NEUTRAL_V1.version: NEUTRAL_V1,
    CHARGE_AWARE_V2.version: CHARGE_AWARE_V2,
}


def get_representation(version: str) -> RepresentationSpec:
    try:
        return REPRESENTATIONS[version]
    except KeyError as error:
        raise KeyError(
            f"unknown representation {version!r}; known: {sorted(REPRESENTATIONS)}"
        ) from error


__all__ = [
    "CHARGE_AWARE_V2",
    "NEUTRAL_V1",
    "REPRESENTATIONS",
    "RepresentationSpec",
    "get_representation",
]
