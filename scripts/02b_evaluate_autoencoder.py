#!/usr/bin/env python3
"""Stage 2b — reconstruction evaluation of a trained autoencoder checkpoint.

Reports, SEPARATELY, raw (argmax) decoding and constrained chemical decoding:
validity, exact graph match, exact molecule (canonical SMILES) match, plus
atom / bond accuracy, the teacher-forced loss and the exact-reconstruction
ceiling of a deterministic one-hot equivariant encoder.

Defaults to the VALIDATION split. The held-out TEST split is for final
evaluation only and requires the explicit flag `--allow-test`; it must not be
used for model selection or tuning.

Exit codes: 0 ok; 2 file missing; 3 checkpoint/vocabulary mismatch;
4 bad arguments; 1 unexpected error.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import traceback

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from zinc_gnf.autoencoder import evaluate_reconstruction  # noqa: E402
from zinc_gnf.dataset import (  # noqa: E402
    DatasetFormatError,
    MolecularGraphDataset,
    load_test_split,
    load_training_splits,
    seeded_subset_indices,
)
from zinc_gnf.trainer_autoencoder import (  # noqa: E402
    ResumeMismatchError,
    load_autoencoder,
    resolve_device,
)
from zinc_gnf.utils.runtime import atomic_write_json  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", default="data/processed/zinc250k_charge_aware_moflow_fixed")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--allow-test", action="store_true",
                        help="required to evaluate the held-out test split (final evaluation only)")
    parser.add_argument("--max-molecules", type=int, default=None)
    parser.add_argument("--subset-seed", type=int, default=12345)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", default=None, help="JSON report path")
    arguments = parser.parse_args(argv)

    if arguments.split == "test" and not arguments.allow_test:
        print("FATAL: the test split is held out for final evaluation; "
              "pass --allow-test to use it deliberately", file=sys.stderr, flush=True)
        return 4
    try:
        device = resolve_device(arguments.device)
        model, checkpoint = load_autoencoder(arguments.checkpoint, device=device)
        if arguments.split == "val":
            _, records, fingerprint = load_training_splits(arguments.data_dir)
        else:
            records, fingerprint = load_test_split(arguments.data_dir)
        dataset = MolecularGraphDataset(records)
        indices = seeded_subset_indices(len(dataset), arguments.max_molecules, arguments.subset_seed)
        print(f"evaluating {len(indices):,} of {len(dataset):,} {arguments.split} molecules "
              f"from step {checkpoint.get('step')} on {device}", flush=True)
        result = evaluate_reconstruction(model, dataset, indices=indices, device=device,
                                         batch_size=arguments.batch_size)
    except FileNotFoundError as error:
        print(f"FATAL: {error}", file=sys.stderr, flush=True)
        return 2
    except (ResumeMismatchError, DatasetFormatError) as error:
        print(f"FATAL: {error}", file=sys.stderr, flush=True)
        return 3
    except (ValueError, RuntimeError) as error:
        print(f"FATAL: {error}", file=sys.stderr, flush=True)
        return 4
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        return 1

    report = {
        "split": arguments.split, "checkpoint": str(arguments.checkpoint),
        "checkpoint_step": checkpoint.get("step"), "device": str(device),
        "molecules_evaluated": len(indices), "molecules_in_split": len(dataset),
        "subset_seed": arguments.subset_seed, "data": fingerprint,
        "model_config": checkpoint.get("model_config"), "metrics": result,
    }
    for name in ("raw", "constrained"):
        g = result[name]
        print(f"{name:>11}: validity={g['validity']:.4f} exact_graph={g['exact_graph_match']:.4f} "
              f"exact_molecule={g['exact_molecule_match']:.4f}", flush=True)
    print(f"atom_acc={result['atom_accuracy']:.4f} bond_acc={result['bond_accuracy']:.4f} "
          f"real_bond_acc={result['real_bond_accuracy']:.4f} mean_loss={result['mean_loss']:.4f} "
          f"equivariant_ceiling={result.get('equivariant_exact_ceiling', float('nan')):.4f}", flush=True)
    if arguments.output:
        print(f"report: {atomic_write_json(report, arguments.output)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
