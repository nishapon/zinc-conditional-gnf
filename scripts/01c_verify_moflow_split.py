#!/usr/bin/env python3
"""Verify MoFlow's ZINC250K row identity and held-out indices. CPU only, read-only.

Compares MoFlow's `data/zinc250k.csv` with our CSV row by row (SMILES and the
logP/qed/SAS columns) and checks that MoFlow's `valid_idx_zinc.json` is
applicable to our row order. Writes a JSON report; changes nothing else.

Both MoFlow files must already be on disk (copy them on the login node;
compute nodes have no internet):

  https://raw.githubusercontent.com/calvin-zcx/moflow/master/data/zinc250k.csv
  https://raw.githubusercontent.com/calvin-zcx/moflow/master/data/valid_idx_zinc.json

Exit codes: 0 verified; 5 NOT verified (do not use the fixed split);
2 a file is missing; 4 bad arguments.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from zinc_gnf.diagnostics import read_zinc_table  # noqa: E402
from zinc_gnf.split_audit import verify_moflow_row_identity  # noqa: E402
from zinc_gnf.utils.runtime import atomic_write_json  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", default="data/raw/zinc250k.csv")
    parser.add_argument("--moflow-csv", required=True)
    parser.add_argument("--moflow-valid-idx", required=True)
    parser.add_argument("--output",
                        default="outputs/metrics/preprocessing_charge_aware/moflow_split_verification.json")
    arguments = parser.parse_args(argv)

    for label, path in (("--csv", arguments.csv),
                        ("--moflow-csv", arguments.moflow_csv),
                        ("--moflow-valid-idx", arguments.moflow_valid_idx)):
        if not Path(path).exists():
            print(f"FATAL: {label} file not found: {path}", file=sys.stderr, flush=True)
            return 2

    table = read_zinc_table(arguments.csv)
    result = verify_moflow_row_identity(
        table.smiles,
        {"logP": table.logp, "qed": table.qed, "SAS": table.sas},
        arguments.moflow_csv, arguments.moflow_valid_idx,
    )
    path = atomic_write_json(result, arguments.output)

    print(f"rows ours={result['rows_ours']:,} moflow={result['rows_moflow']:,}")
    print(f"SMILES mismatches: {result['smiles_mismatches']}")
    print(f"numeric columns: {result['numeric_columns']}")
    print(f"held-out count: {result['heldout_count']:,} "
          f"(index range {result['index_min']}..{result['index_max']})")
    print(f"index base assumed: {result['index_base_assumed']} "
          f"(confirmed by data: {result['index_base_confirmed_by_data']})")
    print(f"NOTE: {result['index_base_note']}")
    print(f"VERIFIED: {result['verified']}   report: {path}")
    return 0 if result["verified"] else 5


if __name__ == "__main__":
    sys.exit(main())
