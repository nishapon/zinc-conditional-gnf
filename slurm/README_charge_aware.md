# Charge-aware ZINC250K preprocessing (representation v2)

The neutral-only pipeline (`zinc250k_neutral_kekule_v1`, `scripts/01_preprocess.py`,
`data/processed/zinc250k/`) is the **baseline and is unchanged**. This path is
separate: different modules, config, and output directories.

## What it does

Every atom is an `(element, formal charge)` type. 17 types: the nine neutral
elements (indices 0-8, identical to v1) plus the eight charge types measured
on all 249,455 molecules in Phase 0: `N+1 O-1 N-1 S-1 O+1 S+1 C-1 P+1`.
Hydrogens stay implicit and RDKit infers them from the typed valence.

Validated on the real dataset in a sandbox (not on the cluster):

- 249,454 / 249,455 molecules accepted; 81,325 charged molecules recovered
  (exactly Phase 0's `non_neutral` count); 168,129 neutral molecules encode
  **identically** to v1.
- The single rejected row is a genuine chemistry limitation:
  `O=[PH2][C@H](Cc1ccccc1)c1ccccc1` (row 16175), a phosphine oxide with two
  explicit hydrogens on pentavalent P. Valence-based hydrogen inference
  cannot reproduce it, so the round trip is not exact and it is rejected
  with the reason `roundtrip_mismatch` (it is also rejected by v1).
- 8m45s serial (~495 mol/s), peak RSS ~1.5 GB, `splits.pkl` ~1.2 GB.

## Run

```bash
cd ~/zinc-conditional-gnf
git pull origin rewrite/zinc250k-pipeline
bash slurm/01b_prepare_charge_aware.sh      # login node: mkdir + file checks only
sbatch slurm/01b_preprocess_charge_aware.sh
```

1 CPU, 6 GB, 90 min, `CPU` partition, no GPU, no network.

## Outputs

| Item | Path |
|---|---|
| Job log / error | `outputs/logs/zinc_charge_prep_<jobid>.log` / `.err` |
| Timestamped run log | `outputs/metrics/preprocessing_charge_aware/run.log` |
| Train/val/test records | `data/processed/zinc250k_charge_aware/splits.pkl` |
| Train-only scaler | `data/processed/zinc250k_charge_aware/scaler.json` |
| Full report | `data/processed/zinc250k_charge_aware/report.json` |
| Every CSV row: status, reason, split | `data/processed/zinc250k_charge_aware/row_index.csv` |
| Rejections | `outputs/metrics/preprocessing_charge_aware/rejected_rows.csv`, `rejection_reasons.json` |
| Atom / bond / charge composition | `outputs/metrics/preprocessing_charge_aware/{atom,bond,charge}_composition.json` |
| QED CSV vs RDKit | `outputs/metrics/preprocessing_charge_aware/qed_discrepancy.json` |
| Duplicates | `outputs/metrics/preprocessing_charge_aware/duplicates.json` |
| Cross-split overlap | `outputs/metrics/preprocessing_charge_aware/split_overlap.json` |
| MoFlow verification (if files present) | `outputs/metrics/preprocessing_charge_aware/moflow_split_verification.json` |

Nothing is written to `data/processed/zinc250k_charge_aware/` unless the whole
run succeeds. Row ids are 0-based CSV positions (header excluded) and appear
on every record as `row_id`.

## Split protocols

- `random_80_10_10` (default): the baseline's seeded shuffle
  (`zinc_gnf.preprocessing.random_split`), seed 42. The scaler is fit on the
  training split only.
- `moflow_fixed` (opt-in): MoFlow's 24,887 held-out molecules are the test
  set; validation is a seeded 10% of the rest. It is refused unless MoFlow's
  `zinc250k.csv` matches ours row for row. Sandbox verification on the real
  files: SMILES identical at all 249,455 positions; logP/qed/SAS agree to
  1.8e-15 (float serialization only). **Assumption:** MoFlow's
  `valid_idx_zinc.json` is 1-based (MoFlow's own code subtracts 1); both bases
  are in range, so the data cannot confirm this. It is recorded in every report.

To enable the verification report (and optionally the fixed split), copy
MoFlow's files **on the login node**:

```bash
wget -O data/external/moflow/zinc250k.csv https://raw.githubusercontent.com/calvin-zcx/moflow/master/data/zinc250k.csv
wget -O data/external/moflow/valid_idx_zinc.json https://raw.githubusercontent.com/calvin-zcx/moflow/master/data/valid_idx_zinc.json
```

Then, only if you decide to use the fixed split:

```bash
sbatch --export=ALL,SPLIT_PROTOCOL=moflow_fixed slurm/01b_preprocess_charge_aware.sh
```

## No deduplication

Duplicates are reported, never removed. Identity is the non-stereochemical
canonical SMILES, so stereoisomers share an identity. On the real data all
1,644 duplicate groups are stereoisomer pairs/triples (1,651 excess records,
0.66%), and about 1% of validation/test records have a training twin.
Interpret novelty and reconstruction claims with that in mind.

## QED labels

The label is RDKit's QED of the **reconstructed** molecule, the same path
generated molecules take, so it is a property of the representation. It
differs systematically from the CSV's QED (mean absolute difference 0.0126 on
the real data; 4 rows above 0.1), and the RDKit version is stored in the
outputs. Regenerate labels if the RDKit version changes.
