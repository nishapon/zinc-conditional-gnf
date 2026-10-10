# Primary experiment: MoFlow-fixed dataset and graph autoencoder

## Protocol (locked)

| Decision | Choice |
|---|---|
| Split | MoFlow's official ZINC250K held-out molecules = **test** (24,887). Validation = a seeded 10% of the remaining pool (22,456). Train = the rest (202,111). The one rejected molecule (row 16175) is in the pool and excluded. |
| Data source | Derived from the already-encoded charge-aware records by `row_id`. Nothing is re-encoded; the random-split dataset is untouched. |
| Held-out index base | 1-based, as in MoFlow's own code (`idx - 1`). The data cannot confirm this; it is recorded in every report. |
| Scaler | Refit on the new training subset only. |
| Overlaps | Reported, not removed (`split_overlap.json`). 261 test records have a canonical twin in train (1.05%). |
| Novelty reference | Must be named explicitly. Main reference = the training set (202,111 records, 201,019 unique). The full dataset including test is never the main reference. |
| Autoencoder input | One-hot identity over the 17 (element, charge) types + bond graph + padding masks. No noise stand-in. |
| Seeds | One autoencoder, frozen; five independently seeded flows. The 5-seed uncertainty covers flow training and generation, **not** autoencoder variability. |
| Validation | Used for checkpoint selection and tuning. The test split is never loaded by training. No train+val training. |

## Run order (all on the cluster, through sbatch)

```bash
cd ~/zinc-conditional-gnf
git pull origin rewrite/zinc250k-pipeline
bash slurm/01b_prepare_charge_aware.sh         # mkdir + checks (login node)

# One-time, on the LOGIN node: MoFlow's two files (compute nodes have no internet)
wget -O data/external/moflow/zinc250k.csv        https://raw.githubusercontent.com/calvin-zcx/moflow/master/data/zinc250k.csv
wget -O data/external/moflow/valid_idx_zinc.json https://raw.githubusercontent.com/calvin-zcx/moflow/master/data/valid_idx_zinc.json

sbatch slurm/01d_derive_moflow_split.sh        # CPU, 1 CPU, 8 GB, <= 30 min (about 20 s measured)
sbatch slurm/02_ae_smoke_gpu.sh                # GPU smoke test, after the derivation finishes
```

To chain them: `sbatch --dependency=afterok:<derive_jobid> slurm/02_ae_smoke_gpu.sh`.

The GPU job needs the **GPU build** of PyTorch in the `zinc_gnf` Conda env
(install on the login node: `pip3 install torch --index-url https://download.pytorch.org/whl/cu121`
or the build matching the cluster's driver). It fails fast, with that hint, if
`torch.cuda.is_available()` is False.

## Outputs

| Item | Path |
|---|---|
| Primary dataset | `data/processed/zinc250k_charge_aware_moflow_fixed/{splits.pkl, scaler.json, report.json, row_index.csv}` |
| Split reports | `outputs/metrics/split_moflow_fixed/{derive.log, split_derivation_checks.json, split_overlap.json, moflow_split_verification.json}` |
| Smoke training log | `outputs/checkpoints/ae_smoke_gpu/train.log` |
| Per-eval metrics | `outputs/checkpoints/ae_smoke_gpu/metrics.jsonl` |
| Checkpoints | `outputs/checkpoints/ae_smoke_gpu/best.pt` (validation-selected), `last.pt` (resume) |
| Summary / reconstruction | `outputs/checkpoints/ae_smoke_gpu/summary.json`, `val_reconstruction.json` |
| SLURM logs | `outputs/logs/zinc_moflow_split_<jobid>.log` / `.err`, `outputs/logs/zinc_ae_smoke_<jobid>.log` / `.err` |

## Reading the reconstruction numbers

- **Raw** = argmax atoms and bonds with no chemistry. **Constrained** =
  valence-aware assembly with a connectivity pass (the teammates' decoder,
  ported to typed valence). They are reported separately and must not be
  mixed up with each other or with MoFlow's published validity.
- **`equivariant_exact_ceiling`**: the fraction of molecules a deterministic
  one-hot, permutation-equivariant encoder with a pairwise bond decoder could
  *ever* reconstruct exactly (colour-refinement bound). Measured on the real
  data it is about 74-77% for ZINC250K against about 93% for QM9 (which matches
  the teammates' ~90% QM9 result). Atoms in equivalent positions, such as the
  ortho/meta atoms of a phenyl ring, get identical embeddings and cannot be
  told apart by the decoder. Exact-match rates should be read against this
  ceiling, and `--symmetry-noise-dim` (default 0) adds noise channels *next to*
  the identity features as an explicit ablation.
