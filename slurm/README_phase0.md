# Phase 0 diagnostics: how to run on the ISML cluster

Phase 0 is **diagnostic only**. It does not change the molecular
representation, the neutral-only filter, the model, or the production
preprocessing, and it never writes `data/processed/`.

Cluster rules these scripts follow (from the ISML cluster tutorial):

- CPU-only jobs on `--partition=CPU`, no `--gres=gpu`.
- Nothing runs on the master node; jobs go through `sbatch` + `srun`.
  (Running the `.sh` files directly exits with code 64 before doing any work.)
- Compute nodes have no internet: the dataset is downloaded on the login node.
- Minimal requests (1 CPU, 2-4 GB), time limits far below the 24 h maximum.
- The `outputs/logs/` directory exists before submission (SLURM does not
  create the directory of `--output`/`--error`).

## One-time preparation (login node)

```bash
cd ~/zinc-conditional-gnf
git pull origin rewrite/zinc250k-pipeline
bash slurm/00_prepare_phase0.sh        # mkdir only; checks the dataset exists
```

If the dataset is missing, download it **on the login node**:

```bash
mkdir -p data/raw
wget -O data/raw/zinc250k.csv \
  https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/main/models/zinc_properties/250k_rndm_zinc_drugs_clean_3.csv
```

The jobs verify the pinned SHA-256
`35e3f1a52b1badc0697e373d73a18ad773f415936ff992f4c6baa2e067b3e6ae`
and stop with a clear message if it differs.

## Jobs

| Order | Command | Limits | Purpose |
|---|---|---|---|
| 1 | `sbatch slurm/00_profile_preprocessing.sh` | 1 CPU, 2G, 30 min | Where does time go on a compute node; full-run estimate |
| 2 | `sbatch slurm/00_chemistry_diagnostics.sh` | 1 CPU, 4G, 4 h | Charge census, rejection reasons, QED agreement |

Quick end-to-end check of job 2 on 2000 molecules (still through sbatch):

```bash
sbatch --export=ALL,MAX_MOLECULES=2000 --time=00:20:00 slurm/00_chemistry_diagnostics.sh
```

If the Conda env is not at `~/miniconda3/envs/zinc_gnf`, pass its python:

```bash
sbatch --export=ALL,ZINC_GNF_PYTHON=/path/to/envs/zinc_gnf/bin/python slurm/00_profile_preprocessing.sh
```

## Where things land

| Item | Path |
|---|---|
| Job log / error | `outputs/logs/zinc_p0_profile_<jobid>.log` / `.err`, `outputs/logs/zinc_p0_chem_<jobid>.log` / `.err` |
| Profiler report | `outputs/diagnostics/profiling/preprocessing_profile.json` and `.txt` |
| Profiler run log | `outputs/diagnostics/profiling/run.log` |
| Chemistry report | `outputs/diagnostics/chemistry/chemistry_report.json` and `chemistry_summary.txt` |
| Chemistry run log | `outputs/diagnostics/chemistry/run.log` |
| Figures | `outputs/figures/diagnostics/*.png` (7 files) |

If a job hits its time limit or is cancelled it writes
`*.partial.json` / `*.partial.txt` next to the normal names instead of
losing the run (exit code 130).

## Monitoring

```bash
squeue -u $USER
tail -f outputs/logs/zinc_p0_chem_<jobid>.log
sacct -j <jobid> --format=JobID,State,ExitCode,Elapsed,TotalCPU,MaxRSS,NodeList
seff <jobid>
```

Exit codes of the Python scripts: `0` ok, `2` CSV missing, `3` SHA-256
mismatch, `4` bad arguments/config, `130` stopped early (partial report),
`1` unexpected error (full traceback is in the `.log`).
