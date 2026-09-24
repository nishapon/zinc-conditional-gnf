# Conditional GNF for ZINC250K

This repository trains a molecular Graph Normalizing Flow on ZINC250K.

The model learns:

```text
p(node embeddings | QED, heavy-atom count)
```

It uses a bond-aware molecular autoencoder, QED-supervised embeddings, a conditional node-level GNF, and chemistry-aware molecular decoding.

## 1. Clone and install

```bash
git clone https://github.com/nishapon/zinc-conditional-gnf.git
cd zinc-conditional-gnf
git switch refactor/modular-zinc-pipeline

conda create -n zinc_gnf python=3.11 -y
conda activate zinc_gnf
pip install -e ".[dev]"
pytest -q
```

## 2. Add the dataset

Place the ZINC250K CSV here:

```text
data/raw/zinc250k.csv
```

Datasets, checkpoints, logs, embeddings, and generated molecules are not tracked by Git.

## 3. Check the SLURM configuration

Before submitting jobs, check the following fields in `slurm/*.sh`:

```bash
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1
```

Also verify the Python environment and repository path used by the scripts.

Create the log directory:

```bash
mkdir -p logs
```

## 4. Run the pipeline

Submit these jobs in order. Check that each stage completed successfully before starting the next one.

### Step 1: Preprocess ZINC250K

```bash
sbatch slurm/run_zinc_preprocessing.sh
```

Creates scaffold-disjoint 80/10/10 splits and training-only condition statistics.

Expected output:

```text
data/processed/zinc250k/zinc_splits.pkl
data/processed/zinc250k/condition_scaler.json
```

### Step 2: Inspect the condition distribution

```bash
sbatch slurm/run_zinc_condition_diagnostic.sh
```

This CPU-only job must be checked before starting model training. It reports the QED and heavy-atom-count frequencies and their joint distribution.

Expected output:

```text
outputs/metrics/condition_distribution/
├── condition_distribution.json
├── qed_frequency.csv
├── qed_frequency.png
├── node_count_frequency.csv
├── node_count_frequency.png
├── joint_qed_node_count_frequency.csv
└── joint_qed_node_count_frequency.png
```

Review these results before submitting the GPU jobs. In particular, check whether low- or high-QED regions contain too few training molecules.

### Step 3: Train the molecular autoencoder

```bash
sbatch slurm/run_zinc_autoencoder_full.sh
```

Expected checkpoint:

```text
outputs/checkpoints/autoencoder/best.pt
```

### Step 4: Add QED supervision

```bash
sbatch slurm/run_zinc_property_aware_autoencoder.sh
```

Expected checkpoint:

```text
outputs/checkpoints/property_aware_autoencoder/best.pt
```

### Step 5: Extract node embeddings

```bash
sbatch slurm/run_zinc_embedding_extraction.sh
```

Expected output:

```text
outputs/embeddings/zinc250k/
```

Embedding normalization statistics are computed from training molecules only.

### Step 6: Smoke-test the conditional flow

```bash
sbatch slurm/run_zinc_conditional_flow_smoke.sh
```

Run this before the full flow job to catch configuration, memory, or numerical errors.

### Step 7: Train the full conditional flow

```bash
sbatch slurm/run_zinc_conditional_flow_full.sh
```

Expected checkpoint:

```text
outputs/checkpoints/conditional_flow/best.pt
```

### Step 8: Generate and evaluate 10,000 molecules

```bash
sbatch slurm/run_zinc_generation_10k.sh
```

Expected output:

```text
outputs/samples/
```

Generation samples QED and heavy-atom count jointly from the training distribution unless fixed conditions are supplied.

## Outputs and logs

```text
logs/                         SLURM output and error logs
outputs/checkpoints/          Best and latest model checkpoints
outputs/embeddings/           Extracted node embeddings
outputs/samples/              Generated molecules and metrics
```

To inspect a running job:

```bash
squeue -u "$USER"
```

To inspect logs:

```bash
tail -f logs/<job-output-file>
tail -f logs/<job-error-file>
```

## Resume interrupted training

Training jobs save:

```text
best.pt
last.pt
```

Resume from `last.pt` using:

```bash
--resume <checkpoint-directory>/last.pt
```

The exact command-line options for any stage can be viewed with:

```bash
python scripts/<script-name>.py --help
```

## Generation results

Raw and chemistry-constrained decoding are reported separately.

The generated-molecule report includes:

* validity and connectivity;
* uniqueness and training-set novelty;
* actual RDKit QED;
* requested-versus-actual QED correlation;
* QED MAE, RMSE, and target-hit rate.

Constrained decoding uses the model’s predicted bond probabilities while enforcing atom-specific valence limits and molecular connectivity. It does not use ground-truth bonds.

## Important files

```text
configs/zinc250k.yaml             Full experiment configuration
scripts/                          Executable pipeline stages
slurm/                            Cluster job scripts
zinc_gnf/models/                  Autoencoder and conditional GNF
zinc_gnf/data/                    Preprocessing, batching and embeddings
zinc_gnf/training/                Training and validation
zinc_gnf/evaluation/              Generation and evaluation
zinc_gnf/chemistry.py             Molecular encoding and decoding
tests/                            Automated tests
```

The validation split is used for checkpoint selection. The test split must remain untouched until the final configuration is fixed.
