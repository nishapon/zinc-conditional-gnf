# Conditional Graph Normalizing Flow for ZINC250K

Conditional molecular graph generation using a bond-aware molecular
autoencoder and a node-level graph normalizing flow.

The model learns `p(H | QED, N)`, where `H` is the set of molecular
node embeddings, `QED` is the requested drug-likeness score, and `N`
is the requested heavy-atom count.

This project extends the molecular pipeline from
[graph-normalising-flow](https://github.com/sridharmatta1/graph-normalising-flow/tree/main/molecular_generation)
from QM9 to ZINC250K and introduces explicit QED conditioning.

## Pipeline

1. Convert ZINC SMILES into atom, charge, hydrogen and bond tensors.
2. Create scaffold-disjoint training, validation and test splits.
3. Train a bond-aware molecular graph autoencoder.
4. Add QED supervision to the node-embedding space.
5. Freeze the property-aware autoencoder.
6. Extract and normalize node embeddings.
7. Train the conditional node-level GNF.
8. Sample embeddings conditioned on QED and heavy-atom count.
9. Decode embeddings into molecular graphs.
10. Evaluate raw and chemistry-constrained decoding.

## Conditions

- RDKit QED
- `log1p(heavy_atom_count)`

Conditions are standardized using training-split statistics only.

## Molecular representation

- Atoms: `C, N, O, F, P, S, Cl, Br, I`
- Formal charges: `-1, 0, +1`
- Hydrogen counts: `0, 1, 2, 3, 4`
- Bonds: no bond, single, double, triple
- Aromatic structures use Kekulé single and double bonds
- Stereochemistry is not modeled

## Repository structure

```text
configs/                 Experiment configurations
scripts/                 Pipeline command-line interfaces
slurm/                   Cluster jobs
zinc_gnf/chemistry.py    Molecular representation and decoding
zinc_gnf/data/           Preprocessing, batching and embeddings
zinc_gnf/models/         Autoencoder and conditional GNF
zinc_gnf/training/       Training and validation engines
zinc_gnf/evaluation/     Generation and molecular metrics
tests/                   Automated tests
```

## Installation

```bash
git clone https://github.com/nishapon/zinc-conditional-gnf.git
cd zinc-conditional-gnf
git switch refactor/modular-zinc-pipeline

conda create -n zinc_gnf python=3.11 -y
conda activate zinc_gnf
pip install -e .
pip install pytest
pytest -q
```

Place the ZINC250K CSV at:

```text
data/raw/zinc250k.csv
```

## Pipeline commands

### 1. Preprocess ZINC250K

```bash
python scripts/preprocess_zinc.py \
  --csv data/raw/zinc250k.csv \
  --output-dir data/processed/zinc250k \
  --target-molecules 0 \
  --candidate-count 0 \
  --seed 42 \
  --max-nodes 38 \
  --train-fraction 0.80 \
  --validation-fraction 0.10
```

The split is Murcko-scaffold-disjoint. A value of `0` processes all
eligible molecules.

### 2. Train the molecular autoencoder

```bash
python scripts/train_autoencoder.py \
  --config configs/zinc250k.yaml \
  --splits data/processed/zinc250k/zinc_splits.pkl \
  --output-dir outputs/checkpoints/autoencoder \
  --device cuda \
  --epochs 30
```

### 3. Train the property-aware autoencoder

```bash
python scripts/train_property_aware_autoencoder.py \
  --config configs/zinc250k.yaml \
  --splits data/processed/zinc250k/zinc_splits.pkl \
  --source-checkpoint outputs/checkpoints/autoencoder/best.pt \
  --output-dir outputs/checkpoints/property_aware_autoencoder \
  --device cuda \
  --warmup-epochs 5 \
  --joint-epochs 25 \
  --qed-loss-weight 1.0
```

### 4. Extract node embeddings

```bash
python scripts/extract_embeddings.py \
  --config configs/zinc250k.yaml \
  --splits data/processed/zinc250k/zinc_splits.pkl \
  --checkpoint outputs/checkpoints/property_aware_autoencoder/best.pt \
  --output-dir outputs/embeddings/zinc250k \
  --device cuda \
  --batch-size 32 \
  --shard-size 2048 \
  --noise-seed 42 \
  --overwrite
```

### 5. Smoke-test the conditional flow

```bash
python scripts/train_flow.py \
  --config configs/zinc250k.yaml \
  --embeddings outputs/embeddings/zinc250k \
  --output-dir outputs/checkpoints/conditional_flow_smoke \
  --device cuda \
  --epochs 2 \
  --batch-size 32
```

### 6. Train the conditional flow

```bash
python scripts/train_flow.py \
  --config configs/zinc250k.yaml \
  --embeddings outputs/embeddings/zinc250k \
  --output-dir outputs/checkpoints/conditional_flow \
  --device cuda \
  --epochs 40 \
  --batch-size 32
```

### 7. Generate 10,000 molecules

```bash
python scripts/generate_molecules.py \
  --config configs/zinc250k.yaml \
  --splits data/processed/zinc250k/zinc_splits.pkl \
  --autoencoder-checkpoint outputs/checkpoints/property_aware_autoencoder/best.pt \
  --flow-checkpoint outputs/checkpoints/conditional_flow/best.pt \
  --embeddings outputs/embeddings/zinc250k \
  --output-dir outputs/samples/zinc250k_seed_42 \
  --device cuda \
  --num-samples 10000 \
  --batch-size 64 \
  --temperature 1.0 \
  --seed 42
```

By default, QED and node count are sampled jointly from the empirical
training distribution. Fixed conditions can be requested using
`--fixed-qed` and `--fixed-n-node`.

## SLURM

Submit the jobs sequentially:

```bash
sbatch slurm/run_zinc_preprocessing.sh
sbatch slurm/run_zinc_autoencoder_full.sh
sbatch slurm/run_zinc_property_aware_autoencoder.sh
sbatch slurm/run_zinc_embedding_extraction.sh
sbatch slurm/run_zinc_conditional_flow_smoke.sh
sbatch slurm/run_zinc_conditional_flow_full.sh
sbatch slurm/run_zinc_generation_10k.sh
```

Each stage should be checked before submitting the next job.

## Molecular decoding

Raw decoding directly selects the highest-probability atom, charge,
hydrogen and bond classes.

Constrained decoding still uses only model-predicted probabilities,
but limits bonds using atom- and charge-specific valence capacities
and constructs a connected molecular graph. Ground-truth bonds are
never used during generation.

## Evaluation

Raw and constrained decoding are reported separately using:

- sanitizability
- connectivity
- radical-free domain validity
- uniqueness
- novelty against the training split
- actual RDKit QED
- requested-versus-actual QED correlation
- QED MAE and RMSE
- QED target-hit rate within `±0.05`

## Resume training

Autoencoder and flow jobs save both `best.pt` and `last.pt`.
Interrupted training can be resumed with:

```bash
--resume <output-directory>/last.pt
```

## Data-leakage policy

- Normalization statistics use training data only.
- Validation is used for checkpoint selection and early stopping.
- The test split is untouched during model development.
- Final test evaluation is performed after hyperparameters are fixed.

## Testing

```bash
pytest -q
```

Datasets, checkpoints, generated molecules and logs are excluded from
Git.
