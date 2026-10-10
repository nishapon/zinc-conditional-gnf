# Conditional Graph Normalizing Flow for ZINC250K

A conditional molecular graph generator for ZINC250K, benchmarked against
MoFlow (Zang & Wang, 2020). Extends the Liu et al. 2019 Graph Normalizing
Flow with Feature-wise Linear Modulation (FiLM) conditioning on
`(heavy-atom count N, standardized QED)`, a conditional Gaussian prior
parameterized by a small MLP, and a chemistry-aware categorical decoder
with valence-aware greedy bond construction.

This repository is the ZINC250K side of a 4-person student research
project. Teammates handle QM9 in a separate repository.

## Pipeline

Six stages, each in its own module and script. Each stage writes
checkpoints and metrics that the next stage reads.

1. **Preprocess** ZINC250K SMILES into `(atom_types, bond_matrix)` graphs,
   with a scaffold-free random 80/10/10 split and a condition scaler fit
   on the training split only.
2. **Train** a bond-aware molecular autoencoder (atom + bond
   reconstruction, iterative bond refinement with soft remaining
   valence).
3. **Extract** per-molecule node embeddings and compute embedding
   mean/std from the training split only.
4. **Train** a conditional node-level graph normalizing flow with FiLM
   conditioning and a conditional Gaussian prior
   `z ~ N(μ(y), diag(σ(y))² · I)`.
5. **Generate** molecules by sampling the conditional prior, running the
   inverse flow, and decoding with valence-aware constrained bond
   assembly.
6. **Evaluate** against MoFlow metrics (validity / uniqueness / novelty /
   N.U.V. / reconstruction) and QED-specific metrics (requested-vs-actual
   correlation, MAE, target-hit rate).

## Molecular representations

Two representations coexist; outputs live in separate directories.

| | Baseline (neutral-only) | Charge-aware |
|---|---|---|
| Version | `zinc250k_neutral_kekule_v1` | `zinc250k_charge_aware_kekule_v2` |
| Atom types | 9 elements | 17 = 9 elements + 8 measured charge types |
| ZINC250K coverage | 168,129 / 249,455 (67.40%) | 249,454 / 249,455 (one phosphine-oxide limitation) |
| Code | `constants.py`, `chemistry.py`, `preprocessing.py` | `constants_charge_aware.py`, `chemistry_charge_aware.py`, `preprocessing_charge_aware.py` |
| Data | `data/processed/zinc250k/` | `data/processed/zinc250k_charge_aware/` |
| Run | `slurm/01_preprocess.sh` | `slurm/01b_preprocess_charge_aware.sh` (see `slurm/README_charge_aware.md`) |

Model code should read vocabulary size and valence from
`zinc_gnf.representations.get_representation(...)`, not from `constants.py`.

## Install

Clone, create a Python 3.11 environment, install editable:

```bash
git clone https://github.com/nishapon/zinc-conditional-gnf.git
cd zinc-conditional-gnf
git switch rewrite/zinc250k-pipeline

conda create -n zinc_gnf python=3.11 -y
conda activate zinc_gnf
pip install -e .[dev]
```

Verify:

```bash
python -c "import zinc_gnf; print(zinc_gnf.__version__)"
pytest -q
```

## Cluster

All compute jobs run via SLURM. Nothing runs on the login node.

- Training jobs land on `--partition=STUD --gres=gpu:1`.
- CPU-only jobs (preprocessing, QED histogram, final metric aggregation)
  land on a CPU partition.
- Every trainer checkpoints frequently and supports `--resume last.pt`
  so no job needs to approach the 24 h limit.
- Each SLURM script picks the Python interpreter via
  `ZINC_GNF_PYTHON`, defaulting to `$HOME/miniconda3/envs/zinc_gnf/bin/python`.

## Repository layout

```text
configs/zinc250k.yaml        Full experiment configuration (hyperparameters)
zinc_gnf/
  constants.py               Atom/bond vocab and valence table
  chemistry.py               SMILES ↔ graph; valence-aware bond decoder
  preprocessing.py           Encode CSV → splits + condition scaler
  dataset.py                 PyTorch Dataset + padded batching
  encoder.py                 Bond-aware self-attention encoder
  decoder.py                 Atom + bond heads, iterative bond refinement
  autoencoder.py             Encoder + decoder wrapper + losses
  embeddings.py              Embedding extraction and normalization
  flow.py                    Conditional node GNF + conditional prior
  flow_dataset.py            Sharded embedding dataset for flow training
  trainer_autoencoder.py     Autoencoder train/eval loop
  trainer_flow.py            Flow train/eval loop
  generator.py               Sample from prior, inverse flow, decode
  metrics.py                 MoFlow metrics + QED-specific metrics
  utils/                     Config, seed, checkpointing, logging
scripts/                     CLI entry points, one per stage
slurm/                       SLURM wrappers for each stage
tests/                       Unit tests for chemistry and shapes
outputs/                     Checkpoints, embeddings, metrics, figures
```

## Status

In active development. See the branch `rewrite/zinc250k-pipeline` for
the current implementation and `main` for the stable skeleton.

## References

1. Liu, Kumar, Ba, Kiros, Swersky — Graph Normalizing Flows, NeurIPS
   2019. <https://arxiv.org/abs/1905.13177>
2. Zang, Wang — MoFlow: An Invertible Flow Model for Generating
   Molecular Graphs, KDD 2020. <https://arxiv.org/abs/2006.10137>
3. Perez, Strub, de Vries, Dumoulin, Courville — FiLM: Visual Reasoning
   with a General Conditioning Layer, AAAI 2018.
   <https://arxiv.org/abs/1709.07871>
4. Dinh, Sohl-Dickstein, Bengio — Density estimation using Real NVP,
   ICLR 2017. <https://arxiv.org/abs/1605.08803>
