# Conditional Graph Normalizing Flow for ZINC

Conditional molecular graph generation using a property-aware graph
autoencoder and a node-level graph normalizing flow.

## Conditions

The model is conditioned on:

- Molecular QED
- Heavy-atom count

## Pipeline

1. Preprocess ZINC molecules.
2. Train the bond-aware molecular autoencoder.
3. Fine-tune embeddings with QED supervision.
4. Extract and normalize node embeddings.
5. Train the conditional node-level GNF.
6. Generate molecules from the conditional prior.
7. Decode using raw and valence-aware constrained decoding.
8. Evaluate validity, uniqueness, novelty and QED control.

## Current experiment

The current pilot uses a scaffold-disjoint subset of 10,000 molecules:

- 8,000 training molecules
- 1,000 validation molecules
- 1,000 untouched test molecules

This repository contains source code and configurations only.
Datasets, checkpoints and generated outputs are excluded from Git.
