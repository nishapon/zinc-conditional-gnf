#!/bin/bash
#SBATCH --job-name=zinc_flow_smoke
#SBATCH --output=outputs/logs/%x_%j.log
#SBATCH --error=outputs/logs/%x_%j.err
#SBATCH --mail-user=ponnambalam@uni-hildesheim.de
#SBATCH --mail-type=END,FAIL
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:40:00

# ---------------------------------------------------------------------------
# GPU SMOKE TEST of the unconditional node-feature graph normalizing flow.
# Short on purpose: it proves the pipeline runs on a real GPU, not that the
# model is any good.
#
# Three stages:
#   1. a GPU numerics check of the real model: forward/inverse round trip and
#      the log-determinant against an autograd Jacobian, in float32 on the GPU
#      (CPU unit tests cannot catch a GPU-specific masking or dtype problem);
#   2. 300 training steps on 20,000 training molecules (validation split only;
#      the test split is never loaded);
#   3. 200 generated molecules decoded and scored, so validity / uniqueness /
#      novelty / N.U.V. and the decoder fallback rate are exercised end to end.
#
# Prerequisites:
#   data/processed/zinc250k_charge_aware_moflow_fixed/splits.pkl
#   outputs/ablation/ae_degree/seed_3/last.pt   (the frozen autoencoder)
# Neither is modified: the autoencoder is opened read-only and its SHA-256 is
# recorded in the flow's config.json and checkpoints.
#
#     sbatch slurm/04_flow_smoke_gpu.sh
#
# Resubmitting resumes from outputs/checkpoints/flow_smoke/last.pt.
# Never run this file with `bash` directly on the master node.
# ---------------------------------------------------------------------------

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "ERROR: not running under SLURM. Submit with sbatch; never run" >&2
    echo "training on the master node." >&2
    exit 64
fi

on_error() {
    local rc=$?
    echo "==============================================" >&2
    echo "JOB FAILED  rc=${rc}  line=${1:-?}  command: ${BASH_COMMAND}" >&2
    echo "flow rc: 2=data/checkpoint missing, 3=resume or autoencoder mismatch," >&2
    echo "         4=bad args/config, 5=non-finite loss, 130=interrupted, 1=traceback" >&2
    echo "inspect: sacct -j ${SLURM_JOB_ID} --format=JobID,State,ExitCode,Elapsed,MaxRSS,NodeList" >&2
    echo "         tail -50 ${OUT:-outputs/checkpoints/flow_smoke}/train.log" >&2
    echo "==============================================" >&2
}
trap 'on_error $LINENO' ERR

WORKDIR="${SLURM_SUBMIT_DIR:-$HOME/zinc-conditional-gnf}"
cd "${WORKDIR}"

PYTHON="${ZINC_GNF_PYTHON:-}"
if [[ -z "${PYTHON}" ]]; then
    for candidate in "$HOME/miniconda3/envs/zinc_gnf/bin/python" \
                     "$HOME/anaconda3/envs/zinc_gnf/bin/python" \
                     "$HOME/miniforge3/envs/zinc_gnf/bin/python"; do
        if [[ -x "${candidate}" ]]; then PYTHON="${candidate}"; break; fi
    done
fi
if [[ -z "${PYTHON}" || ! -x "${PYTHON}" ]]; then
    echo "ERROR: could not find the python of the Conda env 'zinc_gnf'." >&2
    echo "Resubmit with: sbatch --export=ALL,ZINC_GNF_PYTHON=/path/to/envs/zinc_gnf/bin/python slurm/04_flow_smoke_gpu.sh" >&2
    exit 65
fi

export PYTHONUNBUFFERED=1
export PYTHONPATH="${WORKDIR}${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export WANDB_MODE=offline

DATA_DIR="data/processed/zinc250k_charge_aware_moflow_fixed"
AE_CHECKPOINT="${ZINC_GNF_AE:-outputs/ablation/ae_degree/seed_3/last.pt}"
OUT="outputs/checkpoints/flow_smoke"
SAMPLES="outputs/samples/flow_smoke"
mkdir -p outputs/logs "${OUT}" "${SAMPLES}"

echo "=============================================="
echo "Unconditional flow GPU smoke test"
echo "Start time:       $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Job:              ${SLURM_JOB_ID} (${SLURM_JOB_NAME:-?})"
echo "Node:             $(hostname)"
echo "Partition:        ${SLURM_JOB_PARTITION:-unknown}"
echo "CPUs / memory:    ${SLURM_CPUS_PER_TASK:-?} / ${SLURM_MEM_PER_NODE:-?} MB"
echo "CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES:-}'"
echo "Workdir:          ${WORKDIR}"
echo "Python:           ${PYTHON}"
echo "Git commit:       $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "Autoencoder:      ${AE_CHECKPOINT} (read-only, frozen)"
echo "Output:           ${OUT}"
echo "=============================================="
"${PYTHON}" -V
nvidia-smi || { echo "ERROR: nvidia-smi failed: no GPU visible to this job" >&2; exit 67; }

if [[ ! -f "${DATA_DIR}/splits.pkl" ]]; then
    echo "ERROR: ${DATA_DIR}/splits.pkl not found." >&2
    echo "Run the CPU derivation job first: sbatch slurm/01d_derive_moflow_split.sh" >&2
    exit 66
fi
if [[ ! -f "${AE_CHECKPOINT}" ]]; then
    echo "ERROR: frozen autoencoder ${AE_CHECKPOINT} not found." >&2
    echo "Expected the selected learned-degree checkpoint (seed 3, last.pt)." >&2
    exit 66
fi
ls -la "${AE_CHECKPOINT}"
"${PYTHON}" -c "import hashlib,sys;print('autoencoder sha256:',hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" "${AE_CHECKPOINT}"

# GPU preflight plus a numerics check of the REAL model on the GPU.
"${PYTHON}" - <<'PYCHECK'
import sys
import torch
print("torch", torch.__version__, "| cuda build:", torch.version.cuda)
if not torch.cuda.is_available():
    sys.exit("ERROR: torch.cuda.is_available() is False. The Conda env needs the GPU build of "
             "PyTorch (pip3 install torch --index-url https://download.pytorch.org/whl/cu121).")
print("GPU:", torch.cuda.get_device_name(0),
      "| memory GB", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1))
import rdkit, numpy, scipy, yaml
print("dependency check ok; rdkit", rdkit.__version__, "| scipy", scipy.__version__)

from zinc_gnf.flow import FlowConfig, NodeFlow, node_mask_from_counts
from zinc_gnf.utils.config import load_config

config = FlowConfig.from_dict(load_config("configs/zinc250k_flow_uncond.yaml")["flow"])
flow = NodeFlow(config, init_seed=0).cuda()
flow.set_data_statistics(torch.zeros(config.node_embedding_dim),
                         torch.ones(config.node_embedding_dim))
print(f"flow parameters: {flow.num_parameters():,}")
mask = node_mask_from_counts([38, 17, 5], device="cuda")
x = torch.randn((3, 38, config.node_embedding_dim), device="cuda") * mask.unsqueeze(-1)
flow.train()
with torch.no_grad():
    flow(x, mask)                                  # data-dependent ActNorm initialization
flow.eval()


def round_trip_error():
    with torch.no_grad():
        z, log_det = flow(x, mask)
        back = flow.inverse(z, mask)
    assert torch.isfinite(z).all() and torch.isfinite(log_det).all()
    assert float(z[1, 17:].abs().max()) == 0.0, "padding leaked into the latent"
    relative = float((back - x).abs().max()) / max(1e-9, float(x.abs().max()))
    nodes = int(mask.sum())
    return relative, float(z.abs().max()), float(log_det.sum()) / nodes


# 1. As initialized the flow is the identity on whitened data, so the round trip is exact
#    up to float32 rounding.
relative, z_max, log_det_per_node = round_trip_error()
print(f"initialized: round-trip relative error {relative:.3e}, max|z| {z_max:.3f}, "
      f"log|det J|/node {log_det_per_node:.3f}")
assert relative < 1e-4, "forward/inverse disagree on the GPU at initialization"

# 2. Perturb every parameter slightly so the map is non-trivial, and check again.
#    The tolerance is relative on purpose: a flow of 12 blocks with max_log_scale 1.0 can
#    amplify by up to exp(24) if its parameters are pushed into a pathological regime, and
#    inverting such a map in float32 is meaningless. Training keeps log|det J|/node moderate,
#    which the trainer logs every step; a runaway there is the divergence signature to watch.
with torch.no_grad():
    for parameter in flow.parameters():
        parameter.add_(0.01 * torch.randn(parameter.shape, device="cuda"))
relative, z_max, log_det_per_node = round_trip_error()
print(f"perturbed:   round-trip relative error {relative:.3e}, max|z| {z_max:.3f}, "
      f"log|det J|/node {log_det_per_node:.3f}")
assert relative < 1e-3, "forward/inverse disagree on the GPU after a small perturbation"

# 3. The reported log-determinant must equal the autograd Jacobian's, for a small graph.
small_mask = node_mask_from_counts([4], device="cuda")
small_x = torch.randn((1, 4, config.node_embedding_dim), device="cuda")
def transform(flat):
    out, _ = flow(flat.view(1, 4, config.node_embedding_dim), small_mask)
    return out.reshape(-1)
jacobian = torch.autograd.functional.jacobian(transform, small_x.reshape(-1), vectorize=True)
with torch.no_grad():
    _, reported = flow(small_x, small_mask)
_, numeric = torch.linalg.slogdet(jacobian.double())
print(f"log|det J|: reported {float(reported[0]):.5f} vs autograd {float(numeric):.5f}")
assert abs(float(reported[0]) - float(numeric)) < 1e-2, "log-determinant is wrong on the GPU"
print("GPU numerics check passed")
PYCHECK

echo
echo "---- training (smoke: 300 steps on 20,000 molecules) ----"
srun "${PYTHON}" -u scripts/04_train_flow.py \
    --config configs/zinc250k_flow_uncond.yaml \
    --data-dir "${DATA_DIR}" \
    --ae-checkpoint "${AE_CHECKPOINT}" \
    --output-dir "${OUT}" \
    --device cuda \
    --seed 1 \
    --batch-size 128 \
    --max-steps 300 \
    --max-train-molecules 20000 \
    --statistics-molecules 5000 \
    --val-molecules 1000 \
    --eval-every 100 \
    --save-every 100 \
    --log-every 25 \
    --resume auto

echo
echo "---- generation (smoke: 200 molecules at T=1.0 and T=0.85) ----"
srun "${PYTHON}" -u scripts/05_generate.py \
    --flow-checkpoint "${OUT}/best.pt" \
    --ae-checkpoint "${AE_CHECKPOINT}" \
    --data-dir "${DATA_DIR}" \
    --output-dir "${SAMPLES}" \
    --count 200 \
    --temperature 0.85,1.0 \
    --batch-size 64 \
    --seed 0 \
    --device cuda

echo "=============================================="
echo "Finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
ls -la "${OUT}" "${SAMPLES}"
echo "Training log:   ${OUT}/train.log"
echo "Metrics:        ${OUT}/metrics.jsonl , ${OUT}/summary.json"
echo "Samples:        ${SAMPLES}/T*/metrics.txt , ${SAMPLES}/T*/valid_smiles.txt"
echo
echo "A smoke run is NOT a result: 300 steps cannot produce a good density."
echo "What to check: the loss is finite and falling, the GPU numerics check passed,"
echo "some molecules decode, and the degree decoder's fallback rate is small."
echo "Next, if all of that holds: sbatch slurm/04_flow_uncond_gpu.sh"
echo "=============================================="
