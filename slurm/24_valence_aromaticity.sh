#!/bin/bash
#SBATCH --job-name=zinc_valence
#SBATCH --output=outputs/logs/%x_%j.log
#SBATCH --error=outputs/logs/%x_%j.err
#SBATCH --mail-user=ponnambalam@uni-hildesheim.de
#SBATCH --mail-type=END,FAIL
#SBATCH --partition=CPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00

# ---------------------------------------------------------------------------
# STAGE 24: why are the bond COUNTS right and the molecules still invalid?
# CPU only. READ-ONLY. No training. No checkpoint is written. No new training run.
#
# WHAT STAGE 23 ESTABLISHED (job at 18,000 steps, outputs/checkpoints/graph_flow_topology)
#   aggregate raw validity 29.73%, corrected exact-N 79.5%, corrected QED slope +0.0818
#   against a shuffled-QED floor of +0.0049, aromatic rings 0.008 per corrected molecule.
#   At N=23, QED 0.8, T=1.0: generated bonds 24.45 against 24.65 real, components 1.20 -
#   and yet 229/256 = 89.5% of graphs carry an over-valent atom and raw validity is 7.81%.
#   At N=30: bonds 32.98 against 32.99, and 96.9% over-valent.
#
#   That shape - correct global marginals, wrong local correlations - is the whole
#   question. The model has learned HOW MANY bonds a molecule of this size has. It has
#   not learned WHICH ATOM may carry them.
#
# THE TWO HYPOTHESES THIS JOB TESTS, both already confirmed by code inspection and now
# measured on the real checkpoint:
#
#   H-VALENCE  The flow factorises as log p(G|c) = log p_bond(x_b|c) + log p_atom(x_a|A,c)
#              (graph_flow.py, `encode`): bonds are generated FIRST and atoms are
#              conditioned on them. Valence is a constraint between an atom's TYPE and the
#              sum of its incident bond orders, so the bond block structurally CANNOT
#              respect it - it does not know the atom types yet. The atom block is the only
#              stage that could, and it is choosing against a marginal that is 73% carbon,
#              cap 4. Prediction: for most graphs there EXISTS an atom assignment making
#              every atom feasible, i.e. the BOND pattern is realisable and the ATOM
#              CHOICE is what fails. `best_case_atom_assignment` measures exactly that.
#
#   H-AROMATIC `valency_correction` removes one order from the LARGEST-order bond of an
#              over-valent atom. In a Kekule aromatic ring the largest-order bonds are
#              precisely the DOUBLE bonds, so the corrector preferentially destroys the
#              alternation that aromaticity is perceived from - and at 89.5% over-valence
#              almost every molecule passes through that rule. `correction_damage`
#              measures aromaticity before AND after correction on the SAME graphs, which
#              separates "the model never made an aromatic ring" from "the repair
#              destroyed it". Neither explanation is assumed; both are measured.
#
# WHAT IT RUNS (one python process)
#   B-first  the CEILING: aromatic rings in REAL training molecules pushed through our own
#            serialise -> Kekulise -> reconstruct path. If real molecules lose aromaticity
#            there, the representation is the bug and no model could keep it. Everything
#            generated is read against this number, never against RDKit in the abstract.
#   A        over-valence by element, formal charge and allowed valence; bond-order-sum
#            histograms per atom type; the magnitude of the excess; the joint
#            P(atom type, bond-order sum) against real ZINC250K; whether argmax
#            discretisation itself moves the atom marginal or only sharpens it; and how
#            all of it scales with N (15, 23, 30), requested QED (0.4, 0.8) and
#            temperature (1.0, 0.85).
#   C        attribution by read-only counterfactual on the EXISTING checkpoint:
#            best-case relabelling (generated bonds, atoms relabelled as favourably as
#            possible), real atoms with generated bonds, generated atoms with real bonds,
#            and correction damage. Ring sizes and the single/double alternation pattern
#            of 5- and 6-rings are counted with networkx, NOT RDKit, because ~92% of
#            graphs at N=23 are unsanitizable and that is exactly the population at issue.
#
# GUARANTEES
#   * the checkpoint is opened read-only and nothing is trained or saved over it;
#   * no molecule is scored or selected by QED anywhere - the oracle arms substitute real
#     STRUCTURES (atom types, bond matrices) as a diagnostic reference, which is labelled
#     as such in every output field, and never as a generation aid;
#   * the test split is never loaded;
#   * writes ONLY under outputs/metrics/graph_flow_topology/valence_audit. Stage 1-23
#     checkpoints, outputs and code are untouched.
#
#     sbatch slurm/24_valence_aromaticity.sh
#     COUNT=512 sbatch slurm/24_valence_aromaticity.sh        # tighter intervals
#     TARGET_N=23 sbatch slurm/24_valence_aromaticity.sh      # the single decisive cell
#
# Runtime estimate: 12 grid cells (3 N x 2 QED x 2 T) at 256 samples. Sampling is one
# inverse pass on CPU (seconds per cell); the cost is RDKit sanitisation and the cycle
# basis, roughly 4 x 256 sanitisations per cell including the two oracle arms. Stage 22c's
# comparable CPU diagnostic took about 25 min for a similar volume, so expect 30-60 min.
# The 3 h limit is headroom, and the job is cheap to re-run.
# Memory estimate: the train split in memory (~1.5 GB as python records) plus a 1.3M
# parameter model on CPU and 256 graphs of at most 38 atoms. 16 G is ample.
#
# Never run this file with `bash` directly on the login node: it loads the full train
# split and samples 3,072 graphs.
# ---------------------------------------------------------------------------

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "ERROR: not running under SLURM. Submit with sbatch; this loads the full train split." >&2
    exit 64
fi

on_error() {
    local rc=$?
    echo "==============================================" >&2
    echo "JOB FAILED  rc=${rc}  line=${1:-?}  command: ${BASH_COMMAND}" >&2
    echo "rc: 2=missing checkpoint or dataset, 4=bad arguments, 1=traceback" >&2
    echo "inspect: sacct -j ${SLURM_JOB_ID} --format=JobID,State,ExitCode,Elapsed,MaxRSS" >&2
    echo "         tail -50 ${OUT:-outputs/metrics/graph_flow_topology/valence_audit}/audit.log" >&2
    echo "Nothing is written outside that directory, so a failure costs no state." >&2
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
    echo "Resubmit with: sbatch --export=ALL,ZINC_GNF_PYTHON=/path/to/envs/zinc_gnf/bin/python slurm/24_valence_aromaticity.sh" >&2
    exit 65
fi

export PYTHONUNBUFFERED=1
export PYTHONPATH="${WORKDIR}${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

DATA_DIR="data/processed/zinc250k_charge_aware_moflow_fixed"
CHECKPOINT="${CHECKPOINT:-outputs/checkpoints/graph_flow_topology/last.pt}"
OUT="${OUT:-outputs/metrics/graph_flow_topology/valence_audit}"
TARGET_N="${TARGET_N:-15,23,30}"
TARGET_QED="${TARGET_QED:-0.4,0.8}"
TEMPERATURES="${TEMPERATURES:-1.0,0.85}"
COUNT="${COUNT:-256}"
REAL_MOLECULES="${REAL_MOLECULES:-512}"
mkdir -p outputs/logs "${OUT}"

echo "=============================================="
echo "Stage 24: valence and aromaticity audit"
echo "Start time:       $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Job:              ${SLURM_JOB_ID} (${SLURM_JOB_NAME:-?})"
echo "Node:             $(hostname)"
echo "Partition:        ${SLURM_JOB_PARTITION:-unknown}"
echo "CPUs / memory:    ${SLURM_CPUS_PER_TASK:-?} / ${SLURM_MEM_PER_NODE:-?} MB"
echo "Git commit:       $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "Checkpoint:       ${CHECKPOINT} (READ-ONLY, stage 23)"
echo "Grid:             N ${TARGET_N} x QED ${TARGET_QED} x T ${TEMPERATURES}, ${COUNT} each"
echo "Real reference:   ${REAL_MOLECULES} train molecules per cell (the ceiling)"
echo "Writes only:      ${OUT}"
echo "No training, no checkpoint written, no molecule selected by property"
echo "=============================================="
"${PYTHON}" -V

for required in "${CHECKPOINT}" "${DATA_DIR}/splits.pkl" "${DATA_DIR}/scaler.json"; do
    if [[ ! -f "${required}" ]]; then
        echo "ERROR: ${required} not found." >&2
        exit 66
    fi
done

srun "${PYTHON}" -u scripts/26_valence_aromaticity_audit.py \
    --checkpoint "${CHECKPOINT}" \
    --output-dir "${OUT}" \
    --data-dir "${DATA_DIR}" \
    --target-n "${TARGET_N}" \
    --target-qed "${TARGET_QED}" \
    --temperature "${TEMPERATURES}" \
    --count "${COUNT}" \
    --real-molecules "${REAL_MOLECULES}" \
    --seed 17

echo "=============================================="
echo "Finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
ls -la "${OUT}"
echo
echo "READ IN THIS ORDER:"
echo "  0. the 'aromaticity ceiling on REAL training molecules' block in"
echo "     ${OUT}/audit.log. If reconstructable is not ~100% and real aromatic rings per"
echo "     molecule is not ~2, the REPRESENTATION loses aromaticity and nothing"
echo "     downstream can be interpreted - fix that first and ignore the rest."
echo "  1. 'COUNTERFACTUAL best-case relabelling' at N=23, QED 0.8, T=1.0. This is the"
echo "     decisive number. A HIGH feasible rate beside 7.81% raw validity means the"
echo "     generated BOND pattern is chemically realisable and the ATOM CHOICE is the"
echo "     failure - which is what the bonds-first factorisation predicts, since the bond"
echo "     block never sees atom types. A LOW rate instead means the bond distribution"
echo "     itself is wrong and the atom block is not to blame."
echo "  2. 'infeasible for ANY atom type' in the same cell: atoms whose bond-order sum"
echo "     exceeds 6, the largest cap in the vocabulary. These cannot be repaired by any"
echo "     relabelling. A near-zero share supports H-VALENCE; a large share refutes it."
echo "  3. the two oracle counterfactuals. Real atoms + generated bonds isolates the atom"
echo "     block's contribution; generated atoms + real bonds isolates the bond block's."
echo "     Whichever substitution recovers more validity names the stage that is wrong."
echo "  4. 'aromatic rings X -> Y per graph' and 'alternating rings destroyed'. A large"
echo "     destroyed count means the REPAIR is the proximate cause of 0.008 aromatic"
echo "     rings; a near-zero before-count means the model never produced the"
echo "     alternation in the first place. These are different bugs with different fixes."
echo "  5. 'ring patterns (5/6)' against the real row. all_single rings are saturated"
echo "     rings, not failed benzenes: a Kekule 6-ring is only perceived aromatic when"
echo "     the single/double alternation is exactly right."
echo "  6. 'argmax vs soft atom marginal'. If argmax concentrates on carbon far beyond"
echo "     the soft scores, DISCRETISATION is part of the marginal error and not the"
echo "     density - which would be a decoding fix, not an architecture one."
echo
echo "This job diagnoses. It licenses the stage-25 design change; it is not itself one,"
echo "and it makes no claim about validity after correction."
echo "=============================================="
