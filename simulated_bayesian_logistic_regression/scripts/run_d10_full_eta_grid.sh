#!/bin/bash
#PBS -N logreg-d10
#PBS -l select=1
#PBS -l filesystems=YOUR_FILESYSTEMS
#PBS -l walltime=04:00:00
#PBS -A YOUR_ALLOCATION
#PBS -j oe
#PBS -o ./logs/

PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
SCRATCH_ROOT=${SCRATCH_ROOT:-${PROJECT_ROOT}/results}

# Production sweep for d=10 with stochastic gradients on a single GPU.
# Runs the full eta grid for FA-LD, FA-HMC, and adaptive FA-HMC.

mkdir -p "${PROJECT_ROOT}/logs"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ulimit -c 0

# Load your Python environment here. Example:
# module use /path/to/modulefiles
# module load YOUR_PYTHON_MODULE
# conda activate YOUR_ENV

cd ${PROJECT_ROOT}/simulated_bayesian_logistic_regression/scripts

DATA=${PROJECT_ROOT}/simulated_bayesian_logistic_regression/data/synthetic_data_d10.mat
OUT_DIR=${SCRATCH_ROOT}/simulated_bayesian_logistic_regression/sweep/d10_SG
mkdir -p "$OUT_DIR"

ITER_TOTAL=2000000
M=20
NOISE=10
NSAMP=1000
GPU_ID=${GPU_ID:-0}

echo "=== Start $(date) on $(hostname) ==="
nvidia-smi -L

run_cell() {
    local panel=$1 algo=$2 eta=$3 K=$4 T=$5
    local tag="${algo}_eta${eta}_K${K}"
    local out="$OUT_DIR/${tag}.npz"
    local log="$OUT_DIR/${tag}.log"
    if [ -f "$out" ]; then
        echo "SKIP $tag (exists)"
        return 0
    fi
    CUDA_VISIBLE_DEVICES=$GPU_ID python -u run_logistic_regression_cell.py \
        --panel $panel --algo $algo --data "$DATA" \
        --eta $eta --K $K --T $T --iter_total $ITER_TOTAL --M $M \
        --noise_sigma $NOISE --n_samples_target $NSAMP \
        --out "$out" --gpu 0 \
        > "$log" 2>&1
}

declare -a CELLS=()
ETAS=(0.001 0.002 0.005 0.01 0.02 0.05 0.1 0.2 0.5 1.0)

for eta in "${ETAS[@]}"; do
    CELLS+=("d|fa_ld|$eta|1|10")
done

kp() { python -c "import math, sys; print(max(1, round(math.pi / (3 * float(sys.argv[1])))))" "$1"; }

k_candidates() {
    local eta=$1
    local k0=$(kp $eta)
    if python -c "import sys; sys.exit(0 if float(sys.argv[1]) <= 0.01 else 1)" "$eta"; then
        echo "$k0"
    else
        local half=$((k0 / 2)); [ $half -lt 1 ] && half=1
        local doub=$((k0 * 2))
        if [ $half -eq $k0 ]; then
            echo "$k0 $doub"
        else
            echo "$half $k0 $doub"
        fi
    fi
}

for eta in "${ETAS[@]}"; do
    for K in $(k_candidates "$eta"); do
        CELLS+=("d|fa_hmc|$eta|$K|10")
        Ka=$((K * 10))
        CELLS+=("d|adaptive|$eta|$Ka|1")
    done
done

echo "Total cells: ${#CELLS[@]}"
echo "Using GPU_ID=$GPU_ID"

fail=0
for cell in "${CELLS[@]}"; do
    IFS='|' read -r p algo eta K T <<<"$cell"
    run_cell $p $algo $eta $K $T || fail=$((fail+1))
done

echo "Failed: $fail"
echo "=== Done $(date) ==="
