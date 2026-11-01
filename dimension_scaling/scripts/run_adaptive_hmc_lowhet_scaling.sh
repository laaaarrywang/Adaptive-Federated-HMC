#!/bin/bash
#PBS -N afhmc-lowhet
#PBS -l select=1
#PBS -l filesystems=YOUR_FILESYSTEMS
#PBS -l walltime=12:00:00
#PBS -A YOUR_ALLOCATION
#PBS -j oe
#PBS -o ./logs/

PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
SCRATCH_ROOT=${SCRATCH_ROOT:-${PROJECT_ROOT}/results}

# Low-heterogeneity Gaussian scaling experiment.
#
# For each client, draw SAMPLES_PER_CLIENT samples from N(16.2 * 1_d, 1.6 I_d),
# estimate that client's Gaussian, and use the pooled samples from all clients
# to define the W2 target. AFHMC uses K(d)=round(50*(d/100)^(1/2)) and
# eta(d)=0.006*(d/100)^(-1/2). Each seed controls both the client samples and
# the HMC noise.

mkdir -p "${PROJECT_ROOT}/logs"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ulimit -c 0

# Load your Python environment here. Example:
# module use /path/to/modulefiles
# module load YOUR_PYTHON_MODULE
# conda activate YOUR_ENV

cd ${PROJECT_ROOT}/dimension_scaling/scripts

OUT_BASE=${SCRATCH_ROOT}/dimension_scaling/lowhet_adaptive_hmc_scaling
DIMS=${DIMS:-"2 50 100 150 200 250 300 350 400 450 500"}
SEEDS=${SEEDS:-"0 1 2"}
SAMPLES_PER_CLIENT=${SAMPLES_PER_CLIENT:-100000}
MAX_ROUNDS=${MAX_ROUNDS:-1000}
GPU_ID=${GPU_ID:-0}

echo "=== Start $(date) on $(hostname) ==="
nvidia-smi -L

run_one() {
    local d=$1 seed=$2
    local out_dir="$OUT_BASE/seed${seed}"
    mkdir -p "$out_dir"
    local out="$out_dir/scaling_d$(printf '%03d' $d).npz"
    local log="$out_dir/scaling_d$(printf '%03d' $d).log"
    if [ -f "$out" ] && python - "$out" <<'PY'
import sys
import numpy as np
try:
    z = np.load(sys.argv[1])
    first = int(z['first_cross'])
    stopped = int(z['stopped_early'])
except Exception:
    raise SystemExit(1)
if first > 0 and stopped == 1:
    raise SystemExit(0)
raise SystemExit(1)
PY
    then
        echo "SKIP lowhet d=$d seed=$seed (completed): $out"
        return 0
    fi
    echo "RUN lowhet d=$d seed=$seed"
    CUDA_VISIBLE_DEVICES=$GPU_ID python -u run_adaptive_hmc_lowhet_scaling_d.py \
        --d $d --K_at_100 50 --K_exp 0.5 --eta_at_100 0.006 --eta_exp -0.5 \
        --samples_per_client $SAMPLES_PER_CLIENT --variance_mode scalar \
        --stats_seed $seed --seed $seed \
        --threshold 0.1 --consecutive_below 20 --max_rounds $MAX_ROUNDS \
        --verbose_every 50 --checkpoint_every 100 \
        --out "$out" --gpu 0 \
        > "$log" 2>&1
}

echo "Using GPU_ID=$GPU_ID"
fail=0
for seed in $SEEDS; do
    for d in $DIMS; do
        run_one $d $seed || fail=$((fail+1))
    done
done

echo "Failed: $fail"
echo "=== Done $(date) ==="
