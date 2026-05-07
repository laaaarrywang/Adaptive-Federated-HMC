#!/bin/bash
#PBS -N fa-hmc-scaling
#PBS -l select=1
#PBS -l filesystems=YOUR_FILESYSTEMS
#PBS -l walltime=01:00:00
#PBS -A YOUR_ALLOCATION
#PBS -j oe
#PBS -o ./logs/

PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
SCRATCH_ROOT=${SCRATCH_ROOT:-${PROJECT_ROOT}/results}

mkdir -p "${PROJECT_ROOT}/logs"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ulimit -c 0

# Load your Python environment here. Example:
# module use /path/to/modulefiles
# module load YOUR_PYTHON_MODULE
# conda activate YOUR_ENV

cd ${PROJECT_ROOT}/dimension_scaling/scripts

OUT_DIR=${SCRATCH_ROOT}/dimension_scaling/adaptive_hmc_scaling
mkdir -p "$OUT_DIR"
GPU_ID=${GPU_ID:-0}

echo "=== Start $(date) on $(hostname) ==="
nvidia-smi -L

run_one() {
    local d=$1
    local out="$OUT_DIR/scaling_d$(printf '%03d' $d).npz"
    local log="$OUT_DIR/scaling_d$(printf '%03d' $d).log"
    if [ -f "$out" ] && python - "$out" <<'PY'
import sys
import numpy as np
try:
    z = np.load(sys.argv[1])
    first = int(z['first_cross'])
    final_round = int(z['final_round'])
    stopped = int(z['stopped_early'])
except Exception:
    raise SystemExit(1)
if first > 0 and final_round > 0 and stopped == 1:
    raise SystemExit(0)
raise SystemExit(1)
PY
    then
        echo "SKIP adaptive d=$d (completed): $out"
        return 0
    fi
    CUDA_VISIBLE_DEVICES=$GPU_ID python -u run_adaptive_hmc_scaling_d.py \
        --d $d --threshold 0.1 --consecutive_below 20 \
        --verbose_every 50 --checkpoint_every 100 \
        --out "$out" --gpu 0 \
        > "$log" 2>&1
}

echo "Using GPU_ID=$GPU_ID"
fail=0
for d in 2 50 100 150 200 250 300 350 400 450 500; do
    run_one $d || fail=$((fail+1))
done

echo "Failed: $fail"
echo "=== Done $(date) ==="
