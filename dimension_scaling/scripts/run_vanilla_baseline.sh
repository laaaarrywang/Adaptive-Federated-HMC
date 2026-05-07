#!/bin/bash
#PBS -N fa-hmc-vanilla
#PBS -l select=1
#PBS -l filesystems=YOUR_FILESYSTEMS
#PBS -l walltime=03:00:00
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

ADAPT_DIR=${SCRATCH_ROOT}/dimension_scaling/adaptive_hmc_scaling
OUT_DIR=${SCRATCH_ROOT}/dimension_scaling/vanilla_baseline
mkdir -p "$OUT_DIR"
GPU_ID=${GPU_ID:-0}

echo "=== Start $(date) on $(hostname) ==="
nvidia-smi -L

run_one() {
    local d=$1
    local adapt="$ADAPT_DIR/scaling_d$(printf '%03d' $d).npz"
    if [ ! -f "$adapt" ]; then
        echo "Missing adaptive result for d=$d: $adapt"
        return 1
    fi
    local R
    R=$(python - "$adapt" <<'PY'
import sys
import numpy as np
z = np.load(sys.argv[1])
r = int(z['first_cross'])
if r <= 0:
    raise SystemExit(f'first_cross is not positive in {sys.argv[1]}: {r}')
print(r)
PY
) || return 1
    local out="$OUT_DIR/vanilla_d$(printf '%03d' $d).npz"
    local log="$OUT_DIR/vanilla_d$(printf '%03d' $d).log"
    if [ -f "$out" ] && python - "$out" "$R" <<'PY'
import sys
import numpy as np
try:
    z = np.load(sys.argv[1])
    final_round = int(z['final_round'])
    target = int(sys.argv[2])
    trace_w2 = z['trace_w2']
except Exception:
    raise SystemExit(1)
if final_round >= target and trace_w2.size >= target:
    raise SystemExit(0)
raise SystemExit(1)
PY
    then
        echo "SKIP vanilla d=$d (completed): $out"
        return 0
    fi
    CUDA_VISIBLE_DEVICES=$GPU_ID python -u run_vanilla_baseline_d.py \
        --d $d --T_vanilla 10 --R_target $R \
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
