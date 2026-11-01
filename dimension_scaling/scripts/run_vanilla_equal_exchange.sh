#!/bin/bash
#PBS -N fa-hmc-equal-exchange
#PBS -l select=1
#PBS -l filesystems=YOUR_FILESYSTEMS
#PBS -l walltime=12:00:00
#PBS -A YOUR_ALLOCATION
#PBS -j oe
#PBS -o ./logs/

PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
SCRATCH_ROOT=${SCRATCH_ROOT:-${PROJECT_ROOT}/results}

# FA-HMC with FA-HMC-specific stepsize schedules at an equal number of server
# exchanges: AFHMC uses two exchanges per round, so FA-HMC is run for 2R(d)
# rounds, where R(d) is the AFHMC first-crossing round from
# run_adaptive_hmc_scaling.sh (run that sweep first).
#
#   legacy_k5_t10_eta_d14:   K=5, T=10, eta(d)=0.02 d^(-1/4)
#   theorem41_k13_eta_d712:  K(d)=max(1, round(K_AFHMC(d)/10)), T=10,
#                            eta(d)=0.006 (d/100)^(-7/12)

mkdir -p "${PROJECT_ROOT}/logs"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ulimit -c 0

# Load your Python environment here. Example:
# module use /path/to/modulefiles
# module load YOUR_PYTHON_MODULE
# conda activate YOUR_ENV

cd ${PROJECT_ROOT}/dimension_scaling/scripts

ADAPT_DIR=${SCRATCH_ROOT}/dimension_scaling/adaptive_hmc_scaling
OUT_BASE=${SCRATCH_ROOT}/dimension_scaling/vanilla_equal_exchange
GPU_ID=${GPU_ID:-0}

echo "=== Start $(date) on $(hostname) ==="
nvidia-smi -L

run_one() {
    local schedule=$1 d=$2 K_at_100=$3 K_exp=$4 eta_at_100=$5 eta_exp=$6
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
    local R2=$((2 * R))
    local out_dir="$OUT_BASE/$schedule"
    mkdir -p "$out_dir"
    local out="$out_dir/vanilla_d$(printf '%03d' $d).npz"
    local log="$out_dir/vanilla_d$(printf '%03d' $d).log"
    if [ -f "$out" ] && python - "$out" "$R2" <<'PY'
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
        echo "SKIP $schedule d=$d (completed): $out"
        return 0
    fi
    echo "RUN $schedule d=$d AFHMC_R=$R FAHMC_R=$R2"
    CUDA_VISIBLE_DEVICES=$GPU_ID python -u run_vanilla_baseline_d.py \
        --d $d --T_vanilla 10 \
        --K_at_100 $K_at_100 --K_exp $K_exp \
        --eta_at_100 $eta_at_100 --eta_exp $eta_exp \
        --R_target $R2 \
        --verbose_every 100 --checkpoint_every 100 \
        --out "$out" --gpu 0 \
        > "$log" 2>&1
}

echo "Using GPU_ID=$GPU_ID"
fail=0
for d in 2 50 100 150 200 250 300 350 400 450 500; do
    # K_at_100=50 with K_exp=0 gives K_AFHMC=50 and hence K=5 at every d;
    # eta_at_100 = 0.02 / 100^(1/4).
    run_one legacy_k5_t10_eta_d14 $d 50 0 0.006324555320336759 -0.25 || fail=$((fail+1))
    run_one theorem41_k13_eta_d712 $d 50 0.3333333333333333 0.006 -0.5833333333333334 || fail=$((fail+1))
done

echo "Failed: $fail"
echo "=== Done $(date) ==="
