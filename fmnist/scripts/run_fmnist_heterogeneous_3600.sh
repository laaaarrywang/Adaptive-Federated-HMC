#!/bin/bash
#PBS -N fmnist-het-3600
#PBS -l select=1
#PBS -l filesystems=YOUR_FILESYSTEMS
#PBS -l walltime=48:00:00
#PBS -A YOUR_ALLOCATION
#PBS -j oe
#PBS -o ./logs/

PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
SCRATCH_ROOT=${SCRATCH_ROOT:-${PROJECT_ROOT}/results}

# Long-horizon heterogeneous Fashion-MNIST runs used for the equal
# server-exchange comparison: 3600 communication rounds per cell, so that
# FA-HMC round 3600 can be compared with AFHMC round 1800 (AFHMC uses two
# server exchanges per round). Data split: Dirichlet(alpha=0.1).
#
# Only the step sizes selected by the 400-round tuning sweep
# (run_fmnist_heterogeneous_400_tuning.sh) are run; 50 cells in total.
# The full sweep takes roughly 135 GPU-hours. To split it across GPUs or jobs,
# launch NUM_SHARDS copies with SHARD_ID=0..NUM_SHARDS-1, e.g. on one 4-GPU node:
#   for s in 0 1 2 3; do NUM_SHARDS=4 SHARD_ID=$s GPU_ID=$s bash run_fmnist_heterogeneous_3600.sh & done; wait

mkdir -p "${PROJECT_ROOT}/logs"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ulimit -c 0

# Load your Python environment here. Example:
# conda activate YOUR_ENV

cd "${PROJECT_ROOT}/fmnist"

OUT_BASE="${SCRATCH_ROOT}/fmnist/heterogeneous_3600"
mkdir -p "$OUT_BASE"

NUM_CLIENT=10
TEMPERATURE=0.005
BATCH_TRAIN=1000
COMM_ROUNDS=3600
SAVE_SIZE=100
DATA=FashionMNIST
SPLIT=dirichlet
ALPHA=0.1
SEEDS="0 1 2 3 4"
GPU_ID=${GPU_ID:-0}
NUM_SHARDS=${NUM_SHARDS:-1}
SHARD_ID=${SHARD_ID:-0}

# algo|K|step sizes
declare -a ARMS=(
    "fa_hmc|1|1e-4 2e-4"
    "fa_hmc|10|2e-5"
    "fa_hmc|50|5e-6 5e-5"
    "fa_hmc|100|2e-6 5e-6"
    "adaptive|10|2e-5"
    "adaptive|50|5e-6"
    "adaptive|100|2e-6"
)

echo "=== Start $(date) on $(hostname) ==="
echo "COMM_ROUNDS=$COMM_ROUNDS"
echo "Split: $SPLIT, Dirichlet alpha = $ALPHA"
echo "Shard $SHARD_ID of $NUM_SHARDS on GPU_ID=$GPU_ID"
nvidia-smi -L

total_step_for() {
    local algo=$1 K=$2
    if [ "$algo" = "adaptive" ]; then
        echo $COMM_ROUNDS
    else
        echo $((COMM_ROUNDS * 50 / K))
    fi
}

end_id_for() {
    local algo=$1 K=$2
    local total_step=$(total_step_for $algo $K)
    local T
    if [ "$algo" = "adaptive" ]; then T=$K; else T=50; fi
    local a=$T b=$K
    while [ $b -ne 0 ]; do tmp=$b; b=$((a % b)); a=$tmp; done
    local gcd=$a
    local lcm=$((T * K / gcd))
    local cal_number=$((total_step * K / lcm))
    local end_id=$(( (cal_number + SAVE_SIZE - 1) / SAVE_SIZE ))
    [ $end_id -lt 1 ] && end_id=1
    echo $end_id
}

run_cell() {
    local algo=$1 K=$2 eta=$3 seed=$4
    local cell_dir="$OUT_BASE/$algo/K${K}/eta${eta}/seed${seed}"
    mkdir -p "$cell_dir"
    if [ -f "$cell_dir/post.npz" ]; then
        echo "SKIP $algo K=$K eta=$eta seed=$seed"
        return 0
    fi

    local total_step=$(total_step_for $algo $K)
    local end_id=$(end_id_for $algo $K)
    local local_step
    local adaptive_flag
    if [ "$algo" = "adaptive" ]; then
        local_step=$K
        adaptive_flag=1
    else
        local_step=50
        adaptive_flag=0
    fi

    echo "RUN $algo K=$K eta=$eta seed=$seed total_step=$total_step end_id=$end_id"

    CUDA_VISIBLE_DEVICES=$GPU_ID python -u logistic_fashion.py \
        -gpu 0 -federated 1 -num_client $NUM_CLIENT -fast 1 \
        -split $SPLIT -dirichlet_alpha $ALPHA \
        -total_step $total_step -batch_train $BATCH_TRAIN -batch_test 2000 \
        -T $TEMPERATURE -lr $eta \
        -leapfrog_step $K -local_step $local_step -adaptive $adaptive_flag \
        -save_size $SAVE_SIZE -save_gap 1 -bin_size 20 \
        -save_name "$cell_dir/" -save_test_name "$cell_dir/test" \
        -seed $seed -data $DATA \
        > "$cell_dir/train.log" 2>&1 || { echo "TRAIN FAIL $algo K=$K eta=$eta seed=$seed"; return 1; }

    CUDA_VISIBLE_DEVICES=$GPU_ID python -u cal_poster_metrics_initial.py \
        -gpu 0 \
        -leapfrog_step $K -local_step $local_step \
        -save_size $SAVE_SIZE -save_gap 1 -cal_gap 1 -bin_size 20 \
        -start_id 1 -end_id $end_id \
        -save_name "$cell_dir/" -save_test_name "$cell_dir/post" \
        -seed $seed -data $DATA -batch_train $BATCH_TRAIN -batch_test 2000 \
        > "$cell_dir/post.log" 2>&1 || { echo "POST FAIL $algo K=$K eta=$eta seed=$seed"; return 1; }

    rm -f "$cell_dir"/[0-9]*.pt
}

declare -a CELLS=()
for arm in "${ARMS[@]}"; do
    IFS='|' read -r algo K etas <<<"$arm"
    for eta in $etas; do
        for seed in $SEEDS; do
            CELLS+=("$algo|$K|$eta|$seed")
        done
    done
done

echo "Total cells: ${#CELLS[@]}"

fail=0
for i in "${!CELLS[@]}"; do
    [ $((i % NUM_SHARDS)) -eq $SHARD_ID ] || continue
    IFS='|' read -r algo K eta seed <<<"${CELLS[$i]}"
    run_cell $algo $K $eta $seed || fail=$((fail+1))
done

echo "Failed: $fail"
echo "=== Done $(date) ==="
