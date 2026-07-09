#!/bin/bash
# P1 seed replication: seeds {1,2,3} on the key cells
#   - euclidean/hyperbolic x flat/ancestor @ d64   (main comparison)
#   - euclidean/hyperbolic x flat @ d16            (low-dim claim)
# seed 42 already exists from the initial grid.
set -u
cd /vol/fob-vol7/mi18/goldejon/sparse_ner
source /vol/tmp/goldejon/.uv/envs/mm/bin/activate
export PYTHONPATH=src HF_HUB_OFFLINE=1 NLTK_DATA=/vol/tmp/goldejon/nltk_data

mkdir -p results/p1_seeds/logs
GPUS=(0 1 2 3 4 5 6 7 8 9)
i=0
pids=()
run() {
  local geom=$1 sup=$2 dim=$3 seed=$4
  local gpu=${GPUS[$((i % ${#GPUS[@]}))]}
  local tag="${geom:0:3}-${sup:0:3}-d${dim}-s${seed}"
  CUDA_VISIBLE_DEVICES=$gpu python train_probe.py \
    --geometry $geom --supervision $sup --dim $dim --seed $seed \
    --out results/p1_seeds/$tag.json \
    > results/p1_seeds/logs/$tag.log 2>&1 &
  pids+=($!)
  echo "launched $tag on gpu $gpu"
  i=$((i + 1))
}

for seed in 1 2 3; do
  for geom in euclidean hyperbolic; do
    run $geom flat 64 $seed
    run $geom ancestor 64 $seed
    run $geom flat 16 $seed
  done
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
echo "seed grid done (fail=$fail)"
exit $fail
