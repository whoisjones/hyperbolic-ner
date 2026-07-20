#!/bin/bash
# P3 zero-shot tail probe: can name-seeded label embeddings + geometry type
# HELD-OUT types never seen in training? Split is fixed (--split-seed 7),
# identical across all conditions; geometry is the only knob.
#   geometry {euclidean, hyperbolic} x dim {16, 64} x seed {1,2,3} = 12 runs.
# GPUs 2 and 5 are used by another user -> excluded.
set -u
cd /vol/fob-vol7/mi18/goldejon/hyperbolic_ner
source /vol/tmp/goldejon/.uv/envs/mm/bin/activate
export PYTHONPATH=src HF_HUB_OFFLINE=1 NLTK_DATA=/vol/tmp/goldejon/nltk_data

mkdir -p results/p3/logs
GPUS=(0 1 3 4 6 7 8 9)
i=0
pids=()
run() {  # geom dim seed
  local geom=$1 dim=$2 seed=$3
  local gpu=${GPUS[$((i % ${#GPUS[@]}))]}
  local tag="${geom:0:3}-d${dim}-s${seed}"
  CUDA_VISIBLE_DEVICES=$gpu python train_zeroshot.py \
    --geometry $geom --dim $dim --seed $seed \
    --out results/p3/$tag.json \
    > results/p3/logs/$tag.log 2>&1 &
  pids+=($!)
  echo "launched $tag on gpu $gpu"
  i=$((i + 1))
}

for seed in 1 2 3; do
  for geom in euclidean hyperbolic; do
    for dim in 16 64; do
      run $geom $dim $seed
    done
  done
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
echo "p3 done (fail=$fail)"
exit $fail
