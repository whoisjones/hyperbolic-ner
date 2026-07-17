#!/bin/bash
# C2 gradient-conflict measurement: 2 geometries x 3 seeds, one job per GPU.
set -u
cd /vol/fob-vol7/mi18/goldejon/sparse_ner
source /vol/tmp/goldejon/.uv/envs/mm/bin/activate
export PYTHONPATH=src HF_HUB_OFFLINE=1 NLTK_DATA=/vol/tmp/goldejon/nltk_data
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false

mkdir -p results/c2/logs
GPUS=(0 1 2 3 4 5)
i=0
pids=()
for geom in euclidean hyperbolic; do
  for seed in 1 2 3; do
    gpu=${GPUS[$i]}
    tag="${geom:0:3}-d64-s${seed}"
    CUDA_VISIBLE_DEVICES=$gpu python scripts/c2_gradient_conflict.py \
      --geometry $geom --seed $seed --out results/c2/$tag.json \
      > results/c2/logs/$tag.log 2>&1 &
    pids+=($!)
    echo "launched $tag on gpu $gpu"
    i=$((i + 1))
  done
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
echo "c2 done (fail=$fail)"
exit $fail
