#!/bin/bash
# P1 geometry probe grid: 2 geometries x 2 supervisions x 3 dims = 12 runs.
# Distributes runs round-robin over GPUs; one run per GPU at a time.
set -u
cd /vol/fob-vol7/mi18/goldejon/hyperbolic_ner
source /vol/tmp/goldejon/.uv/envs/mm/bin/activate
export PYTHONPATH=src HF_HUB_OFFLINE=1 NLTK_DATA=/vol/tmp/goldejon/nltk_data

mkdir -p results/p1/logs
GPUS=(0 1 2 3 4 5 6 7 8 9)
i=0
pids=()
for geom in euclidean hyperbolic; do
  for sup in flat ancestor; do
    for dim in 16 64 128; do
      gpu=${GPUS[$((i % ${#GPUS[@]}))]}
      tag="${geom:0:3}-${sup:0:3}-d${dim}"
      CUDA_VISIBLE_DEVICES=$gpu python train_probe.py \
        --geometry $geom --supervision $sup --dim $dim \
        --out results/p1/$tag.json \
        > results/p1/logs/$tag.log 2>&1 &
      pids+=($!)
      echo "launched $tag on gpu $gpu (pid $!)"
      i=$((i + 1))
    done
  done
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
echo "grid done (fail=$fail)"
exit $fail
