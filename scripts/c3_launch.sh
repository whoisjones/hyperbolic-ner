#!/bin/bash
# C3 causal loss-family grid: 2 geometries x {infonce, infonce-masked,
# soft-bce} x seeds {1,2,3} = 18 runs on UFET crowd, d64, flat supervision.
# The BCE cells reuse results/p1_seeds/{euc,hyp}-fla-d64-s{1,2,3}.json.
# Strict batches of len(GPUS): one job per GPU, capped CPU threads.
set -u
cd /vol/fob-vol7/mi18/goldejon/hyperbolic_ner
source /vol/tmp/goldejon/.uv/envs/mm/bin/activate
export PYTHONPATH=src HF_HUB_OFFLINE=1 NLTK_DATA=/vol/tmp/goldejon/nltk_data
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false

mkdir -p results/c3/logs
GPUS=(0 1 2 3 4 5 6 7 8 9)

jobs_geom=(); jobs_loss=(); jobs_seed=()
for seed in 1 2 3; do
  for geom in euclidean hyperbolic; do
    for loss in infonce infonce-masked soft-bce; do
      jobs_geom+=("$geom"); jobs_loss+=("$loss"); jobs_seed+=("$seed")
    done
  done
done

n=${#jobs_geom[@]}
fail=0
i=0
while [ $i -lt $n ]; do
  pids=()
  batch_end=$((i + ${#GPUS[@]}))
  [ $batch_end -gt $n ] && batch_end=$n
  gi=0
  for ((j=i; j<batch_end; j++)); do
    geom=${jobs_geom[$j]}; loss=${jobs_loss[$j]}; seed=${jobs_seed[$j]}
    gpu=${GPUS[$gi]}
    tag="${geom:0:3}-${loss}-s${seed}"
    CUDA_VISIBLE_DEVICES=$gpu python train_probe.py \
      --geometry $geom --supervision flat --dim 64 --loss $loss --seed $seed \
      --out results/c3/$tag.json \
      > results/c3/logs/$tag.log 2>&1 &
    pids+=($!)
    echo "launched $tag on gpu $gpu"
    gi=$((gi + 1))
  done
  for p in "${pids[@]}"; do wait "$p" || fail=1; done
  echo "batch [$i:$batch_end) done"
  i=$batch_end
done
echo "c3 done (fail=$fail)"
exit $fail
