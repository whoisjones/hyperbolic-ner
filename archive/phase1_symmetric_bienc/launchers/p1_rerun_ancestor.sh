#!/bin/bash
# Rerun ancestor-supervision cells after the taxonomy blocklist fix
# (blocked vacuous wrong-sense WordNet roots: object, event, unit, document,
# act, activity, ... — see scripts/build_taxonomy_wordnet.py).
# 2 geometries x 3 dims x 4 seeds (42,1,2,3) = 24 runs, matching the flat-row
# seed coverage so the comparison is apples-to-apples.
#
# Runs in strict batches of len(GPUS) so exactly one job occupies each GPU at
# a time (unrestricted concurrency previously caused catastrophic CPU thread
# oversubscription — 24 processes each defaulting to all 256 cores).
set -u
cd /vol/fob-vol7/mi18/goldejon/hyperbolic_ner
source /vol/tmp/goldejon/.uv/envs/mm/bin/activate
export PYTHONPATH=src HF_HUB_OFFLINE=1 NLTK_DATA=/vol/tmp/goldejon/nltk_data
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false

mkdir -p results/p1_anc_v2/logs
GPUS=(0 1 2 3 4 5 6 7 8 9)

jobs_geom=(); jobs_dim=(); jobs_seed=()
for seed in 42 1 2 3; do
  for geom in euclidean hyperbolic; do
    for dim in 16 64 128; do
      jobs_geom+=("$geom"); jobs_dim+=("$dim"); jobs_seed+=("$seed")
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
    geom=${jobs_geom[$j]}; dim=${jobs_dim[$j]}; seed=${jobs_seed[$j]}
    gpu=${GPUS[$gi]}
    tag="${geom:0:3}-anc-d${dim}-s${seed}"
    CUDA_VISIBLE_DEVICES=$gpu python train_probe.py \
      --geometry $geom --supervision ancestor --dim $dim --seed $seed \
      --out results/p1_anc_v2/$tag.json \
      > results/p1_anc_v2/logs/$tag.log 2>&1 &
    pids+=($!)
    echo "launched $tag on gpu $gpu"
    gi=$((gi + 1))
  done
  for p in "${pids[@]}"; do wait "$p" || fail=1; done
  echo "batch [$i:$batch_end) done"
  i=$batch_end
done
echo "ancestor rerun done (fail=$fail)"
exit $fail
