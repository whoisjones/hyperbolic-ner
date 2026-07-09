#!/bin/bash
# P2 controlled noise robustness: does hyperbolic degrade more gracefully?
# Cell = flat supervision @ dim 64 (strongest/cleanest P1 cell), UFET crowd.
#   geometry {euclidean, hyperbolic}
#   x noise  {clean(0), sibling@{10,30,50}%, uniform@{10,30,50}%}
#   x seed   {1,2,3}
# Noise corrupts TRAIN labels only (seeded). sibling=realistic, uniform=control.
set -u
cd /vol/fob-vol7/mi18/goldejon/sparse_ner
source /vol/tmp/goldejon/.uv/envs/mm/bin/activate
export PYTHONPATH=src HF_HUB_OFFLINE=1 NLTK_DATA=/vol/tmp/goldejon/nltk_data

mkdir -p results/p2/logs
GPUS=(0 1 2 3 4 5 6 7 8 9)
i=0
pids=()
run() {  # geom rate arg_mode tag_mode seed
  local geom=$1 rate=$2 arg_mode=$3 tag_mode=$4 seed=$5
  local gpu=${GPUS[$((i % ${#GPUS[@]}))]}
  local rtag=$(printf '%02d' "$(python -c "print(int(round($rate*100)))")")
  local tag="${geom:0:3}-${tag_mode:0:3}-r${rtag}-s${seed}"
  CUDA_VISIBLE_DEVICES=$gpu python train_probe.py \
    --geometry $geom --supervision flat --dim 64 --seed $seed \
    --noise-rate $rate --noise-mode $arg_mode \
    --out results/p2/$tag.json \
    > results/p2/logs/$tag.log 2>&1 &
  pids+=($!)
  echo "launched $tag on gpu $gpu"
  i=$((i + 1))
}

for seed in 1 2 3; do
  for geom in euclidean hyperbolic; do
    run $geom 0.0 sibling clean $seed          # shared baseline (rate 0)
    for rate in 0.1 0.3 0.5; do
      run $geom $rate sibling sibling $seed
      run $geom $rate uniform uniform $seed
    done
  done
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
echo "p2 done (fail=$fail)"
exit $fail
