#!/bin/bash
# P3c data-scaling ablation: isolate whether SCALE (not corpus type) removed the
# hyperbolic zero-shot advantage seen on UFET (P3) but gone on FiNERweb (P3b).
# Subsample FiNERweb-eng TRAIN to {100, 300, 1000} docs (~2k, 6k, 20k spans;
# 100 docs ~= UFET-crowd size). Vocab + held-out split stay fixed (computed from
# the full file), so only training data amount varies.
# Schedule held CONSTANT at epochs=25 (matches UFET passes) so the comparison is
# controlled: same number of passes, different N.
#   geometry {euclidean, hyperbolic} x scale {100,300,1000} x seed {1,2,3} = 18.
# GPUs 2 and 5 excluded (other user).
set -u
cd /vol/fob-vol7/mi18/goldejon/hyperbolic_ner
source /vol/tmp/goldejon/.uv/envs/mm/bin/activate
export PYTHONPATH=src HF_HUB_OFFLINE=1 NLTK_DATA=/vol/tmp/goldejon/nltk_data
D=/vol/tmp/goldejon/hyperbolic_ner/data

mkdir -p results/p3c/logs
GPUS=(0 1 3 4 6 7 8 9)
i=0
pids=()
run() {  # geom docs seed
  local geom=$1 docs=$2 seed=$3
  local gpu=${GPUS[$((i % ${#GPUS[@]}))]}
  local tag="${geom:0:3}-n${docs}-s${seed}"
  CUDA_VISIBLE_DEVICES=$gpu python train_zeroshot.py \
    --geometry $geom --dim 64 --seed $seed \
    --train-file $D/finerweb_eng_train.jsonl --test-file $D/finerweb_eng_test.jsonl \
    --max-len 256 --epochs 25 --batch-size 64 --train-max-records $docs \
    --out results/p3c/$tag.json \
    > results/p3c/logs/$tag.log 2>&1 &
  pids+=($!)
  echo "launched $tag on gpu $gpu"
  i=$((i + 1))
}

for seed in 1 2 3; do
  for docs in 100 300 1000; do
    for geom in euclidean hyperbolic; do
      run $geom $docs $seed
    done
  done
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
echo "p3c done (fail=$fail)"
exit $fail
