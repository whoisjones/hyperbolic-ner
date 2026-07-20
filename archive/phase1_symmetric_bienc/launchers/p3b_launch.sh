#!/bin/bash
# P3b zero-shot scale-up: same protocol as P3 (UFET) but trained on FiNERweb-eng
# (distant, single-tag, richer/noisier GPT type vocab; 2034 types, 73% tail).
# Fixed doc-level split (scripts/p3b_prep_finerweb.py) + fixed held-out set.
#   geometry {euclidean, hyperbolic} x seed {1,2,3} @ dim 64 = 6 runs.
# dim fixed to 64 (P3-UFET showed the geometry effect is dim-invariant).
# FiNERweb is ~20x more spans than UFET-crowd -> fewer epochs, bigger batch.
# GPUs 2 and 5 excluded (other user).
set -u
cd /vol/fob-vol7/mi18/goldejon/hyperbolic_ner
source /vol/tmp/goldejon/.uv/envs/mm/bin/activate
export PYTHONPATH=src HF_HUB_OFFLINE=1 NLTK_DATA=/vol/tmp/goldejon/nltk_data
D=/vol/tmp/goldejon/hyperbolic_ner/data

mkdir -p results/p3b/logs
GPUS=(0 1 3 4 6 7)
i=0
pids=()
run() {  # geom seed
  local geom=$1 seed=$2
  local gpu=${GPUS[$((i % ${#GPUS[@]}))]}
  local tag="${geom:0:3}-d64-s${seed}"
  CUDA_VISIBLE_DEVICES=$gpu python train_zeroshot.py \
    --geometry $geom --dim 64 --seed $seed \
    --train-file $D/finerweb_eng_train.jsonl --test-file $D/finerweb_eng_test.jsonl \
    --max-len 256 --epochs 12 --batch-size 64 \
    --out results/p3b/$tag.json \
    > results/p3b/logs/$tag.log 2>&1 &
  pids+=($!)
  echo "launched $tag on gpu $gpu"
  i=$((i + 1))
}

for seed in 1 2 3; do
  for geom in euclidean hyperbolic; do
    run $geom $seed
  done
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
echo "p3b done (fail=$fail)"
exit $fail
