#!/bin/bash
# P3c retry: re-run only the cells whose JSON is missing (OOM'd in the first
# launch when other users' jobs were on some GPUs). Idempotent: skips any cell
# that already has a result. One run per GPU on a curated free-GPU list.
set -u
cd /vol/fob-vol7/mi18/goldejon/hyperbolic_ner
source /vol/tmp/goldejon/.uv/envs/mm/bin/activate
export PYTHONPATH=src HF_HUB_OFFLINE=1 NLTK_DATA=/vol/tmp/goldejon/nltk_data
D=/vol/tmp/goldejon/hyperbolic_ner/data

# clear any stragglers from earlier partial launches so cells don't double-run
for pid in $(pgrep -f train_zeroshot.py); do kill "$pid" 2>/dev/null; done
sleep 3

GPUS=(2 3 4 6 7 8 9)
i=0
pids=()
for seed in 1 2 3; do
  for docs in 100 300 1000; do
    for geom in euclidean hyperbolic; do
      tag="${geom:0:3}-n${docs}-s${seed}"
      [ -f "results/p3c/$tag.json" ] && continue
      gpu=${GPUS[$((i % ${#GPUS[@]}))]}
      CUDA_VISIBLE_DEVICES=$gpu python train_zeroshot.py \
        --geometry $geom --dim 64 --seed $seed \
        --train-file $D/finerweb_eng_train.jsonl --test-file $D/finerweb_eng_test.jsonl \
        --max-len 256 --epochs 25 --batch-size 64 --train-max-records $docs \
        --out results/p3c/$tag.json > results/p3c/logs/$tag.log 2>&1 &
      pids+=($!)
      echo "relaunched $tag on gpu $gpu"
      i=$((i + 1))
    done
  done
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
echo "p3c retry done (fail=$fail)"
exit $fail
