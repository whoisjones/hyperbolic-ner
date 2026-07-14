#!/bin/bash
# P3d supervised-F1 scaling sweep on FiNERweb-eng (supervised companion to P3c).
# Does the SUPERVISED typing-F1 hyperbolic advantage also decay with training
# data, like the zero-shot mAP gap in P3c? Same corpus, fixed 25-epoch schedule.
#   geometry {euclidean, hyperbolic} x scale {100,300,1000} docs x seed {1,2,3}.
# Throttled: the cluster is busy, so each free GPU runs its share SEQUENTIALLY
# (one run per GPU at a time) to avoid OOM from oversubscription. Idempotent:
# skips any cell whose JSON already exists, so it is safe to re-run.
set -u
cd /vol/fob-vol7/mi18/goldejon/sparse_ner
source /vol/tmp/goldejon/.uv/envs/mm/bin/activate
export PYTHONPATH=src HF_HUB_OFFLINE=1 NLTK_DATA=/vol/tmp/goldejon/nltk_data
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
D=/vol/tmp/goldejon/sparse_ner/data

mkdir -p results/p3d/logs
GPUS=(5 6 7)                 # edit to match currently-free GPUs

# build the job list
jobs=()
for seed in 1 2 3; do
  for docs in 100 300 1000; do
    for geom in euclidean hyperbolic; do
      jobs+=("$geom $docs $seed")
    done
  done
done

run_one() {  # gpu geom docs seed
  local gpu=$1 geom=$2 docs=$3 seed=$4
  local tag="${geom:0:3}-n${docs}-s${seed}"
  [ -f "results/p3d/$tag.json" ] && { echo "skip $tag (exists)"; return 0; }
  echo "start $tag on gpu $gpu"
  CUDA_VISIBLE_DEVICES=$gpu python train_probe.py \
    --geometry $geom --supervision flat --dim 64 --seed $seed \
    --train-file $D/finerweb_eng_suptrain.jsonl \
    --dev-file $D/finerweb_eng_val.jsonl \
    --test-file $D/finerweb_eng_test.jsonl \
    --max-len 256 --batch-size 64 --epochs 25 --train-max-records $docs \
    --out results/p3d/$tag.json > results/p3d/logs/$tag.log 2>&1
  echo "done  $tag (exit $?)"
}

# one worker per GPU, each consumes jobs at index i where i % nGPU == slot
nG=${#GPUS[@]}
pids=()
for slot in $(seq 0 $((nG - 1))); do
  (
    gpu=${GPUS[$slot]}
    i=$slot
    while [ $i -lt ${#jobs[@]} ]; do
      set -- ${jobs[$i]}
      run_one $gpu $1 $2 $3
      i=$((i + nG))
    done
  ) &
  pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
echo "p3d done (fail=$fail)"
exit $fail
