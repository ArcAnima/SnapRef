#!/usr/bin/env bash
# Danbooru pilot on one machine: prepare (if needed) -> train 3,000 steps (checkpoint every 500) -> eval every
# 1,000 steps on a spare GPU while training continues -> HF backup of checkpoints, logs and eval.
#   RUN=pilot_ft    CONFIG=configs/train_pilot_ft.json    INIT=~/adapters/v1a/snapref_v1a.safetensors bash deploy/run_pilot.sh
#   RUN=pilot_scratch CONFIG=configs/train_pilot_scratch.json bash deploy/run_pilot.sh
# Env: GPUS (train GPUs, default all), EVAL_GPU (default last GPU),
#      DATA (~/pilot_data), PREP (~/train/pilot_prep), PY (~/comfy-venv/bin/python).
set -u
cd ~/anima-ref-adapter-research
unset LD_LIBRARY_PATH
PY=${PY:-$HOME/comfy-venv/bin/python}
RUN=${RUN:?}; CONFIG=${CONFIG:?}
DATA=${DATA:-$HOME/pilot_data}; PREP=${PREP:-$HOME/train/pilot_prep}
OUT=$HOME/train/runs/$RUN; CMP=$HOME/train/compare; LOG=$HOME/logs/train
mkdir -p $OUT $CMP $LOG
N=$(nvidia-smi -L | wc -l)
EVAL_GPU=${EVAL_GPU:-$((N - 1))}
GPUS=${GPUS:-$(seq -s, 0 $((N - 1)))}
NG=$(echo $GPUS | tr , '\n' | wc -l)
log() { echo "$(date -u +%T) $*" | tee -a $LOG/pipeline.log; }
set -a; . ~/.config/anima-ref/secrets.env; set +a

if [ ! -s $PREP/index.json ]; then
  log "$RUN: prepare $DATA -> $PREP"
  $PY train/prepare_danbooru.py --data $DATA --out $PREP --config $CONFIG > $LOG/prepare_$RUN.log 2>&1 || { log "$RUN: prepare FAILED"; exit 1; }
  log "$RUN: prepare done $(tail -n 1 $LOG/prepare_$RUN.log)"
fi

evaluate() {  # step
  local s=$(printf %06d $1)
  [ -s $CMP/${RUN}_$s/eval.json ] && return
  CUDA_VISIBLE_DEVICES=$EVAL_GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY train/eval_pilot.py \
      --adapter $OUT/adapter_$s.safetensors --config $OUT/config.json --out $CMP/${RUN}_$s > $LOG/eval_${RUN}_$s.log 2>&1 \
      && log "$RUN: eval $1 $(tail -n 1 $LOG/eval_${RUN}_$s.log)" || log "$RUN: eval $1 FAILED"
}
backup() {
  $PY - "$OUT" "$CMP" "$RUN" "$LOG" <<'EOF' >> $LOG/hf_backup_$RUN.log 2>&1
import sys
from huggingface_hub import HfApi
out, cmp_, run, log = sys.argv[1:5]
api = HfApi(); repo = "0ppxnhximxr/anima-ref-kv"
api.upload_folder(repo_id=repo, folder_path=out, path_in_repo=f"runs/{run}", allow_patterns=["adapter_*.safetensors", "config.json", "train_log.jsonl"], commit_message=f"backup {run}")
api.upload_folder(repo_id=repo, folder_path=cmp_, path_in_repo="compare", allow_patterns=[f"{run}_*/eval.json", f"{run}_*/*.jpg"], commit_message=f"eval {run}")
api.upload_folder(repo_id=repo, folder_path=log, path_in_repo=f"logs/{run}", allow_patterns=[f"*{run}*", "pipeline.log"], commit_message=f"logs {run}")
print("backup ok")
EOF
}

TRAIN_GPUS=$GPUS   # all GPUs train (batch 4 x GPUs x accum = 16); with > 2 GPUs (96 GB each) eval shares the last one
NT=$(echo $TRAIN_GPUS | tr , '\n' | wc -l)
INIT_ARG=""; [ -n "${INIT:-}" ] && INIT_ARG="--init $INIT"
log "$RUN: train on GPUs $TRAIN_GPUS, eval on GPU $EVAL_GPU"
# segments of 1,000 steps: after each, eval runs in the background while the next segment trains
for stop in 1000 2000 3000; do
  s=$(printf %06d $stop)
  if [ ! -s $OUT/adapter_$s.safetensors ]; then
    RES=""; last=$(ls $OUT/state_*.pt 2>/dev/null | tail -n 1); [ -n "$last" ] && RES="--resume $last"
    CUDA_VISIBLE_DEVICES=$TRAIN_GPUS $PY -m torch.distributed.run --nproc_per_node $NT --master_port 29521 train/train_asym.py \
        --prep $PREP --config $CONFIG --out $OUT --stop-at $stop --preview-fixed "" $([ -z "$RES" ] && echo $INIT_ARG) $RES \
        >> $LOG/$RUN.log 2>&1 || { log "$RUN: training FAILED before $stop"; tail -n 20 $LOG/$RUN.log >> $LOG/pipeline.log; exit 1; }
    log "$RUN: reached $stop"
  fi
  if [ "$N" -gt 2 ] || [ $stop -eq 3000 ]; then
    (evaluate $stop; backup) &
  else
    evaluate $stop; backup   # 2 GPUs: the eval GPU is a training GPU, so eval between segments
  fi
done
wait
log "$RUN: done"
