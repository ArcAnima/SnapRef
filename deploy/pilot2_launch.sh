#!/usr/bin/env bash
# H100: wait for pilot_prep_v2, smoke-test the per-pair prompt path, then pilot2_ft in 1,000-step segments.
# Each segment uses every GPU that is free at its start (GPU 1 is busy while the text-encoder benchmark runs):
# 1 GPU -> batch 4 x accum 4, 2 GPUs -> batch 4 x accum 2 (16 either way); eval between segments.
set -u
cd ~/anima-ref-adapter-research; unset LD_LIBRARY_PATH
PY=~/comfy-venv/bin/python; L=~/logs/train; RUN=pilot2_ft; OUT=~/train/runs/$RUN; CMP=~/train/compare
PREP=~/train/pilot_prep_v2; CFG=configs/train_pilot2_ft.json; INIT=~/adapters/v1a/snapref_v1a.safetensors
mkdir -p $OUT; log() { echo "$(date -u +%T) $*" | tee -a $L/pipeline.log; }
set -a; . ~/.config/anima-ref/secrets.env; set +a
until [ -s $PREP/index.json ] && ! pgrep -f "train/[a]dd_pair_prompts.py" >/dev/null; do sleep 60; done
log "$RUN: prep v2 ready: $(tail -n 1 $L/prep_v2.log)"
if ! [ -s $OUT/adapter_001000.safetensors ]; then
  CUDA_VISIBLE_DEVICES=0 $PY train/train_asym.py --prep $PREP --config $CFG --out /tmp/smoke_$RUN --init $INIT --smoke \
      --preview-fixed "" > $L/smoke_$RUN.log 2>&1 || { log "$RUN: smoke FAILED"; tail -n 20 $L/smoke_$RUN.log >> $L/pipeline.log; exit 1; }
  log "$RUN: smoke ok $(grep 'step\"' $L/smoke_$RUN.log | tail -n 1 | cut -c1-120)"
fi
free_gpus() { nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$2 < 2000 {print $1}' | paste -sd,; }
for stop in 1000 2000 3000; do
  s=$(printf %06d $stop)
  if [ ! -s $OUT/adapter_$s.safetensors ]; then
    G=$(free_gpus); [ -z "$G" ] && G=0; NG=$(echo $G | tr , '\n' | wc -l)
    python3 - "$CFG" "$NG" <<'PYEOF'
import json, sys
c = json.load(open(sys.argv[1])); c["train"]["grad_accum"] = 16 // (4 * int(sys.argv[2]))
json.dump(c, open("/tmp/pilot2_seg.json", "w"), indent=1)
PYEOF
    RES=""; last=$(ls $OUT/state_*.pt 2>/dev/null | tail -n 1); [ -n "$last" ] && RES="--resume $last"
    log "$RUN: segment to $stop on GPUs $G"
    CUDA_VISIBLE_DEVICES=$G $PY -m torch.distributed.run --nproc_per_node $NG --master_port 29531 train/train_asym.py \
        --prep $PREP --config /tmp/pilot2_seg.json --out $OUT --stop-at $stop --preview-fixed "" $([ -z "$RES" ] && echo --init $INIT) $RES \
        >> $L/$RUN.log 2>&1 || { log "$RUN: training FAILED before $stop"; tail -n 20 $L/$RUN.log >> $L/pipeline.log; exit 1; }
    log "$RUN: reached $stop"
  fi
  [ -s $CMP/${RUN}_$s/eval.json ] || { CUDA_VISIBLE_DEVICES=$(free_gpus | cut -d, -f1) $PY train/eval_pilot.py --adapter $OUT/adapter_$s.safetensors \
      --config $OUT/config.json --out $CMP/${RUN}_$s > $L/eval_${RUN}_$s.log 2>&1 && log "$RUN: eval $stop $(tail -n 1 $L/eval_${RUN}_$s.log)"; }
  $PY - "$OUT" "$CMP" "$RUN" "$L" <<'PYEOF' >> $L/hf_backup_$RUN.log 2>&1
import sys
from huggingface_hub import HfApi
out, cmp_, run, log = sys.argv[1:5]; api = HfApi(); r = "0ppxnhximxr/anima-ref-kv"
api.upload_folder(repo_id=r, folder_path=out, path_in_repo=f"runs/{run}", allow_patterns=["adapter_*.safetensors", "config.json", "train_log.jsonl"])
api.upload_folder(repo_id=r, folder_path=cmp_, path_in_repo="compare", allow_patterns=[f"{run}_*/eval.json", f"{run}_*/*.jpg", f"{run}_*/style/*.png"])
PYEOF
done
log "$RUN: done"
