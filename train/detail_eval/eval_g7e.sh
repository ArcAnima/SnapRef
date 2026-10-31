#!/bin/bash
# Detail eval of one v2a checkpoint on the g7e.24xlarge (4 GPUs): render the frozen set in 4 shards, features,
# score against every model already rendered here, upload results to HF runs/v2a/detail_eval/v2a_XXXXXX/.
#   bash train/detail_eval/eval_g7e.sh 15000
set -euo pipefail
unset LD_LIBRARY_PATH
STEP=${1:?step}; NAME=$(printf "v2a_%06d" "$STEP")
ROOT=${ROOT:-$HOME/detail_eval}; CODE=$HOME/anima-ref-adapter-research
cd "$CODE"
ADAPTER=$HOME/train/runs/v2a/adapter_$(printf "%06d" "$STEP").safetensors
[ -f "$ADAPTER" ] || { echo "no $ADAPTER" >&2; exit 2; }
OUT=$ROOT/renders/$NAME; mkdir -p "$OUT" "$ROOT/logs"
T0=$(date +%s)
for g in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$g ~/comfy-venv/bin/python train/detail_eval/render.py --set "$ROOT/set" --adapter "$ADAPTER" \
    --out "$OUT" --shard $g/4 > "$ROOT/logs/render_${NAME}_g$g.log" 2>&1 &
done
wait
N=$(ls "$OUT"/*.png | wc -l); [ "$N" -eq 128 ] || { echo "only $N/128 renders" >&2; exit 3; }
SEC=$(( $(date +%s) - T0 ))
python3 -c "import json; json.dump({'adapter': '$ADAPTER', 'seconds_per_image': round($SEC * 4 / 128, 2), 'images': 128, 'gpus': 4, 'wall_s': $SEC}, open('$OUT/render.json', 'w'))"
SET=$ROOT/set CUDA_VISIBLE_DEVICES=0 bash train/detail_eval/features.sh "$OUT"
FIXED="noref v1a v2a_001000 v2a_005000"
MODELS=""
for m in $FIXED $(ls "$ROOT/renders" | grep -E "^v2a_[0-9]{6}$" | sort); do
  [ -f "$ROOT/renders/$m/feat_torch.npz" ] || continue
  [[ " $MODELS " == *" $m "* ]] && continue
  [[ " $FIXED " == *" $m "* || ! "$m" > "$NAME" ]] && MODELS="$MODELS $m"
done
CUDA_VISIBLE_DEVICES=0 ~/ccip-venv/bin/python train/detail_eval/score.py --set "$ROOT/set" --renders "$ROOT/renders" \
  --models $MODELS --base v1a --out "$ROOT/results/$NAME" --sheets "$NAME" --taxonomy "$ROOT/taxonomy/tags.csv"
cp "$OUT/render.json" "$ROOT/results/$NAME/render.json"
set -a; . ~/.config/anima-ref/secrets.env; set +a
~/comfy-venv/bin/python - "$ROOT/results/$NAME" "$NAME" <<'PY'
import os, sys
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
api.upload_folder(repo_id="0ppxnhximxr/anima-ref-kv", folder_path=sys.argv[1], path_in_repo=f"runs/v2a/detail_eval/{sys.argv[2]}",
                  commit_message=f"detail eval {sys.argv[2]} (g7e, 4 GPUs)")
print("UPLOADED", sys.argv[2], flush=True)
PY
echo "EVAL_G7E_DONE $NAME models:$MODELS wall ${SEC}s"
