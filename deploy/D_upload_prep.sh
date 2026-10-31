#!/bin/bash
# D: stage a prep for HF (hardlinks, no copies; contexts.safetensors split into 40 GB parts) and upload the folder.
#   bash deploy/D_upload_prep.sh v2_prep   |   asym_prep   |   v2_mix_prep
# Stage: ~/hf_up/preps/<name>/ -> repo 0ppxnhximxr/anima-ref-danbooru-v2 (dataset) preps/<name>/. Nothing is deleted on HF.
set -euo pipefail
unset LD_LIBRARY_PATH
NAME=$1
SRC=~/v2_pool/hf/preps/$NAME
DST=~/hf_up/preps/$NAME
mkdir -p "$DST"
cd "$SRC"
case $NAME in
  v2_prep)    FILES="index.json texts.json encoder_check.json latent_check.json README.md D_build_v2_prep.py" ;;
  asym_prep)  FILES="index.json README.md asym_config.json $(ls latents_*.safetensors | tr '\n' ' ')" ;;
  v2_mix_prep) FILES="index.json README.md D_merge_preps.py $(ls index_3000groups.json 2>/dev/null || true)" ;;
esac
for f in $FILES; do ln -f "$SRC/$f" "$DST/$f"; done
if [ "$NAME" != v2_mix_prep ]; then
  size=$(stat -c %s contexts.safetensors)
  if [ "$size" -gt $((40 * 1024 ** 3)) ]; then
    if [ ! -f "$DST/SHA256SUMS" ]; then
      t=$(date +%s)
      sha256sum contexts.safetensors > "$DST/whole.sha" &   # same file read twice at once: mostly from page cache
      split -b 40G -d -a 2 contexts.safetensors "$DST/contexts.safetensors.part"
      wait
      (cd "$DST" && ls contexts.safetensors.part* | xargs -P 8 -n 1 sha256sum | sort -k2 > parts.sha \
        && cat whole.sha parts.sha > SHA256SUMS && rm whole.sha parts.sha)
      echo "split+sha $(( $(date +%s) - t ))s"
    fi
  else
    ln -f contexts.safetensors "$DST/contexts.safetensors"
    (cd "$DST" && sha256sum contexts.safetensors > SHA256SUMS)
  fi
fi
ls -la "$DST"
set -a; . ~/.config/anima-ref/secrets.env; set +a
t=$(date +%s)
~/comfy-venv/bin/python - "$NAME" <<'PY'
import os, sys
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"].strip())
# huggingface_hub 2.x: upload_folder splits large folders into several commits and resumes on a re-run
api.upload_folder(repo_id="0ppxnhximxr/anima-ref-danbooru-v2", repo_type="dataset",
                  folder_path=os.path.expanduser("~/hf_up"), allow_patterns=[f"preps/{sys.argv[1]}/*"],
                  commit_message=f"D: preps/{sys.argv[1]}")
print("UPLOAD_DONE", sys.argv[1], flush=True)
PY
echo "upload $(( $(date +%s) - t ))s"
