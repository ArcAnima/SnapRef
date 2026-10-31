"""Back up / transfer research folders through the HF model repo (one commit per folder, no per-file calls).

    python train/style_research/hf_sync.py up LOCAL_DIR PATH_IN_REPO [--allow PATTERN ...]
    python train/style_research/hf_sync.py down PATH_IN_REPO LOCAL_DIR

Token from HF_TOKEN (secrets.env). Repo 0ppxnhximxr/anima-ref-kv (private). Nothing is deleted on HF:
upload_folder is called without delete_patterns.
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

REPO = "0ppxnhximxr/anima-ref-kv"


def main() -> int:
    op, a, b = sys.argv[1:4]
    allow = sys.argv[sys.argv.index("--allow") + 1:] if "--allow" in sys.argv else None
    token = os.environ["HF_TOKEN"].strip()
    if op == "up":
        HfApi(token=token).upload_folder(repo_id=REPO, folder_path=a, path_in_repo=b, allow_patterns=allow,
                                         commit_message=f"style_agent: {b}")
        print(f"UPLOADED {a} -> {REPO}/{b}", flush=True)
    elif op == "down":
        cache = snapshot_download(REPO, allow_patterns=[f"{a.rstrip('/')}/*"], token=token,
                                  local_dir=str(Path(b).expanduser().parent / ".hf_download"))
        src = Path(cache) / a
        dst = Path(b).expanduser()
        dst.mkdir(parents=True, exist_ok=True)
        for p in src.rglob("*"):
            if p.is_file():
                q = dst / p.relative_to(src)
                q.parent.mkdir(parents=True, exist_ok=True)
                if not q.exists():
                    shutil.copy2(p, q)
        print(f"DOWNLOADED {REPO}/{a} -> {dst}", flush=True)
    else:
        raise SystemExit("op must be up or down")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
