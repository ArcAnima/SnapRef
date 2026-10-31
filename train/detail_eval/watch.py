"""Watch HF for new v2a adapters (every 5,000 steps) and run the detail eval on each; upload results to HF.

    setsid nohup ~/comfy-venv/bin/python train/detail_eval/watch.py > ~/detail_eval/logs/watch.log 2>&1 < /dev/null & disown

Every --every seconds: list 0ppxnhximxr/anima-ref-kv runs/v2a/adapter_XXXXXX.safetensors; for each step that is a
multiple of 5,000, above 5,000 and without ~/detail_eval/results/v2a_XXXXXX/scores.json: download, render the frozen set
(render.py), features (features.sh), score against noref / v1a / v2a 1000 / v2a 5000 and every earlier watched step
(score.py, base v1a), upload results/v2a_XXXXXX/ to runs/v2a/detail_eval/v2a_XXXXXX/ in one commit. Waits while the
first-comparison chains (~/detail_eval/chain*.sh) still run. Exits after --last-step is done. Never deletes anything on HF.
ComfyUI (comfy-a10) stays on: render + ComfyUI fit in the A10's 24 GB.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = "0ppxnhximxr/anima-ref-kv"
HOME = Path.home()
CODE = Path(__file__).resolve().parents[2]
PATTERN = re.compile(r"^runs/v2a/adapter_(\d{6})\.safetensors$")
FIXED = ["noref", "v1a", "v2a_001000", "v2a_005000"]


def log(msg: str) -> None:
    print(time.strftime("%Y-%m-%d %H:%M:%S"), msg, flush=True)


def busy() -> bool:
    out = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout
    return any("detail_eval/chain" in line for line in out.splitlines())


def run(cmd: list[str]) -> None:
    log("run " + " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=CODE)


def evaluate(step: int, api, root: Path) -> None:
    from huggingface_hub import hf_hub_download
    name = f"v2a_{step:06d}"
    adapter = root / "adapters" / f"{name}.safetensors"
    if not adapter.exists():
        shutil.copy(hf_hub_download(REPO, f"runs/v2a/adapter_{step:06d}.safetensors"), adapter)
    renders = root / "renders"
    run([str(HOME / "comfy-venv/bin/python"), "train/detail_eval/render.py", "--set", str(root / "set"),
         "--adapter", str(adapter), "--out", str(renders / name)])
    run(["bash", "train/detail_eval/features.sh", str(renders / name)])
    watched = sorted(p.name for p in renders.glob("v2a_*") if p.name not in FIXED and (p / "feat_torch.npz").exists())
    models = [m for m in FIXED if (renders / m / "feat_torch.npz").exists()] + [m for m in watched if m <= name]
    out = root / "results" / name
    run([str(HOME / "ccip-venv/bin/python"), "train/detail_eval/score.py", "--set", str(root / "set"), "--renders",
         str(renders), "--models", *models, "--base", "v1a", "--out", str(out), "--sheets", name])
    shutil.copy(renders / name / "render.json", out / "render.json")
    api.upload_folder(repo_id=REPO, folder_path=str(out), path_in_repo=f"runs/v2a/detail_eval/{name}",
                      commit_message=f"detail eval {name}")
    log(f"uploaded runs/v2a/detail_eval/{name}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=HOME / "detail_eval")
    ap.add_argument("--every", type=int, default=600)
    ap.add_argument("--last-step", type=int, default=30000)
    args = ap.parse_args()
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ["HF_TOKEN"])
    log(f"watching {REPO} runs/v2a/ for steps 10000..{args.last_step}")
    failures = {}
    while True:
        if busy():
            time.sleep(args.every)
            continue
        try:
            files = api.list_repo_files(REPO)
        except Exception as e:   # network / rate limit: try again next round
            log(f"list failed: {type(e).__name__}")
            time.sleep(args.every)
            continue
        steps = sorted(int(m.group(1)) for f in files if (m := PATTERN.match(f)))
        todo = [s for s in steps if s > 5000 and s % 5000 == 0 and s <= args.last_step
                and not (args.root / "results" / f"v2a_{s:06d}" / "scores.json").exists() and failures.get(s, 0) < 2]
        for s in todo:
            log(f"new adapter step {s}")
            try:
                evaluate(s, api, args.root)
            except Exception as e:
                failures[s] = failures.get(s, 0) + 1   # two tries per step, then it is skipped (see the log)
                log(f"step {s} failed ({failures[s]}/2): {type(e).__name__}: {e}")
        if (args.root / "results" / f"v2a_{args.last_step:06d}" / "scores.json").exists() or failures.get(args.last_step, 0) >= 2:
            log("last step handled; exiting")
            return 0
        time.sleep(args.every)


if __name__ == "__main__":
    sys.exit(main())
