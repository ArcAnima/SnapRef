"""Back up a running datagen run to a private HF dataset repo as tar chunks of finished groups.

    python deploy/hf_tar_backup.py --run scenes_v2 --repo 0ppxnhximxr/anima-ref-scenes-v2 --name h100x2 \
        --data-dir ~/data/scenes_v2 [--loop 10800]

Each machine packs only what it rendered itself. A group is packed when every record of it that this
machine owns (in its current shards or already complete in its results files) is complete here, so a
group split by a reshard lands in one tar per machine. Chunks close at --chunk-gb; when all local
shards are finished the remainder is packed too. Packed group ids and sha256 are kept in
DATA_DIR/hf_backup_<name>.json; tars stay in DATA_DIR/../hf_stage_<run>/ (nothing is deleted locally or on the Hub).
Each round also uploads results files, manifest files, config and catalogs under meta/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tarfile
import time
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
META_RUN_FILES = ("manifest.jsonl", "groups.jsonl", "manifest_summary.json")
META_DATA_FILES = ("data/character_tags_v2new.jsonl", "data/character_tags_v2new_clean.jsonl",
                   "data/danbooru_characters_v2new_ge300.csv", "data/danbooru_artists_ge300.csv",
                   "data/scenes.jsonl", "data/pose_scenes.jsonl")


def jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:  # a line being written right now
            pass
    return rows


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def one_round(args, api: HfApi) -> bool:
    run_dir = ROOT / "runs" / args.run
    state_path = args.data_dir / f"hf_backup_{args.name}.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"packed": [], "chunks": []}
    packed = set(state["packed"])
    results = [p for p in sorted(args.data_dir.glob("results_gpu*.jsonl")) if not p.name.endswith(".log.jsonl")]
    done = {}
    for p in results:
        for r in jsonl(p):
            if r.get("status") == "complete":
                done[r["record_id"]] = r
    owned = {}  # group -> record ids this machine owns
    pending_local = 0
    for k in args.shards:
        for r in jsonl(run_dir / f"shard_{k}.jsonl"):
            owned.setdefault(r["group_id"], set()).add(r["record_id"])
            pending_local += r["record_id"] not in done
    for rid, r in done.items():
        owned.setdefault(r["group_id"], set()).add(rid)
    ready = sorted(g for g, ids in owned.items() if g not in packed and ids <= done.keys())
    finished = pending_local == 0

    stage = args.data_dir.parent / f"hf_stage_{args.run}"
    stage.mkdir(exist_ok=True)
    limit = args.chunk_gb * 1e9
    chunk, size = [], 0
    for i, g in enumerate(ready):
        files = [args.data_dir / done[rid][key] for rid in sorted(owned[g]) for key in ("image", "latent")
                 if done[rid].get(key)]
        chunk.append((g, files))
        size += sum(f.stat().st_size for f in files)
        if size < limit and not (finished and i == len(ready) - 1):
            continue
        n = len(state["chunks"])
        tar_path = stage / f"{args.run}_{args.name}_{n:03d}.tar"
        tmp = tar_path.with_suffix(".tar.tmp")
        with tarfile.open(tmp, "w") as tar:
            for _, fs in chunk:
                for f in fs:
                    tar.add(f, arcname=str(f.relative_to(args.data_dir)))
        tmp.rename(tar_path)
        digest = sha256(tar_path)
        api.upload_file(path_or_fileobj=str(tar_path), path_in_repo=f"tars/{tar_path.name}", repo_id=args.repo,
                        repo_type="dataset", commit_message=f"{tar_path.name} ({len(chunk)} groups)")
        state["chunks"].append({"file": tar_path.name, "sha256": digest, "bytes": tar_path.stat().st_size,
                                "groups": len(chunk), "images": sum(len(fs) for _, fs in chunk) // 2,
                                "uploaded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
        state["packed"].extend(g for g, _ in chunk)
        state_path.write_text(json.dumps(state))
        print(f"[{time.strftime('%H:%M:%S')}] uploaded {tar_path.name} {size / 1e9:.1f} GB {len(chunk)} groups", flush=True)
        chunk, size = [], 0

    state_path.write_text(json.dumps(state))
    meta = stage / f"meta_{args.name}"
    meta.mkdir(exist_ok=True)
    for p in results:
        shutil.copy2(p, meta / p.name)
    shutil.copy2(state_path, meta / state_path.name)
    (meta / f"SHA256SUMS_{args.name}").write_text("".join(f"{c['sha256']}  tars/{c['file']}\n" for c in state["chunks"]))
    api.upload_folder(folder_path=str(meta), path_in_repo=f"meta/{args.name}", repo_id=args.repo, repo_type="dataset",
                      commit_message=f"meta {args.name}")
    if not args.skip_run_meta:
        common = stage / "meta_run"
        (common / "data").mkdir(parents=True, exist_ok=True)
        for f in META_RUN_FILES:
            shutil.copy2(run_dir / f, common / f)
        shutil.copy2(ROOT / "configs" / f"datagen_{args.run}.json", common / f"datagen_{args.run}.json")
        for f in META_DATA_FILES:
            if (ROOT / f).exists():
                shutil.copy2(ROOT / f, common / f)
        api.upload_folder(folder_path=str(common), path_in_repo="meta/run", repo_id=args.repo, repo_type="dataset",
                          commit_message="run manifest, config, catalogs")
    print(f"[{time.strftime('%H:%M:%S')}] round done: packed {len(state['packed'])} groups, "
          f"{len(state['chunks'])} chunks, local pending {pending_local}", flush=True)
    return finished and len(packed) + len(ready) == len(state["packed"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--name", required=True, help="machine name, keeps tar/meta names apart")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--shards", type=int, nargs="+", required=True, help="shard indices this machine runs")
    parser.add_argument("--chunk-gb", type=float, default=12)
    parser.add_argument("--loop", type=int, help="seconds between rounds; stop once everything local is packed")
    parser.add_argument("--skip-run-meta", action="store_true")
    args = parser.parse_args()
    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(args.repo, repo_type="dataset", private=True, exist_ok=True)
    while True:
        if one_round(args, api) or not args.loop:
            return 0
        time.sleep(args.loop)


if __name__ == "__main__":
    raise SystemExit(main())
