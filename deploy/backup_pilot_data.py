"""Back up a Danbooru pilot dataset to a private HF dataset repo: images as ~1 GB tar shards (no per-file uploads),
the records/pairs files, and optionally a prep directory (latents + contexts + index) as-is.

    python deploy/backup_pilot_data.py --data ~/pilot_data --repo 0ppxnhximxr/anima-ref-danbooru-pilot \
        [--prep ~/train/pilot_prep --prep-name pilot_prep] [--shard-gb 1.0]

Layout in the repo:
  images/images_000.tar ...   members <post id>.webp; images/index.json maps shard -> ids
  records.jsonl, pairs*.jsonl, summary.json
  preps/<prep-name>/...        latents_*.safetensors, contexts.safetensors, index.json (contexts_part*.pt skipped)
Resumable: shards already in the repo with the same size are skipped. The repo is created private.
"""
import argparse
import json
import os
import tarfile
from pathlib import Path

from huggingface_hub import HfApi


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--prep", type=Path)
    ap.add_argument("--prep-name")
    ap.add_argument("--shard-gb", type=float, default=1.0)
    ap.add_argument("--work", type=Path, default=Path.home() / "backup_work")
    args = ap.parse_args()
    api = HfApi()
    api.create_repo(args.repo, repo_type="dataset", private=True, exist_ok=True)
    assert api.repo_info(args.repo, repo_type="dataset").private, "repo is not private; refusing to upload"
    have = {s.rfilename: s.size for s in api.dataset_info(args.repo, files_metadata=True).siblings}
    args.work.mkdir(parents=True, exist_ok=True)

    imgs = sorted((args.data / "images").glob("*.webp"), key=lambda p: int(p.stem))
    limit = int(args.shard_gb * (1 << 30))
    shards, cur, size = [], [], 0
    for p in imgs:
        cur.append(p)
        size += p.stat().st_size
        if size >= limit:
            shards.append(cur)
            cur, size = [], 0
    if cur:
        shards.append(cur)
    index = {}
    for k, files in enumerate(shards):
        name = f"images_{k:03d}.tar"
        index[name] = [int(p.stem) for p in files]
        if f"images/{name}" in have:
            continue
        tar_path = args.work / name
        with tarfile.open(tar_path, "w") as tf:
            for p in files:
                tf.add(p, arcname=p.name)
        api.upload_file(path_or_fileobj=str(tar_path), path_in_repo=f"images/{name}", repo_id=args.repo,
                        repo_type="dataset", commit_message=f"images shard {k}")
        tar_path.unlink()
        print(f"uploaded {name} ({len(files)} images)", flush=True)
    idx_path = args.work / "index.json"
    idx_path.write_text(json.dumps(index), encoding="utf-8")
    api.upload_file(path_or_fileobj=str(idx_path), path_in_repo="images/index.json", repo_id=args.repo, repo_type="dataset")
    meta = [p for p in args.data.iterdir() if p.is_file() and p.suffix in (".jsonl", ".json")]
    for p in meta:
        api.upload_file(path_or_fileobj=str(p), path_in_repo=p.name, repo_id=args.repo, repo_type="dataset")
    if args.prep:
        name = args.prep_name or args.prep.name
        api.upload_folder(repo_id=args.repo, repo_type="dataset", folder_path=str(args.prep), path_in_repo=f"preps/{name}",
                          allow_patterns=["latents_*.safetensors", "contexts.safetensors", "index.json", "*.json"],
                          ignore_patterns=["contexts_part*", "newctx_part*"], commit_message=f"prep {name}")
        print(f"uploaded prep {name}", flush=True)
    print(json.dumps({"shards": len(shards), "images": len(imgs), "meta": [p.name for p in meta]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
