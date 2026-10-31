"""Pilot prep v2: the pilot_prep latents with per-pair target prompts (build_pairs_v2.py), only the texts are new.

    python train/add_pair_prompts.py --prep ~/train/pilot_prep --pairs ~/pilot_pairs_v2/pairs.jsonl --out ~/train/pilot_prep_v2

  * latents are not copied: the new index lists pilot_prep's shard files by relative path;
  * context table = "" and each record's ref_caption (sliced from pilot_prep's contexts, not re-encoded) + every
    distinct pair prompt (encoded here, PROCS_PER_GPU processes per GPU);
  * index["pairs"][mode] = [[target, ref, prompt_index], ...]; train_asym.py uses the third element as the target
    prompt when present.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import anima_env  # noqa: E402


def encode(part: int, parts: int, texts_path: Path, out: Path, config: dict, comfy_root: str) -> None:
    device = torch.device("cuda", 0)
    torch.set_num_threads(4)
    anima_env.setup_comfy(comfy_root)
    _, dm = anima_env.load_diffusion_model(config["comfy"]["unet_name"], device)
    clip = anima_env.load_text_encoder(config["comfy"]["clip_name"])
    texts = json.loads(texts_path.read_text(encoding="utf-8"))
    ctx, started = {}, time.perf_counter()
    for i in range(part, len(texts), parts):
        ctx[i] = anima_env.encode_prompt(clip, dm, texts[i], device)
        if len(ctx) % 5000 == 0:
            print(f"part {part}: {len(ctx)}/{len(texts) // parts} {time.perf_counter() - started:.0f}s", flush=True)
    torch.save(ctx, out / f"newctx_part{part}.pt")


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "--part":
        part, parts, texts, out, cfg, comfy = sys.argv[2:8]
        encode(int(part), int(parts), Path(texts), Path(out), json.loads(Path(cfg).read_text(encoding="utf-8")), comfy)
        return 0
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", type=Path, required=True)
    ap.add_argument("--pairs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--comfy-root", default=str(Path.home() / "ComfyUI"))
    ap.add_argument("--procs-per-gpu", type=int, default=3)
    args = ap.parse_args()
    old = json.loads((args.prep / "index.json").read_text(encoding="utf-8"))
    config = old["config"]
    args.out.mkdir(parents=True, exist_ok=True)
    pairs_in = [json.loads(l) for l in args.pairs.read_text(encoding="utf-8").splitlines() if l.strip()]
    recs_old = old["records"]
    pairs_in = [p for p in pairs_in if str(p["target"]) in recs_old and str(p["ref"]) in recs_old]
    texts = sorted({p["prompt"] for p in pairs_in})
    (args.out / "new_prompts.json").write_text(json.dumps(texts, ensure_ascii=False), encoding="utf-8")
    cfg_path = args.out / "config.json"
    cfg_path.write_text(json.dumps(config), encoding="utf-8")
    gpus = max(1, torch.cuda.device_count())
    parts = gpus * args.procs_per_gpu
    started = time.perf_counter()
    procs = [subprocess.Popen([sys.executable, __file__, "--part", str(w), str(parts), str(args.out / "new_prompts.json"),
                               str(args.out), str(cfg_path), args.comfy_root],
                              env={**os.environ, "CUDA_VISIBLE_DEVICES": str(w % gpus)}) for w in range(parts)]
    if any(p.wait() != 0 for p in procs):
        raise RuntimeError("an encoder process failed")
    new = [None] * len(texts)
    for w in range(parts):
        for i, c in torch.load(args.out / f"newctx_part{w}.pt").items():
            new[i] = c
    assert all(c is not None for c in new)
    # keep "" and the ref captions of the old table
    octx = load_file(str(args.prep / "contexts.safetensors"))
    odata, ooff = octx["data"], octx["offsets"].tolist()
    keep = sorted({0} | {r["prompts"]["ref_caption"] for r in recs_old.values()})
    remap = {o: n for n, o in enumerate(keep)}
    chunks = [odata[ooff[o]:ooff[o + 1]] for o in keep] + new
    offsets = torch.tensor([0] + [c.shape[0] for c in chunks]).cumsum(0)
    save_file({"data": torch.cat(chunks).contiguous(), "offsets": offsets}, str(args.out / "contexts.safetensors"))
    base = len(keep)
    pidx = {t: base + i for i, t in enumerate(texts)}
    records = {k: {**v, "prompts": {"ref_caption": remap[v["prompts"]["ref_caption"]]}} for k, v in recs_old.items()}
    pairs = {}
    for p in pairs_in:
        pairs.setdefault(p["mode"], []).append([str(p["target"]), str(p["ref"]), pidx[p["prompt"]]])
    files = [os.path.relpath(args.prep / f, args.out) for f in old["files"]["latents"]]
    index = {**old, "records": records, "pairs": pairs, "files": {"latents": files}, "prompts": None,
             "pair_prompts": True, "source_prep": str(args.prep), "pairs_file": str(args.pairs)}
    (args.out / "index.json").write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")
    for w in range(parts):
        (args.out / f"newctx_part{w}.pt").unlink()
    print(json.dumps({"records": len(records), "texts_new": len(texts), "texts_kept": base,
                      "pairs": {m: len(v) for m, v in pairs.items()}, "seconds": round(time.perf_counter() - started)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
