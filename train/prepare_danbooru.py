"""Training tensors for the Danbooru pilot (build_danbooru_pilot.py output), in the train_asym.py index format.

    python train/prepare_danbooru.py --data ~/pilot_data --out ~/train/pilot_prep [--config configs/train_pilot.json]

  * each post is resized and centre-cropped to the nearest of 7 ~1 MP buckets (BUCKETS) and VAE-encoded once; it is
    the target latent and the reference latent. Fixed buckets keep batches of 4 distinct pairs (free sizes gave
    one bucket per pair);
  * prompts per post: stripped (cs), with_artist (char), with_character (style), ref_caption and "" are encoded
    once with the Anima text encoder (one process per GPU);
  * index["pairs"] = {mode: [[target, ref], ...]} from pairs.jsonl; train_asym.py samples modes by
    adapter.modes weights and pairs uniformly within a mode (bucketed by target size and reference size).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import torch
from PIL import Image, ImageOps
from safetensors.torch import save_file

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import anima_env  # noqa: E402
from prepare_asym import write_shards  # noqa: E402

PROMPT_OF_MODE = {"cs": "stripped", "char": "with_artist", "style": "with_character"}
BUCKETS = [(1024, 1024), (896, 1152), (1152, 896), (832, 1216), (1216, 832), (768, 1344), (1344, 768)]


def bucket_for(w: int, h: int) -> tuple[int, int]:
    return min(BUCKETS, key=lambda b: abs(math.log((w / h) / (b[0] / b[1]))))


def work(part: int, parts: int, data: Path, out: Path, config: dict, comfy_root: str) -> None:
    """One GPU: VAE latents and text contexts for every parts-th record."""
    device = torch.device("cuda", 0)
    torch.set_num_threads(4)   # several workers per GPU; single-prompt encoding is launch-overhead bound
    anima_env.setup_comfy(comfy_root)
    _, dm = anima_env.load_diffusion_model(config["comfy"]["unet_name"], device)
    recs = [json.loads(l) for l in (data / "records.jsonl").read_text(encoding="utf-8").splitlines()]
    texts = json.loads((out / "prompts.json").read_text(encoding="utf-8"))
    clip = anima_env.load_text_encoder(config["comfy"]["clip_name"])
    ctx, started = {}, time.perf_counter()
    for i in range(part, len(texts), parts):
        ctx[i] = anima_env.encode_prompt(clip, dm, texts[i], device)
        if len(ctx) % 5000 == 0:
            print(f"part {part}: texts {len(ctx)}/{len(texts) // parts} {time.perf_counter() - started:.0f}s", flush=True)
    torch.save(ctx, out / f"contexts_part{part}.pt")
    print(f"part {part}: {len(ctx)} texts in {time.perf_counter() - started:.0f}s", flush=True)
    del clip, dm
    torch.cuda.empty_cache()
    vae = anima_env.load_vae(config["comfy"]["vae_name"])
    lat, bad = {}, []
    for n, r in enumerate(recs[part::parts]):
        p = data / "images" / f"{r['id']}.webp"
        try:
            img = Image.open(p)
            if img.mode in ("RGBA", "LA", "P"):   # transparency onto white (convert("RGB") alone leaves black/garbage)
                img = img.convert("RGBA")
                bg = Image.new("RGB", img.size, (255, 255, 255))
                bg.paste(img, mask=img.split()[-1])
                img = bg
            img = img.convert("RGB")
        except Exception:
            bad.append(r["id"])
            continue
        img = ImageOps.fit(img, bucket_for(img.width, img.height), Image.Resampling.LANCZOS, centering=(0.5, 0.4))
        lat[str(r["id"])] = anima_env.vae_encode_image(vae, img).to(torch.float16)
        if n % 2000 == 0:
            print(f"part {part}: latents {n}/{len(recs) // parts} {time.perf_counter() - started:.0f}s", flush=True)
    files = write_shards(out / f"latents_p{part}", lat)
    (out / f"latents_part{part}.json").write_text(json.dumps({"files": files, "bad": bad,
                                                              "sizes": {k: list(v.shape[-2:]) for k, v in lat.items()}}),
                                                  encoding="utf-8")


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "--part":
        part, parts, data, out, cfg, comfy = sys.argv[2:8]
        work(int(part), int(parts), Path(data), Path(out), json.loads(Path(cfg).read_text(encoding="utf-8")), comfy)
        return 0
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "train_pilot.json")
    ap.add_argument("--comfy-root", default=str(Path.home() / "ComfyUI"))
    ap.add_argument("--procs-per-gpu", type=int, default=3)
    args = ap.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    args.out.mkdir(parents=True, exist_ok=True)
    recs = [json.loads(l) for l in (args.data / "records.jsonl").read_text(encoding="utf-8").splitlines()]
    recs = [r for r in recs if (args.data / "images" / f"{r['id']}.webp").exists()]
    prompts: dict[str, int] = {"": 0}
    index_records = {}
    for r in recs:
        entry = {k: prompts.setdefault(v, len(prompts)) for k, v in r["prompts"].items()}
        entry["ref_caption"] = prompts.setdefault(r["ref_caption"], len(prompts))
        index_records[str(r["id"])] = {"character": r["character"], "artist": r["artist"], "rating": r["rating"],
                                       "prompts": entry, "focus": None, "framing": None}
    texts = list(prompts)
    (args.out / "prompts.json").write_text(json.dumps(texts, ensure_ascii=False), encoding="utf-8")
    gpus = max(1, torch.cuda.device_count())
    parts = gpus * args.procs_per_gpu
    started = time.perf_counter()
    procs = [subprocess.Popen([sys.executable, __file__, "--part", str(w), str(parts), str(args.data), str(args.out),
                               str(args.config), args.comfy_root], env={**os.environ, "CUDA_VISIBLE_DEVICES": str(w % gpus)})
             for w in range(parts)]
    if any(p.wait() != 0 for p in procs):
        raise RuntimeError("a worker failed")
    contexts = [None] * len(texts)
    for w in range(parts):
        for i, c in torch.load(args.out / f"contexts_part{w}.pt").items():
            contexts[i] = c
    offsets = torch.tensor([0] + [c.shape[0] for c in contexts]).cumsum(0)
    save_file({"data": torch.cat(contexts).contiguous(), "offsets": offsets}, str(args.out / "contexts.safetensors"))
    files, bad = [], set()
    for w in range(parts):
        info = json.loads((args.out / f"latents_part{w}.json").read_text(encoding="utf-8"))
        files += info["files"]
        bad |= set(map(str, info["bad"]))
        for k, (h, wd) in info["sizes"].items():
            index_records[k]["width"], index_records[k]["height"] = wd * 8, h * 8
    index_records = {k: v for k, v in index_records.items() if k not in bad and "width" in v}
    pairs = {}
    for line in (args.data / "pairs.jsonl").read_text(encoding="utf-8").splitlines():
        p = json.loads(line)
        t, r = str(p["target"]), str(p["ref"])
        if t in index_records and r in index_records:
            pairs.setdefault(p["mode"], []).append([t, r])
    index = {"created": time.time(), "config": config, "data_dir": str(args.data), "records": index_records,
             "groups": {}, "pairs": pairs, "prompt_of_mode": PROMPT_OF_MODE, "prompts": texts, "eval_cases": [],
             "held_out_characters": [], "files": {"latents": files}}
    (args.out / "index.json").write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"records": len(index_records), "bad": len(bad), "texts": len(texts),
                      "pairs": {m: len(v) for m, v in pairs.items()}, "seconds": round(time.perf_counter() - started)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
