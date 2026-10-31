"""v2 latent check: decode the HF latents of random records with Anima's VAE and compare with the WebP they were
encoded from (images/shard_NNNNN.tar of 0ppxnhximxr/anima-ref-danbooru-v2), PSNR in RGB 0..255.

    python train/D_check_latents.py --prep ~/v2_pool/hf/preps/v2_prep --hf ~/v2_pool/hf [--n 20]
"""
from __future__ import annotations

import argparse
import io
import json
import random
import sys
import tarfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors import safe_open

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import anima_env  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", type=Path, required=True)
    ap.add_argument("--hf", type=Path, required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--comfy-root", default=str(Path.home() / "ComfyUI"))
    ap.add_argument("--save", type=Path, help="write a side-by-side jpg of the first 4 here")
    args = ap.parse_args()
    index = json.loads((args.prep / "index.json").read_text(encoding="utf-8"))
    ids = random.Random(args.seed).sample(sorted(index["records"]), args.n)
    where = {}
    for js in sorted((args.hf / "images").glob("shard_*.json")):
        for k in json.loads(js.read_text()):
            where[k] = js.stem[6:]
    anima_env.setup_comfy(args.comfy_root)
    vae = anima_env.load_vae(index["config"]["comfy"]["vae_name"])
    rows, tiles = [], []
    for pid in ids:
        num = where[pid]
        with safe_open(str(args.hf / "latents" / f"latents_{num}.safetensors"), "pt") as f:
            lat = f.get_tensor(pid)
        with tarfile.open(args.hf / "images" / f"shard_{num}.tar") as tf:
            img = Image.open(io.BytesIO(tf.extractfile(f"{pid}.webp").read())).convert("RGB")
        dec = anima_env.vae_decode(vae, lat[None].float())[0].numpy()
        ref = np.asarray(img, dtype=np.float32) / 255.0
        r = index["records"][pid]
        assert dec.shape == ref.shape == (r["height"], r["width"], 3), (pid, dec.shape, ref.shape, r["width"], r["height"])
        mse = float(((dec * 255 - ref * 255) ** 2).mean())
        psnr = 10 * np.log10(255 ** 2 / mse)
        rows.append({"id": pid, "shard": num, "size": [r["width"], r["height"]], "latent": list(lat.shape),
                     "dtype": str(lat.dtype), "psnr": round(psnr, 2)})
        print(json.dumps(rows[-1]), flush=True)
        if len(tiles) < 4:
            tiles.append(np.concatenate([ref, dec], 1))
    ps = [r["psnr"] for r in rows]
    summary = {"n": len(ps), "psnr_mean": round(float(np.mean(ps)), 2), "psnr_min": min(ps), "psnr_max": max(ps)}
    print(json.dumps(summary), flush=True)
    (args.prep / "latent_check.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    if args.save:
        h = min(t.shape[0] for t in tiles)
        sheet = np.concatenate([np.asarray(Image.fromarray((t * 255).astype(np.uint8)).resize(
            (round(t.shape[1] * h / t.shape[0]), h))) for t in tiles], 1)
        Image.fromarray(sheet).resize((sheet.shape[1] // 2, h // 2)).save(args.save, quality=88)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
