"""Render the frozen detail-eval set with one adapter (or none), mode cs, every case x seed. Resumable.

    ~/comfy-venv/bin/python train/detail_eval/render.py --set ~/detail_eval/set --adapter ADAPTER --out ~/detail_eval/renders/v1a
    ~/comfy-venv/bin/python train/detail_eval/render.py --set ~/detail_eval/set --noref --out ~/detail_eval/renders/noref

Same sampler as train/style_research/A_eval_identity.py: asym.sample, 28 steps, cfg 4.5, shift 3, reference cache on
the positive branch only, empty reference text, mode embedding "cs". --noref renders the base model without adapter.
Writes OUT/<case>_s<seed>.png and OUT/render.json (seconds per image).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import anima_env  # noqa: E402
import asym  # noqa: E402
from common import load_manifest  # noqa: E402
from eval_pilot import SHIFT, load_ref  # noqa: E402
from sample import to_pil  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--adapter", type=Path)
    ap.add_argument("--noref", action="store_true")
    ap.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[2] / "configs" / "train_v2a.json",
                    help="only its comfy model names are used; LoRA shapes come from the adapter's metadata")
    ap.add_argument("--limit", type=int, default=0, help="first N jobs only (timing)")
    ap.add_argument("--shard", default="0/1", help="K/N: render every N-th job starting at K (one process per GPU)")
    ap.add_argument("--comfy-root", default=str(Path.home() / "ComfyUI"))
    args = ap.parse_args()
    if args.noref == (args.adapter is not None):
        raise SystemExit("give exactly one of --adapter / --noref")
    man = load_manifest(args.set)
    jobs = [(c, s) for c in man["cases"] for s in c["seeds"]]
    jobs = jobs[: args.limit] if args.limit else jobs
    k, n_sh = (int(v) for v in args.shard.split("/"))
    todo = [(c, s) for i, (c, s) in enumerate(jobs) if i % n_sh == k and not (args.out / f"{c['name']}_s{s}.png").exists()]
    args.out.mkdir(parents=True, exist_ok=True)
    if not todo:
        print(json.dumps({"done": len(jobs), "rendered": 0}))
        return 0
    device = torch.device("cuda", 0)
    comfy = json.loads(args.config.read_text(encoding="utf-8"))["comfy"]
    anima_env.setup_comfy(args.comfy_root)
    _, dm = anima_env.load_diffusion_model(comfy["unet_name"], device)
    mode_emb = None
    if args.adapter:
        from safetensors import safe_open
        with safe_open(str(args.adapter), "pt") as h:
            meta = json.loads((h.metadata() or {}).get("anima_ref_kv", "{}"))
        modes = list(meta.get("modes", ["cs", "char"]))
        modules = asym.inject(dm, meta["lora"])
        mode_embed = torch.zeros(len(modes), dm.model_channels, device=device)
        asym.load(modules, args.adapter, {"mode_embed": mode_embed})
        mode_emb = mode_embed[modes.index(man["mode"])][None]
    clip = anima_env.load_text_encoder(comfy["clip_name"])
    vae = anima_env.load_vae(comfy["vae_name"])
    lf = anima_env.latent_format()
    enc_cache = {}

    def enc(t):
        if t not in enc_cache:
            enc_cache[t] = anima_env.encode_prompt(clip, dm, t, device)
        return enc_cache[t]
    started, n = time.perf_counter(), 0
    for c, seed in todo:
        ref = None
        if args.adapter:
            ref = load_ref(vae, lf, Image.open(args.set / c["name"] / "ref.png").convert("RGB"), device)
        x = asym.sample(dm, enc(c["prompt"]), enc(c["negative"]), enc(""), (1, 16, 1, c["height"] // 8, c["width"] // 8),
                        seed, man["steps"], man["cfg"], SHIFT, device, ref, mode_emb=mode_emb)
        img = to_pil(anima_env.vae_decode(vae, lf.process_out(x.to(device)).float())[0])
        tmp = args.out / f".{c['name']}_s{seed}.png"
        img.save(tmp)
        tmp.rename(args.out / f"{c['name']}_s{seed}.png")
        n += 1
        if n % 20 == 0:
            print(f"{n}/{len(todo)} {(time.perf_counter() - started) / n:.1f}s/img", flush=True)
    sec = (time.perf_counter() - started) / n
    if n_sh > 1:   # render.json is written once by eval_g7e.sh after all shards
        print(json.dumps({"shard": args.shard, "rendered": n, "seconds_per_image": round(sec, 2)}))
        return 0
    (args.out / "render.json").write_text(json.dumps({"adapter": str(args.adapter) if args.adapter else "noref",
                                                     "seconds_per_image": round(sec, 2), "images": len(jobs)}), encoding="utf-8")
    print(json.dumps({"done": len(jobs), "rendered": n, "seconds_per_image": round(sec, 2)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
