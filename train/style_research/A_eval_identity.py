"""Identity half of eval_pilot.py only (52 cases, mode cs, DINOv2 cos to the truth image on held-out cases), without
its style renders (eval_style_v2.py + rescore_style.py cover style). Same renders, seed and scorer as eval_pilot.py.

    python train/style_research/A_eval_identity.py --adapter RUN/adapter_001000.safetensors --out OUT
Writes OUT/identity.json (dino_truth_held_out: the 0.719 number of v1a) and OUT/identity/<case>.jpg.
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
import anima_env  # noqa: E402
import asym  # noqa: E402
from eval_pilot import SHIFT, dino_scorer, load_ref  # noqa: E402
from sample import labeled_row, to_pil  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--identity", type=Path, default=Path.home() / "eval" / "v1ab")
    ap.add_argument("--config", type=Path, help="train config (default: config.json next to the adapter)")
    ap.add_argument("--comfy-root", default=str(Path.home() / "ComfyUI"))
    args = ap.parse_args()
    device = torch.device("cuda", 0)
    cfg_path = args.config or args.adapter.parent / "config.json"
    if not cfg_path.exists():
        cfg_path = args.adapter.parent / "training_config.json"   # ~/adapters/v1a
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    started = time.perf_counter()
    anima_env.setup_comfy(args.comfy_root)
    _, dm = anima_env.load_diffusion_model(cfg["comfy"]["unet_name"], device)
    modules = asym.inject(dm, cfg["adapter"]["lora"])
    from safetensors import safe_open
    with safe_open(str(args.adapter), "pt") as h:
        modes = list(json.loads((h.metadata() or {}).get("anima_ref_kv", "{}")).get("modes", ["cs", "char"]))
    mode_embed = torch.zeros(len(modes), dm.model_channels, device=device)
    asym.load(modules, args.adapter, {"mode_embed": mode_embed})
    clip = anima_env.load_text_encoder(cfg["comfy"]["clip_name"])
    vae = anima_env.load_vae(cfg["comfy"]["vae_name"])
    lf = anima_env.latent_format()
    enc_cache = {}

    def enc(t):
        if t not in enc_cache:
            enc_cache[t] = anima_env.encode_prompt(clip, dm, t, device)
        return enc_cache[t]
    cases = json.loads((args.identity / "summary.json").read_text(encoding="utf-8"))["cases"][:52]
    emb, _ = dino_scorer(device)
    (args.out / "identity").mkdir(parents=True, exist_ok=True)
    rows = []
    for c in cases:
        cd = args.identity / c["name"]
        ref_img = Image.open(cd / "ref.png").convert("RGB")
        ref = load_ref(vae, lf, ref_img, device)
        x = asym.sample(dm, enc(c["prompt"]), enc(c["negative"]), enc(""), (1, 16, 1, int(c["height"]) // 8, int(c["width"]) // 8),
                        int(c["seed"]), 28, 4.5, SHIFT, device, ref, mode_emb=mode_embed[modes.index("cs")][None])
        img = to_pil(anima_env.vae_decode(vae, lf.process_out(x.to(device)).float())[0])
        truth = cd / "truth.jpg"
        r = {"case": c["name"], "held_out": c["name"].startswith("held_out"), "dino_ref": float(emb(img) @ emb(ref_img))}
        if truth.exists():
            r["dino_truth"] = float(emb(img) @ emb(Image.open(truth)))
        rows.append(r)
        panels = [("reference", ref_img), (f"cs {args.adapter.stem}", img)] + ([("truth", Image.open(truth).convert("RGB"))] if truth.exists() else [])
        labeled_row(panels).save(args.out / "identity" / f"{c['name']}.jpg", quality=85)
    held = [r["dino_truth"] for r in rows if r["held_out"] and "dino_truth" in r]
    res = {"adapter": str(args.adapter), "dino_truth_held_out": round(sum(held) / len(held), 4), "n_held_out": len(held),
           "dino_ref": round(sum(r["dino_ref"] for r in rows) / len(rows), 4), "rows": rows,
           "seconds": round(time.perf_counter() - started)}
    (args.out / "identity.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("dino_truth_held_out", "n_held_out", "dino_ref", "seconds")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
