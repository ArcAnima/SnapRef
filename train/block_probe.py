"""Where do identity and style live? v1a with the reference cache kept only in some blocks (inference only, A10).

    ~/comfy-venv/bin/python train/block_probe.py --adapter ~/adapters/v1a/snapref_v1a.safetensors \
        --config ~/adapters/v1a/training_config.json --out ~/probe/blocks

Variants (28 blocks): all; none (no reference); keep only one band (early 0-6, mid-early 7-13, mid-late 14-20,
late 21-27); drop one band. For each variant:
  identity: 16 held-out cases of ~/eval/v1ab, DINOv2 to the truth image (as eval_pilot.py)
  style:    16 held-out artists of ~/eval/style_eval, 1 scene, CSD style score (as eval_pilot.py)
If identity and style fall with different bands, the two can get separate strength knobs without retraining.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import anima_env  # noqa: E402
import asym  # noqa: E402
from eval_pilot import dino_scorer, load_ref  # noqa: E402
from sample import to_pil  # noqa: E402

BANDS = {"early": range(0, 7), "midE": range(7, 14), "midL": range(14, 21), "late": range(21, 28)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--identity", type=Path, default=Path.home() / "eval" / "v1ab")
    ap.add_argument("--style", type=Path, default=Path.home() / "style_data" / "style_eval")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--comfy-root", default=str(Path.home() / "ComfyUI"))
    args = ap.parse_args()
    dev = torch.device("cuda", 0)
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    anima_env.setup_comfy(args.comfy_root)
    _, dm = anima_env.load_diffusion_model(cfg["comfy"]["unet_name"], dev)
    modules = asym.inject(dm, cfg["adapter"]["lora"])
    modes = list(cfg["adapter"]["modes"])
    E = torch.zeros(len(modes), dm.model_channels, device=dev)
    asym.load(modules, args.adapter, {"mode_embed": E})
    clip = anima_env.load_text_encoder(cfg["comfy"]["clip_name"])
    cache_txt = {}
    enc = lambda t: cache_txt.setdefault(t, anima_env.encode_prompt(clip, dm, t, dev))  # noqa: E731
    vae = anima_env.load_vae(cfg["comfy"]["vae_name"])
    lf = anima_env.latent_format()
    n = len(dm.blocks)
    variants = {"all": set(range(n)), "none": set()}
    for name, r in BANDS.items():
        variants["only_" + name] = set(r)
        variants["drop_" + name] = set(range(n)) - set(r)

    @torch.inference_mode()
    def render(ref_img, prompt, negative, w, h, seed, keep):
        ref = load_ref(vae, lf, ref_img, dev)
        ctx_r = anima_env.pad_contexts([enc("")]).to(dev, torch.bfloat16)
        full = asym.encode_reference(dm, ref, ctx_r, (h // 8, w // 8), mode_emb=E[modes.index("cs")][None])
        cache = [kv if i in keep else None for i, kv in enumerate(full)] if keep else None
        noise = torch.randn((1, 16, 1, h // 8, w // 8), generator=torch.Generator(device="cpu").manual_seed(seed))
        x = noise.to(dev)
        sig = anima_env.flow_sigmas(28, 3.0)
        pos = anima_env.pad_contexts([enc(prompt)]).to(dev, torch.bfloat16)
        neg = anima_env.pad_contexts([enc(negative)]).to(dev, torch.bfloat16)
        for i in range(28):
            s, s2 = float(sig[i]), float(sig[i + 1])
            xb, st = x.to(torch.bfloat16), torch.full((1,), s, device=dev)
            vp = asym.target_forward(dm, xb, st, pos, cache).float()
            vn = asym.target_forward(dm, xb, st, neg, None).float()
            x = x + (s2 - s) * (vn + 4.5 * (vp - vn))
        return to_pil(anima_env.vae_decode(vae, lf.process_out(x).float())[0])

    summ = json.loads((args.identity / "summary.json").read_text(encoding="utf-8"))
    icases = [c for c in summ["cases"] if c["name"].startswith("held_out")][: args.n]
    sd = json.loads((args.style / "cases.json").read_text(encoding="utf-8"))
    scases = sd["cases"][: args.n]
    root = args.style.parent
    q = "masterpiece, best quality, score_7, newest, highres, absurdres, safe, "
    sneg = "worst quality, low quality, score_1, score_2, score_3, blurry, jpeg artifacts, text, watermark, signature"
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    imgs = {}
    for vname, keep in variants.items():
        for c in icases:
            ref = Image.open(args.identity / c["name"] / "ref.png").convert("RGB")
            imgs[(vname, "id", c["name"])] = render(ref, c["prompt"], c["negative"], int(c["width"]), int(c["height"]), int(c["seed"]), keep)
        for c in scases:
            ref = Image.open(root / c["ref"]["path"]).convert("RGB")
            im = render(ref, q + c["ref"]["gender"] + ", " + sd["scenes"][0], sneg, 832, 1216, 4242, keep)
            imgs[(vname, "st", c["artist"])] = im
            im.save(args.out / f"{vname}__{c['artist']}.jpg", quality=85)
        print(f"{vname} done {time.perf_counter() - started:.0f}s", flush=True)
    del dm, modules, clip
    torch.cuda.empty_cache()
    demb, _ = dino_scorer(dev)
    from style_score import csd_scorer
    csd = csd_scorer(dev)
    tgt = {c["artist"]: torch.stack([csd(Image.open(root / p["path"])) for p in c["style_targets"]]) for c in scases}
    truth = {c["name"]: demb(Image.open(args.identity / c["name"] / "truth.jpg")) for c in icases}
    res = {}
    for vname in variants:
        d = [float(demb(imgs[(vname, "id", c["name"])]) @ truth[c["name"]]) for c in icases]
        s = []
        for c in scases:
            e = csd(imgs[(vname, "st", c["artist"])])
            own = (tgt[c["artist"]] @ e).mean().item()
            other = torch.cat([tgt[b] @ e for b in tgt if b != c["artist"]]).mean().item()
            s.append(own - other)
        res[vname] = {"dino_truth": round(sum(d) / len(d), 4), "csd_style": round(sum(s) / len(s), 4)}
        print(vname, res[vname], flush=True)
    (args.out / "block_probe.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
