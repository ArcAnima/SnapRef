"""Pilot checkpoint evaluation: identity (52 cases, DINOv2 to the truth image) and style (held-out artists, CSD).

    python train/eval_pilot.py --adapter RUN/adapter_001000.safetensors --out ~/train/compare/<run>_001000 \
        --identity ~/eval/v1ab --style ~/eval/style_eval [--modes cs style]

identity: the cases of compare_refs.py's summary.json in --identity (refs and truth images under <case>/), rendered
  with mode cs at the same seed/steps; scored with train/score_panels.py's DINOv2 metric (dino_truth on held-out).
style: per held-out artist the reference post x 2 scene prompts x seed 4242, mode cs and mode style (when the
  adapter has it), no artist tag in the prompt; scored with train/style_score.py's CSD style metric.
Writes <out>/eval.json, <out>/identity/<case>.jpg (ref | ours | truth) and <out>/style/<artist>_<mode>_<scene>.png.
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
from sample import labeled_row, to_pil  # noqa: E402

SHIFT = 3.0


def load_ref(vae, lf, img, device):
    s = min(1.0, (1024 * 1024 / (img.width * img.height)) ** 0.5)
    img = img.resize(tuple(max(16, int(n * s) // 16 * 16) for n in img.size), Image.Resampling.LANCZOS)
    return lf.process_in(anima_env.vae_encode_image(vae, img)[None].to(device)).to(torch.bfloat16)


@torch.no_grad()
def dino_scorer(device):
    from transformers import AutoImageProcessor, AutoModel
    proc = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    model = AutoModel.from_pretrained("facebook/dinov2-base").to(device).eval().half()

    def emb(img):
        x = proc(images=img.convert("RGB"), return_tensors="pt")["pixel_values"].to(device).half()
        return F.normalize(model(pixel_values=x).pooler_output.float(), dim=-1)[0]
    return emb, model


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--identity", type=Path, default=Path.home() / "eval" / "v1ab")
    ap.add_argument("--style", type=Path, default=Path.home() / "eval" / "style_eval")
    ap.add_argument("--config", type=Path, help="train config (default: config.json next to the adapter)")
    ap.add_argument("--comfy-root", default=str(Path.home() / "ComfyUI"))
    ap.add_argument("--identity-cases", type=int, default=52)
    ap.add_argument("--style-scenes", type=int, default=2)
    args = ap.parse_args()
    device = torch.device("cuda", 0)
    cfg = json.loads((args.config or args.adapter.parent / "config.json").read_text(encoding="utf-8"))
    started = time.perf_counter()
    anima_env.setup_comfy(args.comfy_root)
    _, dm = anima_env.load_diffusion_model(cfg["comfy"]["unet_name"], device)
    modules = asym.inject(dm, cfg["adapter"]["lora"])
    from safetensors import safe_open
    with safe_open(str(args.adapter), "pt") as h:
        meta = json.loads((h.metadata() or {}).get("anima_ref_kv", "{}"))
    modes = list(meta.get("modes", ["cs", "char"]))
    mode_embed = torch.zeros(len(modes), dm.model_channels, device=device)
    asym.load(modules, args.adapter, {"mode_embed": mode_embed})
    clip = anima_env.load_text_encoder(cfg["comfy"]["clip_name"])
    cache = {}

    def enc(t):
        if t not in cache:
            cache[t] = anima_env.encode_prompt(clip, dm, t, device)
        return cache[t]
    vae = anima_env.load_vae(cfg["comfy"]["vae_name"])
    lf = anima_env.latent_format()
    steps, cfg_scale = 28, 4.5

    def render(ref_img, prompt, negative, w, h, seed, mode):
        ref = load_ref(vae, lf, ref_img, device)
        x = asym.sample(dm, enc(prompt), enc(negative), enc(""), (1, 16, 1, h // 8, w // 8), seed, steps, cfg_scale,
                        SHIFT, device, ref, mode_emb=mode_embed[modes.index(mode)][None])
        return to_pil(anima_env.vae_decode(vae, lf.process_out(x.to(device)).float())[0])

    args.out.mkdir(parents=True, exist_ok=True)
    result = {"adapter": str(args.adapter), "modes": modes}

    # identity
    summ = json.loads((args.identity / "summary.json").read_text(encoding="utf-8"))
    cases = summ["cases"][: args.identity_cases]
    emb, dino = dino_scorer(device)
    rows = []
    (args.out / "identity").mkdir(exist_ok=True)
    for c in cases:
        cd = args.identity / c["name"]
        ref_img = Image.open(cd / "ref.png").convert("RGB")
        img = render(ref_img, c["prompt"], c["negative"], int(c["width"]), int(c["height"]), int(c["seed"]), "cs")
        truth = cd / "truth.jpg"
        r = {"case": c["name"], "held_out": c["name"].startswith("held_out"), "dino_ref": float(emb(img) @ emb(ref_img))}
        if truth.exists():
            r["dino_truth"] = float(emb(img) @ emb(Image.open(truth)))
        rows.append(r)
        panels = [("reference", ref_img), (f"cs {args.adapter.stem}", img)] + ([("truth", Image.open(truth).convert("RGB"))] if truth.exists() else [])
        labeled_row(panels).save(args.out / "identity" / f"{c['name']}.jpg", quality=85)
    held = [r["dino_truth"] for r in rows if r["held_out"] and "dino_truth" in r]
    result["identity"] = {"dino_truth_held_out": round(sum(held) / len(held), 4), "n_held_out": len(held),
                          "dino_ref": round(sum(r["dino_ref"] for r in rows) / len(rows), 4), "rows": rows}
    del dino
    torch.cuda.empty_cache()

    # style
    sd = json.loads((args.style / "cases.json").read_text(encoding="utf-8"))
    style_root = args.style.parent
    smodes = [m for m in ("cs", "style") if m in modes]
    neg = "worst quality, low quality, score_1, score_2, score_3, blurry, jpeg artifacts, text, watermark, signature"
    q = "masterpiece, best quality, score_7, newest, highres, absurdres, safe, "
    (args.out / "style").mkdir(exist_ok=True)
    files = []
    for c in sd["cases"]:
        ref_img = Image.open(style_root / c["ref"]["path"]).convert("RGB")
        for si, scene in enumerate(sd["scenes"][: args.style_scenes]):
            for m in smodes:
                img = render(ref_img, q + c["ref"]["gender"] + ", " + scene, neg, 832, 1216, 4242, m)
                p = args.out / "style" / f"{c['artist']}__{m}__s{si}.png"
                img.save(p)
                files.append((c["artist"], m, p))
    del dm, modules, clip
    torch.cuda.empty_cache()
    sys.argv = ["style_score"]
    from style_score import csd_scorer
    s_emb = csd_scorer(device)
    tgt = {c["artist"]: torch.stack([s_emb(Image.open(style_root / p["path"])) for p in c["style_targets"]]) for c in sd["cases"]}
    srows = []
    for a, m, p in files:
        e = s_emb(Image.open(p))
        own = (tgt[a] @ e).mean().item()
        other = torch.cat([tgt[b] @ e for b in tgt if b != a]).mean().item()
        srows.append({"artist": a, "mode": m, "file": p.name, "style": own - other})
    result["style"] = {m: round(sum(r["style"] for r in srows if r["mode"] == m) / max(1, sum(r["mode"] == m for r in srows)), 4)
                       for m in smodes}
    result["style"]["rows"] = srows
    # contact sheets at the top level (the training dashboard lists *.jpg there)
    from PIL import ImageDraw
    def sheet(rows_, path):
        th = 300
        rows_ = [[im.resize((max(1, round(im.width * th / im.height)), th)) for im in r] for r in rows_]
        W = max(sum(im.width + 4 for im in r) for r in rows_)
        out = Image.new("RGB", (W, (th + 4) * len(rows_)), "white")
        for k, r in enumerate(rows_):
            x = 0
            for im in r:
                out.paste(im, (x, k * (th + 4))); x += im.width + 4
        out.save(path, quality=82)
    sheet([[Image.open(args.out / "identity" / f"{c['name']}.jpg")] for c in cases if c["name"].startswith("held_out")][:8],
          args.out / "identity_sheet.jpg")
    srows_img = []
    for c in sd["cases"][:10]:
        r = [Image.open(style_root / c["ref"]["path"]).convert("RGB")]
        for m in smodes:
            r.append(Image.open(args.out / "style" / f"{c['artist']}__{m}__s0.png"))
        r.append(Image.open(style_root / c["style_targets"][0]["path"]).convert("RGB"))
        srows_img.append(r)
    sheet(srows_img, args.out / "style_sheet.jpg")   # ref | modes... | another post of the artist
    result["seconds"] = round(time.perf_counter() - started)
    (args.out / "eval.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps({"identity_dino_truth": result["identity"]["dino_truth_held_out"],
                      "style": {m: result["style"][m] for m in smodes}, "seconds": result["seconds"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
