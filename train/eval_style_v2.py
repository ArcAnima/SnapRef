"""Style-reference eval v2: the target character comes from the prompt, the style from the reference (decision 15:
an artist's posts share one style whatever the scene or colours).

    python train/eval_style_v2.py --adapter A.safetensors --out OUT [--mode style] [--baselines]

Cases: each of the 33 held-out style artists (~/eval/style_eval/cases.json, reference = the artist's best post)
paired with one of the 32 held-out identity characters (~/eval/v1ab summary.json; their tags from
character_tags.jsonl), x 2 scene prompts, seed 4242. Prompt = quality, safe, gender, solo, character name, its
common appearance and outfit tags, scene, i.e. what build_pairs_v2.py gives style pairs. No artist tag.
Methods: the adapter in --mode (style, or cs for an adapter without style); with --baselines also
  noref     the same prompt without a reference (Anima's default look for that prompt: the floor),
  artist    no reference, "@<artist>" added to the prompt (Anima's own idea of the artist: a text upper bound when
            Anima knows the artist; the artists are held out of our training, not of Anima's).
Scores per image (all saved, nothing averaged across metrics):
  style_<metric>   mean cos to the artist's other posts minus mean cos to the other artists' posts, metric in
                   {csd, megastyle} (style_metric_quartet.py checks which one follows the artist, not the content);
  id_target        DINOv2 cos to the target character's images (v1ab ref + truth: generated, own style),
  id_leak          DINOv2 cos to the style reference image (the reference character leaking in; lower is better).
Writes OUT/eval.json (per-method means and rows), OUT/img/<method>/<artist>__s<k>.png, OUT/sheet.jpg.
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
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "datagen"))
import anima_env  # noqa: E402
import asym  # noqa: E402
from catalog import anima_tag  # noqa: E402
from eval_pilot import SHIFT, dino_scorer, load_ref  # noqa: E402
from sample import to_pil  # noqa: E402

Q = "masterpiece, best quality, score_7, newest, highres, absurdres, safe"
NEG = "worst quality, low quality, score_1, score_2, score_3, blurry, jpeg artifacts, text, watermark, signature"
SCENES = ["solo, full body, standing, street, city, day", "solo, cowboy shot, sitting, cafe, indoors, holding cup"]


def norm(name: str) -> str:
    return name.replace("\\", "").replace(" ", "_")


def build_cases(style_dir: Path, identity_dir: Path, char_tags: Path) -> list[dict]:
    sd = json.loads((style_dir / "cases.json").read_text(encoding="utf-8"))
    idc = [c for c in json.loads((identity_dir / "summary.json").read_text(encoding="utf-8"))["cases"] if c["name"].startswith("held_out")]
    tags = {d["character"]: d for d in map(json.loads, char_tags.read_text(encoding="utf-8").splitlines())}
    cases = []
    for k, sc in enumerate(sd["cases"]):
        ic = idc[k % len(idc)]
        ch = norm(ic["character"])
        t = tags[ch]
        look = [x for x, share in t["appearance"] + t["outfit"] if share >= 0.3]
        cases.append({"artist": sc["artist"], "ref": sc["ref"]["path"], "style_targets": [p["path"] for p in sc["style_targets"]],
                      "character": ch, "identity_case": ic["name"], "gender": t["gender"],
                      "prompt_body": ", ".join([t["gender"], "solo", anima_tag(ch)] + [anima_tag(x) for x in look])})
    return cases


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--mode", default="style")
    ap.add_argument("--baselines", action="store_true")
    ap.add_argument("--style", type=Path, default=Path.home() / "eval" / "style_eval")
    ap.add_argument("--identity", type=Path, default=Path.home() / "eval" / "v1ab")
    ap.add_argument("--char-tags", type=Path, default=Path.home() / "eval" / "character_tags.jsonl")
    ap.add_argument("--config", type=Path, default=HERE.parent / "configs" / "train_pilot2_ft.json")
    ap.add_argument("--comfy-root", default=str(Path.home() / "ComfyUI"))
    args = ap.parse_args()
    device = torch.device("cuda", 0)
    started = time.perf_counter()
    cases = build_cases(args.style, args.identity, args.char_tags)
    root = args.style.parent
    cfg = json.loads(((args.adapter.parent / "config.json") if args.adapter and (args.adapter.parent / "config.json").exists()
                      else args.config).read_text(encoding="utf-8"))
    anima_env.setup_comfy(args.comfy_root)
    _, dm = anima_env.load_diffusion_model(cfg["comfy"]["unet_name"], device)
    methods = []
    memb = None
    if args.adapter:
        modules = asym.inject(dm, cfg["adapter"]["lora"])
        from safetensors import safe_open
        with safe_open(str(args.adapter), "pt") as h:
            modes = list(json.loads((h.metadata() or {}).get("anima_ref_kv", "{}")).get("modes", ["cs", "char"]))
        mode_embed = torch.zeros(len(modes), dm.model_channels, device=device)
        asym.load(modules, args.adapter, {"mode_embed": mode_embed})
        memb = mode_embed[modes.index(args.mode)][None]
        methods.append(f"{args.adapter.stem}_{args.mode}")
    if args.baselines:
        methods += ["noref", "artist"]
    clip = anima_env.load_text_encoder(cfg["comfy"]["clip_name"])
    vae = anima_env.load_vae(cfg["comfy"]["vae_name"])
    lf = anima_env.latent_format()
    enc_cache = {}

    def enc(t):
        if t not in enc_cache:
            enc_cache[t] = anima_env.encode_prompt(clip, dm, t, device)
        return enc_cache[t]
    files = []
    for c in cases:
        ref_img = Image.open(root / c["ref"]).convert("RGB")
        for si, scene in enumerate(SCENES):
            body = f"{Q}, {c['prompt_body']}, {scene}"
            for m in methods:
                if m == "artist":
                    prompt, ref = f"{Q}, @{anima_tag(c['artist'])}, {c['prompt_body']}, {scene}", None
                else:
                    prompt, ref = body, (None if m == "noref" else load_ref(vae, lf, ref_img, device))
                x = asym.sample(dm, enc(prompt), enc(NEG), enc(""), (1, 16, 1, 1216 // 8, 832 // 8), 4242, 28, 4.5, SHIFT,
                                device, ref, mode_emb=memb if ref is not None else None)
                img = to_pil(anima_env.vae_decode(vae, lf.process_out(x.to(device)).float())[0])
                p = args.out / "img" / m / f"{c['artist']}__s{si}.png"
                p.parent.mkdir(parents=True, exist_ok=True)
                img.save(p)
                files.append((c, si, m, p))
    del dm, clip
    torch.cuda.empty_cache()

    # scores: one metric loaded at a time
    rows = {(c["artist"], si, m): {"artist": c["artist"], "scene": si, "method": m, "character": c["character"]} for c, si, m, _ in files}
    from style_metric_quartet import siglip_scorer
    sys.argv = ["style_score"]
    from style_score import csd_scorer
    from huggingface_hub import hf_hub_download
    for name in ("csd", "megastyle"):
        if name == "csd":
            emb, model = csd_scorer(device), None
        else:
            emb, model = siglip_scorer(hf_hub_download("Gaojunyao/MegaStyle", "megastyle_encoder.pth"))
        tgt = {c["artist"]: torch.stack([emb(Image.open(root / p).convert("RGB")) for p in c["style_targets"]]) for c in cases}
        for c, si, m, p in files:
            e = emb(Image.open(p).convert("RGB"))
            own = (tgt[c["artist"]] @ e).mean().item()
            other = torch.cat([tgt[b] @ e for b in tgt if b != c["artist"]]).mean().item()
            rows[(c["artist"], si, m)][f"style_{name}"] = round(own - other, 4)
        del emb, model
        torch.cuda.empty_cache()
    emb, dino = dino_scorer(device)
    for c, si, m, p in files:
        e = emb(Image.open(p))
        icd = args.identity / c["identity_case"]
        gallery = [emb(Image.open(icd / f).convert("RGB")) for f in ("ref.png", "truth.jpg") if (icd / f).exists()]
        r = rows[(c["artist"], si, m)]
        r["id_target"] = round(float(torch.stack(gallery).mv(e).mean()), 4)
        r["id_leak"] = round(float(emb(Image.open(root / c["ref"]).convert("RGB")) @ e), 4)
    keys = [k for k in next(iter(rows.values())) if k.startswith(("style_", "id_"))]
    summary = {m: {k: round(sum(r[k] for r in rows.values() if r["method"] == m) / sum(r["method"] == m for r in rows.values()), 4)
                   for k in keys} for m in methods}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "eval.json").write_text(json.dumps({"adapter": str(args.adapter), "mode": args.mode, "summary": summary,
                                                    "cases": cases, "rows": list(rows.values()),
                                                    "seconds": round(time.perf_counter() - started)}, indent=1), encoding="utf-8")
    # sheet: style ref | each method | another post of the artist, first 10 artists, scene 0
    th = 300
    lines = []
    for c in cases[:10]:
        ims = [Image.open(root / c["ref"]).convert("RGB")] + [Image.open(args.out / "img" / m / f"{c['artist']}__s0.png") for m in methods] \
              + [Image.open(root / c["style_targets"][0]).convert("RGB")]
        lines.append([im.resize((max(1, round(im.width * th / im.height)), th)) for im in ims])
    W = max(sum(im.width + 4 for im in r) for r in lines)
    sheet = Image.new("RGB", (W, (th + 4) * len(lines)), "white")
    for k, r in enumerate(lines):
        x = 0
        for im in r:
            sheet.paste(im, (x, k * (th + 4)))
            x += im.width + 4
    sheet.save(args.out / "sheet.jpg", quality=82)
    print(json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
