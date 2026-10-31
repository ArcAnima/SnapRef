"""Which image embedding measures an artist's style rather than the character? Real Danbooru triplets, tags only.

    ~/comfy-venv/bin/python train/style_metric_quartet.py ~/v2_pool ~/style_data/artist_style_profile.json OUT_DIR [--n 600]

Triplet = anchor A·X (artist A drawing character X), positive A·Y (same artist, another character), hard negative
B·X (another artist, same character), all from the v2 pool (solo, one character, one artist), no parent/child
variants, outfits not checked (a real style metric must survive outfit changes too). A metric is right on a triplet
when cos(AX, AY) > cos(AX, BX). Reported per metric: accuracy, mean margin, AUROC of positive vs hard-negative
cosines, all with an artist-cluster bootstrap 95% CI, and accuracy per style tier of the anchor's artist
(artist_style_profile.py). CSD failed this kind of test on the pilot data (content > style).
Metrics: CSD ViT-L (style_score.py), MegaStyle-Encoder (SigLIP so400m fine-tuned for style, pooled output,
research licence: eval only), plain SigLIP so400m (the MegaStyle base, to see what the fine-tune adds), DINOv2-base
CLS, DINOv2 early-layer Gram, CLIP ViT-L/14.
"""
import argparse
import collections
import io
import json
import random
import sys
import tarfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
DEV = "cuda"


def load_records(pool: Path) -> dict[int, dict]:
    return {r["id"]: r for r in map(json.loads, (pool / "records.jsonl").read_text(encoding="utf-8").splitlines())}


def build_triplets(recs: dict[int, dict], n: int, seed: int) -> list[tuple[int, int, int]]:
    rng = random.Random(seed)
    by_ac, by_a, by_c = collections.defaultdict(list), collections.defaultdict(set), collections.defaultdict(set)
    for pid, r in recs.items():
        by_ac[(r["artist"], r["character"])].append(pid)
        by_a[r["artist"]].add(r["character"])
        by_c[r["character"]].add(r["artist"])

    def related(a, b):
        pa, pb = recs[a].get("parent_id"), recs[b].get("parent_id")
        return pa == b or pb == a or (pa and pa == pb)
    keys = [k for k in by_ac if len(by_a[k[0]]) >= 2 and len(by_c[k[1]]) >= 2]
    rng.shuffle(keys)
    out, used_artists = [], collections.Counter()
    for a, x in keys:
        if used_artists[a] >= 2:   # spread over many artists
            continue
        anchor = rng.choice(by_ac[(a, x)])
        y = rng.choice(sorted(by_a[a] - {x}))
        b = rng.choice(sorted(by_c[x] - {a}))
        pos, neg = rng.choice(by_ac[(a, y)]), rng.choice(by_ac[(b, x)])
        if related(anchor, pos) or related(anchor, neg):
            continue
        out.append((anchor, pos, neg))
        used_artists[a] += 1
        if len(out) >= n:
            break
    return out


def read_images(pool: Path, ids: set[int]) -> dict[int, Image.Image]:
    img_dir = pool / "hf" / "images"
    out = {}
    for js in sorted(img_dir.glob("shard_*.json")):
        have = {int(k) for k in json.loads(js.read_text())} & ids
        if not have:
            continue
        with tarfile.open(js.with_suffix(".tar")) as tf:
            for m in tf.getmembers():
                pid = int(m.name.split(".")[0])
                if pid in have:
                    out[pid] = Image.open(io.BytesIO(tf.extractfile(m).read())).convert("RGB")
    return out


def siglip_scorer(state_path: str | None):
    from transformers import AutoImageProcessor, SiglipVisionModel
    m = SiglipVisionModel.from_pretrained("google/siglip-so400m-patch14-384")
    if state_path:
        ck = torch.load(state_path, map_location="cpu", weights_only=True)
        ck = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
        # saved from transformers 4 (vision_model.* keys); transformers 5's SiglipVisionModel drops that prefix
        own = set(m.state_dict())
        if not (set(ck) & own):
            ck = {k.removeprefix("vision_model."): v for k, v in ck.items()}
        msg = m.load_state_dict(ck, strict=False)
        print("megastyle load:", len(msg.missing_keys), "missing,", len(msg.unexpected_keys), "unexpected", file=sys.stderr)
        assert len(msg.missing_keys) < 5, msg.missing_keys[:5]
    m = m.to(DEV).eval().half()
    p = AutoImageProcessor.from_pretrained("google/siglip-so400m-patch14-384")   # image side only: no tokenizer

    @torch.no_grad()
    def f(im):
        x = p(images=[im], return_tensors="pt")["pixel_values"].to(DEV).half()
        return F.normalize(m(pixel_values=x).pooler_output.float()[0], dim=0)
    return f, m


def dino_scorers():
    from transformers import AutoImageProcessor, AutoModel
    m = AutoModel.from_pretrained("facebook/dinov2-base").to(DEV).eval().half()
    p = AutoImageProcessor.from_pretrained("facebook/dinov2-base")

    @torch.no_grad()
    def both(im):
        x = p(images=im, return_tensors="pt")["pixel_values"].to(DEV).half()
        o = m(pixel_values=x, output_hidden_states=True)
        cls = F.normalize(o.last_hidden_state[0, 0].float(), dim=0)
        grams = []
        for layer in (3, 6):
            t = o.hidden_states[layer][0, 1:].float()
            t = t - t.mean(0, keepdim=True)
            grams.append(F.normalize(((t.T @ t) / t.shape[0]).flatten(), dim=0))
        return cls, torch.cat(grams) / 2 ** 0.5
    return both, m


def clip_scorer():
    from transformers import CLIPModel, CLIPProcessor
    m = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(DEV).eval().half()
    p = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    @torch.no_grad()
    def f(im):
        x = p(images=im, return_tensors="pt")["pixel_values"].to(DEV).half()
        out = m.get_image_features(pixel_values=x)
        out = out.pooler_output if hasattr(out, "pooler_output") else out   # transformers 5 returns a model output
        return F.normalize(out.float()[0], dim=0)
    return f, m


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    s = np.concatenate([pos, neg])
    r = s.argsort().argsort() + 1
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pool", type=Path)
    ap.add_argument("profile", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--seed", type=int, default=20261030)
    ap.add_argument("--megastyle", default=None, help="megastyle_encoder.pth (default: download Gaojunyao/MegaStyle)")
    args = ap.parse_args()
    recs = load_records(args.pool)
    trip = build_triplets(recs, args.n, args.seed)
    ids = {i for t in trip for i in t}
    imgs = read_images(args.pool, ids)
    trip = [t for t in trip if all(i in imgs for i in t)]
    print(f"triplets {len(trip)}, images {len(imgs)}", file=sys.stderr)
    if args.megastyle is None:
        from huggingface_hub import hf_hub_download
        args.megastyle = hf_hub_download("Gaojunyao/MegaStyle", "megastyle_encoder.pth")
    embs = {}

    def run(name, fn):
        embs[name] = {i: fn(imgs[i]).cpu() for i in sorted(imgs)}
        print("embedded", name, file=sys.stderr, flush=True)
    sys.argv = ["style_score"]
    from style_score import csd_scorer
    f = csd_scorer(DEV)
    run("csd", f)
    del f
    for name, path in (("megastyle", args.megastyle), ("siglip_so400m", None)):
        f, m = siglip_scorer(path)
        run(name, f)
        del f, m
        torch.cuda.empty_cache()
    both, m = dino_scorers()
    cache = {i: both(imgs[i]) for i in sorted(imgs)}
    embs["dinov2_cls"] = {i: v[0].cpu() for i, v in cache.items()}
    embs["dinov2_gram"] = {i: v[1].cpu() for i, v in cache.items()}
    del both, m, cache
    torch.cuda.empty_cache()
    f, m = clip_scorer()
    run("clip_l14", f)
    del f, m

    profile = json.loads(args.profile.read_text(encoding="utf-8")) if args.profile.exists() else {}
    tiers = [profile.get(recs[t[0]]["artist"], {}).get("tier", "unknown") for t in trip]
    artists = [recs[t[0]]["artist"] for t in trip]
    rng = np.random.default_rng(0)
    groups = collections.defaultdict(list)
    for k, a in enumerate(artists):
        groups[a].append(k)
    glist = list(groups.values())
    result = {"triplets": len(trip), "artists": len(glist), "tiers": dict(collections.Counter(tiers)), "metrics": {}}
    for name, e in embs.items():
        pos = np.array([float(e[a] @ e[p]) for a, p, _ in trip])
        neg = np.array([float(e[a] @ e[n]) for a, _, n in trip])
        right = (pos > neg).astype(float)
        boots = []
        for _ in range(1000):
            idx = [k for g in rng.choice(len(glist), len(glist)) for k in glist[g]]
            boots.append((right[idx].mean(), (pos[idx] - neg[idx]).mean()))
        boots = np.array(boots)
        per_tier = {t: round(float(right[[k for k, x in enumerate(tiers) if x == t]].mean()), 3)
                    for t in sorted(set(tiers)) if tiers.count(t) >= 20}
        result["metrics"][name] = {
            "accuracy": round(float(right.mean()), 3), "accuracy_ci95": [round(float(x), 3) for x in np.percentile(boots[:, 0], [2.5, 97.5])],
            "margin": round(float((pos - neg).mean()), 4), "margin_ci95": [round(float(x), 4) for x in np.percentile(boots[:, 1], [2.5, 97.5])],
            "auroc": round(auroc(pos, neg), 3), "cos_pos": round(float(pos.mean()), 3), "cos_neg": round(float(neg.mean()), 3),
            "accuracy_by_tier": per_tier}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "result.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    (args.out / "triplets.json").write_text(json.dumps(trip), encoding="utf-8")
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
