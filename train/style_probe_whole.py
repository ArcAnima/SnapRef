"""Same tests as style_probe_test.py (A unseen artists, L lighting, R artist retrieval), but each image is fed WHOLE:
resized to about the backbone's square area (448² DINOv2, 384² SigLIP) with its aspect ratio kept, sides rounded to
the patch size, no crop. The probes (trained on centre crops) are used unchanged, so this shows whether they still
work on a full picture.

    ~/comfy-venv/bin/python train/style_probe_whole.py ~/v2_pool ~/probe/style_probe ~/style_data ~/probe/style_probe_test OUT_DIR

Reads the triplets of the earlier run (OUT of style_probe_test.py: triplets.json if present, else rebuilt with its
seed) so both numbers are on the same sets.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import tarfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from style_metric_quartet import load_records  # noqa: E402
from style_probe import BACKBONES, DEV  # noqa: E402
from style_probe_test import make_net, score, triplets  # noqa: E402

PATCH = {"dinov2": 14, "siglip": 14}


def whole(img: Image.Image, size: int, patch: int, mean, std) -> torch.Tensor:
    s = (size * size / (img.width * img.height)) ** 0.5
    w, h = max(patch, round(img.width * s / patch) * patch), max(patch, round(img.height * s / patch) * patch)
    x = torch.from_numpy(np.asarray(img.resize((w, h), Image.Resampling.BICUBIC), dtype=np.float32) / 255.0).permute(2, 0, 1)
    return (x - torch.tensor(mean).view(3, 1, 1)) / torch.tensor(std).view(3, 1, 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pool", type=Path)
    ap.add_argument("probe_dir", type=Path)
    ap.add_argument("style_data", type=Path)
    ap.add_argument("prev", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=20261031)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    recs = load_records(args.pool)
    seen = set()
    for name in BACKBONES:
        seen |= set(np.load(args.probe_dir / f"feat_{name}.npz")["ids"].tolist())
    seen_artists = {recs[i]["artist"] for i in seen if i in recs}
    unseen = {r["artist"] for r in recs.values()} - seen_artists
    trip_a = triplets(recs, unseen, args.n, args.seed)            # same seeds as style_probe_test.py
    trip_l = triplets(recs, unseen, args.n, args.seed + 1, lighting=True)
    profile = json.loads((args.style_data / "artist_style_profile.json").read_text(encoding="utf-8"))
    tiers = {a: v["tier"] for a, v in profile.items()}
    ids = sorted({i for t in trip_a + trip_l for i in t})
    artist_of = {i: recs[i]["artist"] for i in ids}
    # image bytes: pool tars by id
    want, imgs = set(ids), {}
    for js in sorted((args.pool / "hf" / "images").glob("shard_*.json")):
        have = {int(k) for k in json.loads(js.read_text())} & want
        if have:
            with tarfile.open(js.with_suffix(".tar")) as tf:
                for m in tf.getmembers():
                    pid = int(m.name.split(".")[0])
                    if pid in have:
                        imgs[pid] = tf.extractfile(m).read()
    # posts missing from the tar index have no image; the crop run dropped those triplets the same way
    trip_a = [t for t in trip_a if all(i in imgs for i in t)]
    trip_l = [t for t in trip_l if all(i in imgs for i in t)]
    ids = sorted({i for t in trip_a + trip_l for i in t})
    print(f"triplets A {len(trip_a)}, L {len(trip_l)}, images {len(ids)}", file=sys.stderr, flush=True)
    sd = json.loads((args.style_data / "style_eval" / "cases.json").read_text(encoding="utf-8"))
    r_items = [(c["artist"], args.style_data / p["path"]) for c in sd["cases"] for p in [c["ref"]] + c["style_targets"]]
    prev = json.loads((args.prev / "result.json").read_text(encoding="utf-8"))
    result = {"A": {}, "L": {}, "R": {}, "crop_result": {k: {m: prev[k].get(m) for m in prev[k]} for k in ("A", "L", "R")}}
    for name, spec in BACKBONES.items():
        if name == "dinov2":
            from transformers import AutoModel
            model = AutoModel.from_pretrained(spec["repo"])
            kw = {}
        else:
            from transformers import SiglipVisionModel
            model = SiglipVisionModel.from_pretrained(spec["repo"])
            kw = {"interpolate_pos_encoding": True}
        model = model.to(DEV).eval().half()

        @torch.no_grad()
        def feat(im):
            x = whole(im, spec["size"], PATCH[name], spec["mean"], spec["std"])[None].to(DEV, torch.float16)
            hs = model(pixel_values=x, output_hidden_states=True, **kw).hidden_states
            parts = []
            for layer in spec["layers"]:
                t = hs[layer][:, spec["skip_tokens"]:].float()
                parts += [t.mean(1), t.std(1)]
            return torch.cat(parts, 1)[0]
        F_ = {}
        for k, i in enumerate(ids):
            F_[i] = feat(Image.open(io.BytesIO(imgs[i])).convert("RGB"))
            if k % 1000 == 0:
                print(f"{name}: {k}/{len(ids)}", file=sys.stderr, flush=True)
        FR = torch.stack([feat(Image.open(p).convert("RGB")) for _, p in r_items])
        del model
        torch.cuda.empty_cache()
        X = torch.stack([F_[i] for i in ids])
        for pt in sorted(args.probe_dir.glob(f"probe_{name}_*.pt")):
            p = torch.load(pt, map_location="cpu", weights_only=True)
            net = make_net(p)
            mu, sdv = p["mu"].to(DEV), p["sd"].to(DEV)
            with torch.no_grad():
                E = F.normalize(net((X - mu) / sdv), dim=1)
                R = F.normalize(net((FR - mu) / sdv), dim=1)
            emb = {i: E[k] for k, i in enumerate(ids)}
            key = f"{name}_{p['head']}"
            result["A"][key] = score(emb, trip_a, artist_of, tiers)
            result["L"][key] = score(emb, trip_l, artist_of)
            sim = R @ R.T
            sim.fill_diagonal_(-2)
            arts = [a for a, _ in r_items]
            rank = sim.argsort(1, descending=True).tolist()
            result["R"][key] = {"top1": round(float(np.mean([arts[rank[k][0]] == arts[k] for k in range(len(arts))])), 3),
                                "top5": round(float(np.mean([any(arts[j] == arts[k] for j in rank[k][:5]) for k in range(len(arts))])), 3)}
            print(key, "whole:", result["A"][key]["accuracy"], result["L"][key]["accuracy"], result["R"][key],
                  "| crop:", prev["A"][key]["accuracy"], prev["L"][key]["accuracy"], prev["R"][key], file=sys.stderr, flush=True)
    (args.out / "result.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("A", "L", "R")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
