"""Test the learned style metrics (style_probe.py) on images they never saw.

    ~/comfy-venv/bin/python train/style_probe_test.py ~/v2_pool ~/probe/style_probe ~/style_data OUT_DIR

  A  triplets (AX anchor, AY same artist / other character, BX other artist / same character) whose three artists
     are absent from the probe's training, validation and first test set (no image of theirs in feat_*.npz);
     up to 3,000, two per artist; accuracy with an artist-cluster bootstrap CI, per style tier
     (artist_style_profile.py), and per tier of the anchor's artist.
  L  the lighting test of decision 15: AX dark (night/dark/sunset ... scene tags), AY same artist in daylight
     (no dark tag), BX same character by another artist; again unseen artists only.
  R  artist retrieval on the 33 style-eval artists (held out of the v2 pool: never downloaded with it), 4 posts
     each, all different characters: for each post, is the nearest other post by the same artist? (chance 3/131)
Probes: every probe_*.pt in the probe dir, plus the untrained statistics (raw) per backbone.
Writes OUT_DIR/result.json.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from style_metric_quartet import auroc, load_records  # noqa: E402
from style_probe import BACKBONES, DARK, DEV, extract, preprocess  # noqa: E402


def make_net(p: dict) -> torch.nn.Module:
    d = p["dims"]
    net = torch.nn.Linear(d, 256) if p["head"] == "linear" else torch.nn.Sequential(
        torch.nn.Linear(d, 1024), torch.nn.GELU(), torch.nn.Dropout(0.1), torch.nn.Linear(1024, 256))
    net.load_state_dict(p["state"])
    return net.to(DEV).eval()


def triplets(recs, artists_ok, n, seed, lighting=False):
    rng = random.Random(seed)
    by_ac, by_a, by_c = collections.defaultdict(list), collections.defaultdict(set), collections.defaultdict(set)
    for pid, r in recs.items():
        by_ac[(r["artist"], r["character"])].append(pid)
        by_a[r["artist"]].add(r["character"])
        by_c[r["character"]].add(r["artist"])
    dark = {pid: bool(set(r.get("scene", [])) & DARK) for pid, r in recs.items()}

    def related(a, b):
        pa, pb = recs[a].get("parent_id"), recs[b].get("parent_id")
        return pa == b or pb == a or (pa and pa == pb)
    keys = [k for k in by_ac if k[0] in artists_ok and len(by_a[k[0]]) >= 2 and len(by_c[k[1]] & artists_ok) >= 2]
    rng.shuffle(keys)
    out, used = [], collections.Counter()
    for a, x in keys:
        if used[a] >= 2:
            continue
        anchors = [p for p in by_ac[(a, x)] if dark[p]] if lighting else by_ac[(a, x)]
        if not anchors:
            continue
        anchor = rng.choice(anchors)
        pos_c = [p for y in by_a[a] - {x} for p in by_ac[(a, y)] if (not dark[p] if lighting else True)]
        neg_c = [p for b in (by_c[x] & artists_ok) - {a} for p in by_ac[(b, x)]]
        if not pos_c or not neg_c:
            continue
        pos, neg = rng.choice(pos_c), rng.choice(neg_c)
        if related(anchor, pos) or related(anchor, neg):
            continue
        out.append((anchor, pos, neg))
        used[a] += 1
        if len(out) >= n:
            break
    return out


def score(emb, trip, artist_of, tiers=None):
    pos = np.array([float(emb[a] @ emb[p]) for a, p, _ in trip])
    neg = np.array([float(emb[a] @ emb[n]) for a, _, n in trip])
    right = (pos > neg).astype(float)
    groups = collections.defaultdict(list)
    for k, t in enumerate(trip):
        groups[artist_of[t[0]]].append(k)
    gl = list(groups.values())
    rng = np.random.default_rng(0)
    boots = [right[[k for g in rng.choice(len(gl), len(gl)) for k in gl[g]]].mean() for _ in range(1000)]
    r = {"accuracy": round(float(right.mean()), 3), "ci95": [round(float(x), 3) for x in np.percentile(boots, [2.5, 97.5])],
         "auroc": round(auroc(pos, neg), 3), "n": len(trip)}
    if tiers is not None:
        tl = [tiers.get(artist_of[t[0]], "unknown") for t in trip]
        r["by_tier"] = {t: [round(float(right[[k for k, x in enumerate(tl) if x == t]].mean()), 3), tl.count(t)]
                        for t in sorted(set(tl)) if tl.count(t) >= 15}
    return r


@torch.no_grad()
def feats_of_files(name, paths):
    spec = BACKBONES[name]
    if name == "dinov2":
        from transformers import AutoModel
        model = AutoModel.from_pretrained(spec["repo"])
    else:
        from transformers import SiglipVisionModel
        model = SiglipVisionModel.from_pretrained(spec["repo"])
    model = model.to(DEV).eval().half()
    out = []
    for p in paths:
        x = preprocess(Image.open(p).convert("RGB"), spec["size"], spec["mean"], spec["std"])[None].to(DEV, torch.float16)
        hs = model(pixel_values=x, output_hidden_states=True).hidden_states
        parts = []
        for layer in spec["layers"]:
            t = hs[layer][:, spec["skip_tokens"]:].float()
            parts += [t.mean(1), t.std(1)]
        out.append(torch.cat(parts, 1)[0].cpu())
    del model
    torch.cuda.empty_cache()
    return torch.stack(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pool", type=Path)
    ap.add_argument("probe_dir", type=Path)
    ap.add_argument("style_data", type=Path)
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
    trip_a = triplets(recs, unseen, args.n, args.seed)
    trip_l = triplets(recs, unseen, args.n, args.seed + 1, lighting=True)
    profile = json.loads((args.style_data / "artist_style_profile.json").read_text(encoding="utf-8"))
    tiers = {a: v["tier"] for a, v in profile.items()}
    ids = sorted({i for t in trip_a + trip_l for i in t})
    artist_of = {i: recs[i]["artist"] for i in ids}
    print(f"unseen artists {len(unseen)}, triplets A {len(trip_a)}, L {len(trip_l)}, images {len(ids)}", file=sys.stderr, flush=True)
    sd = json.loads((args.style_data / "style_eval" / "cases.json").read_text(encoding="utf-8"))
    r_items = [(c["artist"], args.style_data / p["path"]) for c in sd["cases"] for p in [c["ref"]] + c["style_targets"]]
    result = {"unseen_artists": len(unseen), "A": {}, "L": {}, "R": {}}
    for name in BACKBONES:
        f = extract(name, args.pool / "hf" / "images", ids, args.out / f"feat_{name}.npz")
        fr = feats_of_files(name, [p for _, p in r_items])
        probes = {"raw": None}
        for pt in sorted(args.probe_dir.glob(f"probe_{name}_*.pt")):
            probes[pt.stem.split("_", 2)[2]] = torch.load(pt, map_location="cpu", weights_only=True)
        ok = [i for i in ids if i in f]
        X = torch.tensor(np.stack([f[i] for i in ok]), dtype=torch.float32, device=DEV)
        for head, p in probes.items():
            mu = (p["mu"] if p else X.mean(0).cpu()).to(DEV)
            sdv = (p["sd"] if p else X.std(0).cpu() + 1e-4).to(DEV)
            net = make_net(p) if p else (lambda z: z)
            E = F.normalize(net((X - mu) / sdv), dim=1)
            emb = {i: E[k] for k, i in enumerate(ok)}
            key = f"{name}_{head}"
            result["A"][key] = score(emb, [t for t in trip_a if all(i in emb for i in t)], artist_of, tiers)
            result["L"][key] = score(emb, [t for t in trip_l if all(i in emb for i in t)], artist_of)
            R = F.normalize(net((fr.to(DEV) - mu) / sdv), dim=1)
            sim = R @ R.T
            sim.fill_diagonal_(-2)
            nn_ = sim.argmax(1).tolist()
            arts = [a for a, _ in r_items]
            top1 = np.mean([arts[j] == arts[k] for k, j in enumerate(nn_)])
            rank = sim.argsort(1, descending=True).tolist()
            top5 = np.mean([any(arts[j] == arts[k] for j in rank[k][:5]) for k in range(len(arts))])
            result["R"][key] = {"top1": round(float(top1), 3), "top5": round(float(top5), 3), "n": len(arts),
                                "chance_top1": round(3 / (len(arts) - 1), 3)}
            print(key, result["A"][key]["accuracy"], result["L"][key]["accuracy"], result["R"][key], file=sys.stderr, flush=True)
    (args.out / "result.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
