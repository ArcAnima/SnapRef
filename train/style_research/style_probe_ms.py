"""Multi-scale style probe: style_probe.py's global DINOv2 statistics plus the same statistics on full-resolution
local crops (line and texture detail that a 448 px view of a whole 1 MP image blurs), and a lighting-hard test set.

    ~/comfy-venv/bin/python train/style_research/style_probe_ms.py ~/v2_pool ~/probe/style_metric_v2 ~/probe/style_probe OUT

Reuses style_probe.py's split (same seed: same probe artists, val artists, val triplets) and its cached global
features (STYLE_PROBE_DIR/feat_dinov2.npz). New features per image ("local"): short side -> 896 px, centre 896 crop,
2 x 2 grid of 448 crops; DINOv2-base layer 2..12 channel mean and std over the patch tokens of all four crops.
Heads (linear / MLP, supervised contrastive with artist labels, hard negatives = same character other artist):
global, local, global+local. Test sets:
  * the 487 quartet triplets (style_metric_quartet.py);
  * lighting-hard triplets from the triplet artists (never seen by any probe): anchor A.X dark scene, positive A.Y
    not dark (same artist, other character), negative B.X dark (other artist, same character, same lighting as the
    anchor). A metric that follows lighting or content picks the negative.
Saves OUT/probe_<key>.pt (format of style_probe.py plus "views") and OUT/result.json.
"""
from __future__ import annotations

import argparse
import collections
import io
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from style_metric_quartet import build_triplets, load_records  # noqa: E402
from style_probe import BACKBONES, DARK, TarImages, supcon, triplet_scores  # noqa: E402

DEV = "cuda"
SPEC = BACKBONES["dinov2"]


def local_views(img: Image.Image, size: int = 448) -> torch.Tensor:
    """Short side -> 2*size, centre crop 2*size square, 2 x 2 grid of size crops -> [4, 3, size, size] normalised."""
    s = 2 * size / min(img.size)
    img = img.resize((max(2 * size, round(img.width * s)), max(2 * size, round(img.height * s))), Image.Resampling.BICUBIC)
    l, t = (img.width - 2 * size) // 2, (img.height - 2 * size) // 2
    x = torch.from_numpy(np.asarray(img.crop((l, t, l + 2 * size, t + 2 * size)), dtype=np.float32) / 255.0).permute(2, 0, 1)
    x = (x - torch.tensor(SPEC["mean"]).view(3, 1, 1)) / torch.tensor(SPEC["std"]).view(3, 1, 1)
    return torch.stack([x[:, i * size:(i + 1) * size, j * size:(j + 1) * size] for i in range(2) for j in range(2)])


class LocalImages(TarImages):
    def __getitem__(self, k):
        pid, tar, off, size = self.items[k]
        with open(tar, "rb") as f:
            f.seek(off)
            img = Image.open(io.BytesIO(f.read(size))).convert("RGB")
        return pid, local_views(img)


@torch.no_grad()
def extract_local(img_dir: Path, ids: list[int], cache: Path) -> dict[int, np.ndarray]:
    if cache.exists():
        d = np.load(cache, allow_pickle=False)
        return dict(zip(d["ids"].tolist(), d["feat"]))
    from transformers import AutoModel
    model = AutoModel.from_pretrained(SPEC["repo"]).to(DEV).eval().half()
    ds = LocalImages(img_dir, ids, SPEC)
    dl = torch.utils.data.DataLoader(ds, batch_size=16, num_workers=24, pin_memory=True)
    out_ids, feats = [], []
    for k, (pid, x) in enumerate(dl):
        b = x.shape[0]
        hs = model(pixel_values=x.flatten(0, 1).to(DEV, torch.float16), output_hidden_states=True).hidden_states
        parts = []
        for layer in SPEC["layers"]:
            t = hs[layer][:, SPEC["skip_tokens"]:].float()            # [b*4, n, c]
            t = t.reshape(b, -1, t.shape[-1])                          # tokens of the four crops together
            parts += [t.mean(1), t.std(1)]
        feats.append(torch.cat(parts, 1).half().cpu().numpy())
        out_ids += pid.tolist()
        if k % 200 == 0:
            print(f"local: {len(out_ids)}/{len(ds)}", file=sys.stderr, flush=True)
    feat = np.concatenate(feats)
    np.savez(cache, ids=np.array(out_ids), feat=feat)
    return dict(zip(out_ids, feat))


@torch.no_grad()
def extract_global(img_dir: Path, ids: list[int], have: dict, cache: Path) -> dict[int, np.ndarray]:
    """Global features for ids missing from style_probe.py's cache (the lighting-hard images)."""
    missing = [i for i in ids if i not in have]
    if cache.exists():
        d = np.load(cache, allow_pickle=False)
        have.update(dict(zip(d["ids"].tolist(), d["feat"])))
        missing = [i for i in ids if i not in have]
    if not missing:
        return have
    from style_probe import extract
    extra = extract("dinov2", img_dir, missing, cache)
    have.update(extra)
    return have


def lighting_triplets(recs: dict, artists: set, n: int, seed: int) -> list[tuple[int, int, int]]:
    rng = random.Random(seed)
    dark = {p: bool(set(r.get("scene", [])) & DARK) for p, r in recs.items()}
    by_ac = collections.defaultdict(list)
    for p, r in recs.items():
        by_ac[(r["artist"], r["character"])].append(p)
    by_a, by_c = collections.defaultdict(set), collections.defaultdict(set)
    for (a, c) in by_ac:
        by_a[a].add(c)
        by_c[c].add(a)

    def related(x, y):
        px, py = recs[x].get("parent_id"), recs[y].get("parent_id")
        return px == y or py == x or (px and px == py)
    out, used = [], collections.Counter()
    keys = [k for k in by_ac if k[0] in artists]
    rng.shuffle(keys)
    for a, x in keys:
        if used[a] >= 3:
            continue
        anchors = [p for p in by_ac[(a, x)] if dark[p]]
        if not anchors:
            continue
        anchor = rng.choice(anchors)
        pos = [p for y in by_a[a] - {x} for p in by_ac[(a, y)] if not dark[p] and not related(anchor, p)]
        neg = [p for b in by_c[x] - {a} for p in by_ac[(b, x)] if dark[p] and not related(anchor, p)]
        if not pos or not neg:
            continue
        out.append((anchor, rng.choice(pos), rng.choice(neg)))
        used[a] += 1
        if len(out) >= n:
            break
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pool", type=Path)
    ap.add_argument("quartet", type=Path)
    ap.add_argument("probe_dir", type=Path, help="style_probe.py output (feat_dinov2.npz)")
    ap.add_argument("out", type=Path)
    ap.add_argument("--max-images", type=int, default=100000)
    ap.add_argument("--per-artist", type=int, default=4)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=20261030)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    recs = load_records(args.pool)
    trip = [tuple(t) for t in json.loads((args.quartet / "triplets.json").read_text())]
    t_artists = {recs[i]["artist"] for t in trip for i in t}
    t_chars = {recs[i]["character"] for t in trip for i in t}
    # --- style_probe.py's selection, unchanged (same rng sequence) ---
    by_artist = collections.defaultdict(list)
    for pid, r in recs.items():
        if r["artist"] not in t_artists and r["character"] not in t_chars:
            by_artist[r["artist"]].append(pid)
    artists = [a for a, ps in by_artist.items() if len({recs[p]["character"] for p in ps}) >= 2]
    rng.shuffle(artists)
    chosen = {}
    for a in artists:
        ps = sorted(by_artist[a], key=lambda p: rng.random())
        seen, pick = set(), []
        for p in ps:
            if recs[p]["character"] not in seen:
                pick.append(p)
                seen.add(recs[p]["character"])
        pick += [p for p in ps if p not in pick]
        chosen[a] = pick[: args.per_artist]
        if sum(map(len, chosen.values())) >= args.max_images:
            break
    val_artists = set(list(chosen)[: len(chosen) // 10])
    probe_ids = [p for ps in chosen.values() for p in ps]
    val_recs = {p: recs[p] for a in val_artists for p in chosen[a]}
    val_trip = build_triplets(val_recs, 400, args.seed + 1)
    # --- new: lighting-hard triplets among the triplet artists (no probe saw their posts) ---
    light = lighting_triplets(recs, t_artists, 400, args.seed + 2)
    all_ids = sorted(set(probe_ids) | {i for t in trip + light for i in t})
    print(f"probe images {len(probe_ids)}, val triplets {len(val_trip)}, test {len(trip)}, lighting-hard {len(light)}",
          file=sys.stderr, flush=True)
    img_dir = args.pool / "hf" / "images"
    d = np.load(args.probe_dir / "feat_dinov2.npz", allow_pickle=False)
    glob = dict(zip(d["ids"].tolist(), d["feat"]))
    glob = extract_global(img_dir, all_ids, glob, args.out / "feat_dinov2_extra.npz")
    loc = extract_local(img_dir, all_ids, args.out / "feat_dinov2_local.npz")
    ids_ok = [p for p in all_ids if p in glob and p in loc]
    row = {p: k for k, p in enumerate(ids_ok)}
    dark = {p: bool(set(recs[p].get("scene", [])) & DARK) for p in ids_ok}
    mism = lambda a, b: dark.get(a, False) != dark.get(b, False)  # noqa: E731
    artist_of = {p: recs[p]["artist"] for p in ids_ok}
    tr_art = [a for a in chosen if a not in val_artists]
    train_ids = [p for a in tr_art for p in chosen[a] if p in row]
    tests = {"test": [t for t in trip if all(i in row for i in t)], "val": [t for t in val_trip if all(i in row for i in t)],
             "lighting_hard": [t for t in light if all(i in row for i in t)]}
    scored = sorted({i for ts in tests.values() for t in ts for i in t})
    result = {"probe_images": len(probe_ids), "n": {k: len(v) for k, v in tests.items()}, "scores": {}}
    label = {a: k for k, a in enumerate(tr_art)}
    posts_ac = collections.defaultdict(list)
    for a in tr_art:
        for p in chosen[a]:
            if p in row:
                posts_ac[(a, recs[p]["character"])].append(p)
    arts_of_char = collections.defaultdict(set)
    for a, ch in posts_ac:
        arts_of_char[ch].add(a)
    multi = [ch for ch, s in arts_of_char.items() if len(s) >= 2]

    def batch():
        ids, labs = [], []
        for ch in rng.sample(multi, min(48, len(multi))):
            for a in rng.sample(sorted(arts_of_char[ch]), 2):
                ids.append(rng.choice(posts_ac[(a, ch)]))
                others = [p for p in chosen[a] if p in row and recs[p]["character"] != ch]
                if others:
                    ids.append(rng.choice(others))
                labs += [label[a]] * (len(ids) - len(labs))
        return torch.tensor([row[p] for p in ids], device=DEV), torch.tensor(labs, device=DEV)

    views = {"global": ("global",), "local": ("local",), "global_local": ("global", "local")}
    for vname, parts in views.items():
        X = torch.tensor(np.concatenate([np.stack([(glob if v == "global" else loc)[p] for p in ids_ok]) for v in parts], 1),
                         dtype=torch.float16, device=DEV)
        tx = X[[row[p] for p in train_ids]].float()
        mu, sd = tx.mean(0), tx.std(0) + 1e-4
        del tx
        Z = ((X.float() - mu) / sd).half()
        del X
        sidx = torch.tensor([row[p] for p in scored], device=DEV)

        def as_emb(fn):
            T = F.normalize(fn(Z[sidx].float()), dim=1)
            return {p: T[k] for k, p in enumerate(scored)}
        raw = as_emb(lambda z: z)
        result["scores"][f"{vname}_raw"] = {k: triplet_scores(raw, ts, artist_of, mism) for k, ts in tests.items()}
        for head in ("linear", "mlp"):
            torch.manual_seed(args.seed)
            dim = Z.shape[1]
            net = (torch.nn.Linear(dim, 256) if head == "linear" else
                   torch.nn.Sequential(torch.nn.Linear(dim, 1024), torch.nn.GELU(), torch.nn.Dropout(0.1), torch.nn.Linear(1024, 256))).to(DEV)
            opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=0.05)
            best = (-1.0, None, 0)
            for step in range(1, args.steps + 1):
                net.train()
                idx, labs = batch()
                x = Z[idx].float() + 0.1 * torch.randn(len(idx), dim, device=DEV)
                loss = supcon(F.normalize(net(x), dim=1), labs)
                opt.zero_grad()
                loss.backward()
                opt.step()
                if step % 250 == 0:
                    net.eval()
                    with torch.no_grad():
                        e = as_emb(net)
                    v = triplet_scores(e, tests["val"], artist_of, mism)["accuracy"]
                    print(f"{vname}/{head} step {step} loss {loss.item():.3f} val {v}", file=sys.stderr, flush=True)
                    if v > best[0]:
                        best = (v, {k: t.detach().clone() for k, t in net.state_dict().items()}, step)
            net.load_state_dict(best[1])
            net.eval()
            with torch.no_grad():
                e = as_emb(net)
            key = f"{vname}_{head}"
            result["scores"][key] = {k: triplet_scores(e, ts, artist_of, mism) for k, ts in tests.items()}
            result["scores"][key]["step"] = best[2]
            torch.save({"backbone": "dinov2", "spec": SPEC, "views": list(parts), "head": head, "state": best[1],
                        "mu": mu.cpu(), "sd": sd.cpu(), "dims": dim}, args.out / f"probe_{key}.pt")
            print(key, json.dumps({k: v["accuracy"] for k, v in result["scores"][key].items() if isinstance(v, dict)}),
                  file=sys.stderr, flush=True)
        del Z
        torch.cuda.empty_cache()
    (args.out / "lighting_hard_triplets.json").write_text(json.dumps(light), encoding="utf-8")
    (args.out / "result.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps({k: {t: v[t]["accuracy"] for t in tests} for k, v in result["scores"].items()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
