"""Learn an anime style metric from artist labels (decision 15: one artist = one style), and test it on the
style_metric_quartet.py triplets, where every off-the-shelf embedding failed (CSD 39%, MegaStyle 50%, DINOv2 Gram 58%).

    ~/comfy-venv/bin/python train/style_probe.py ~/v2_pool ~/probe/style_metric_v2 OUT_DIR [--max-images 100000]

Features (frozen, computed once, cached in OUT_DIR/feat_<backbone>.npy): per image, for several layers of the
backbone, the channel mean and std over the patch tokens (AdaIN-style statistics: texture, line and shading carry
over, the spatial layout that tells a character apart mostly does not). Backbones: DINOv2-base at 448 px,
SigLIP so400m at 384 px.
Probe: z-scored features -> linear (or 1-hidden-layer MLP) -> 256-d, L2-normalised, supervised-contrastive loss with
artist labels. Each batch is built around characters drawn by two artists: for each, both artists' posts of that
character (hard negatives: same content, other style) and one post of each artist with another character
(positives: same style, other content).
Split: the probe never sees the triplet artists, the triplet characters or the triplet images; 10% of the remaining
artists are a validation split (triplets built the same way) for picking the head and the step.
Reports triplet accuracy / AUROC with an artist-cluster bootstrap CI for: raw statistics (no training), each probe,
and the subset where anchor and positive differ in lighting tags (night/dark vs not; decision 15 says an artist's
style survives that, so a style metric has to as well). Saves the best probe as OUT_DIR/style_probe.pt with the
feature recipe, for eval_style_v2.py.
"""
from __future__ import annotations

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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from style_metric_quartet import auroc, build_triplets, load_records  # noqa: E402

DEV = "cuda"
DARK = {"night", "dark", "night_sky", "sunset", "evening", "twilight", "dim_lighting", "backlighting"}
BACKBONES = {
    "dinov2": {"repo": "facebook/dinov2-base", "size": 448, "mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225),
               "layers": [2, 4, 6, 8, 10, 12], "skip_tokens": 1},
    "siglip": {"repo": "google/siglip-so400m-patch14-384", "size": 384, "mean": (0.5, 0.5, 0.5), "std": (0.5, 0.5, 0.5),
               "layers": [3, 7, 11, 15, 19, 23, 27], "skip_tokens": 0},
}


def preprocess(img: Image.Image, size: int, mean, std) -> torch.Tensor:
    """Short side -> size (bicubic), centre crop size x size, normalise. Whole image detail at a fixed size."""
    s = size / min(img.size)
    img = img.resize((max(size, round(img.width * s)), max(size, round(img.height * s))), Image.Resampling.BICUBIC)
    l, t = (img.width - size) // 2, (img.height - size) // 2
    x = torch.from_numpy(np.asarray(img.crop((l, t, l + size, t + size)), dtype=np.float32) / 255.0).permute(2, 0, 1)
    return (x - torch.tensor(mean).view(3, 1, 1)) / torch.tensor(std).view(3, 1, 1)


class TarImages(torch.utils.data.Dataset):
    """(post id) -> preprocessed tensors, read straight from the shard tars by byte offset."""

    def __init__(self, img_dir: Path, ids: list[int], spec: dict):
        self.spec = spec
        where = {}
        want = set(ids)
        for tar in sorted(img_dir.glob("shard_*.tar")):
            with tarfile.open(tar) as tf:
                for m in tf.getmembers():
                    pid = int(m.name.split(".")[0])
                    if pid in want:
                        where[pid] = (str(tar), m.offset_data, m.size)
        self.items = [(i, *where[i]) for i in ids if i in where]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, k):
        pid, tar, off, size = self.items[k]
        with open(tar, "rb") as f:
            f.seek(off)
            img = Image.open(io.BytesIO(f.read(size))).convert("RGB")
        return pid, preprocess(img, self.spec["size"], self.spec["mean"], self.spec["std"])


@torch.no_grad()
def extract(name: str, img_dir: Path, ids: list[int], cache: Path) -> dict[int, np.ndarray]:
    if cache.exists():
        d = np.load(cache, allow_pickle=False)
        return dict(zip(d["ids"].tolist(), d["feat"]))
    spec = BACKBONES[name]
    if name == "dinov2":
        from transformers import AutoModel
        model = AutoModel.from_pretrained(spec["repo"])
    else:
        from transformers import SiglipVisionModel
        model = SiglipVisionModel.from_pretrained(spec["repo"])
    model = model.to(DEV).eval().half()
    ds = TarImages(img_dir, ids, spec)
    dl = torch.utils.data.DataLoader(ds, batch_size=64, num_workers=24, pin_memory=True)
    out_ids, feats = [], []
    for k, (pid, x) in enumerate(dl):
        hs = model(pixel_values=x.to(DEV, torch.float16), output_hidden_states=True).hidden_states
        parts = []
        for layer in spec["layers"]:
            t = hs[layer][:, spec["skip_tokens"]:].float()
            parts += [t.mean(1), t.std(1)]
        feats.append(torch.cat(parts, 1).half().cpu().numpy())
        out_ids += pid.tolist()
        if k % 100 == 0:
            print(f"{name}: {len(out_ids)}/{len(ds)}", file=sys.stderr, flush=True)
    feat = np.concatenate(feats)
    np.savez(cache, ids=np.array(out_ids), feat=feat)
    del model
    torch.cuda.empty_cache()
    return dict(zip(out_ids, feat))


def triplet_scores(emb: dict[int, torch.Tensor], trip, artists_of, dark_mismatch) -> dict:
    pos = np.array([float(emb[a] @ emb[p]) for a, p, _ in trip])
    neg = np.array([float(emb[a] @ emb[n]) for a, _, n in trip])
    right = (pos > neg).astype(float)
    groups = collections.defaultdict(list)
    for k, t in enumerate(trip):
        groups[artists_of[t[0]]].append(k)
    gl = list(groups.values())
    rng = np.random.default_rng(0)
    boots = [right[[k for g in rng.choice(len(gl), len(gl)) for k in gl[g]]].mean() for _ in range(1000)]
    sub = [k for k, t in enumerate(trip) if dark_mismatch(t[0], t[1])]
    return {"accuracy": round(float(right.mean()), 3), "ci95": [round(float(x), 3) for x in np.percentile(boots, [2.5, 97.5])],
            "auroc": round(auroc(pos, neg), 3), "n": len(trip),
            "accuracy_lighting_mismatch": round(float(right[sub].mean()), 3) if sub else None, "n_lighting_mismatch": len(sub)}


def supcon(z: torch.Tensor, labels: torch.Tensor, tau: float = 0.1) -> torch.Tensor:
    sim = z @ z.T / tau
    self_mask = torch.eye(len(z), dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(self_mask, -1e9)
    pos = (labels[:, None] == labels[None, :]) & ~self_mask
    logp = sim - torch.logsumexp(sim, 1, keepdim=True)
    has = pos.any(1)
    return -((logp * pos).sum(1)[has] / pos.sum(1)[has]).mean()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pool", type=Path)
    ap.add_argument("quartet", type=Path, help="style_metric_quartet.py output (triplets.json)")
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
    # probe posts: other artists and characters only; up to --per-artist posts per artist, distinct characters first
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
    # validation triplets among the val artists (same recipe as the test triplets)
    val_recs = {p: recs[p] for a in val_artists for p in chosen[a]}
    val_trip = build_triplets(val_recs, 400, args.seed + 1)
    all_ids = sorted(set(probe_ids) | {i for t in trip for i in t})
    print(f"probe images {len(probe_ids)} ({len(chosen)} artists, {len(val_artists)} val), val triplets {len(val_trip)}, "
          f"test triplets {len(trip)}", file=sys.stderr, flush=True)
    dark = {p: bool(set(recs[p].get("scene", [])) & DARK) for p in all_ids}
    mism = lambda a, b: dark.get(a, False) != dark.get(b, False)  # noqa: E731
    artist_of = {p: recs[p]["artist"] for p in all_ids}
    result = {"probe_images": len(probe_ids), "probe_artists": len(chosen), "val_triplets": len(val_trip), "test": {}, "val": {}}

    for name in BACKBONES:
        feats = extract(name, args.pool / "hf" / "images", all_ids, args.out / f"feat_{name}.npz")
        ids_ok = [p for p in all_ids if p in feats]
        X = torch.tensor(np.stack([feats[p] for p in ids_ok]), dtype=torch.float16, device=DEV)
        row = {p: k for k, p in enumerate(ids_ok)}
        train_ids = [p for a, ps in chosen.items() if a not in val_artists for p in ps if p in row]
        tx = X[[row[p] for p in train_ids]].float()
        mu, sd = tx.mean(0), tx.std(0) + 1e-4
        del tx
        Z = ((X.float() - mu) / sd).half()   # fp16 table; batches are cast to fp32 below
        del X

        tr = [t for t in trip if all(i in row for i in t)]
        vt = [t for t in val_trip if all(i in row for i in t)]
        scored = sorted({i for t in tr + vt for i in t})
        sidx = torch.tensor([row[p] for p in scored], device=DEV)

        def as_emb(fn):   # embeddings of the triplet images only (the whole table would not fit next to Z)
            T = F.normalize(fn(Z[sidx].float()), dim=1)
            return {p: T[k] for k, p in enumerate(scored)}
        raw = as_emb(lambda z: z)
        result["test"][f"{name}_raw"] = triplet_scores(raw, tr, artist_of, mism)
        result["val"][f"{name}_raw"] = triplet_scores(raw, vt, artist_of, mism)
        # batches around characters drawn by >= 2 training artists
        tr_art = [a for a in chosen if a not in val_artists]
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
        print(f"{name}: dims {Z.shape[1]}, characters with >= 2 train artists: {len(multi)}", file=sys.stderr, flush=True)

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
        for head in ("linear", "mlp"):
            torch.manual_seed(args.seed)
            d = Z.shape[1]
            net = (torch.nn.Linear(d, 256) if head == "linear" else
                   torch.nn.Sequential(torch.nn.Linear(d, 1024), torch.nn.GELU(), torch.nn.Dropout(0.1), torch.nn.Linear(1024, 256))).to(DEV)
            opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=0.05)
            best = (-1.0, None, 0)
            for step in range(1, args.steps + 1):
                net.train()
                idx, labs = batch()
                x = Z[idx].float() + 0.1 * torch.randn(len(idx), Z.shape[1], device=DEV)   # small feature noise against memorising
                loss = supcon(F.normalize(net(x), dim=1), labs)
                opt.zero_grad()
                loss.backward()
                opt.step()
                if step % 250 == 0:
                    net.eval()
                    with torch.no_grad():
                        e = as_emb(net)
                    v = triplet_scores(e, vt, artist_of, mism)["accuracy"]
                    print(f"{name}/{head} step {step} loss {loss.item():.3f} val {v}", file=sys.stderr, flush=True)
                    if v > best[0]:
                        best = (v, {k: t.detach().clone() for k, t in net.state_dict().items()}, step)
            net.load_state_dict(best[1])
            net.eval()
            with torch.no_grad():
                e = as_emb(net)
            key = f"{name}_{head}"
            result["val"][key] = {**triplet_scores(e, vt, artist_of, mism), "step": best[2]}
            result["test"][key] = triplet_scores(e, tr, artist_of, mism)
            torch.save({"backbone": name, "spec": BACKBONES[name], "head": head, "state": best[1],
                        "mu": mu.cpu(), "sd": sd.cpu(), "dims": d}, args.out / f"probe_{key}.pt")
        print(json.dumps({k: v for k, v in result["test"].items() if k.startswith(name)}), file=sys.stderr, flush=True)
    (args.out / "result.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
