"""Does CSD measure style or content? Metadata + CSD only, no generation (A10).

    ~/comfy-venv/bin/python train/csd_check.py ~/danbooru2026 ~/style_data/heldout_artists.json OUT.json [--n 300]

Picks Danbooru posts (SFW, solo, 1 artist, 1 character, held-out artists excluded so nothing overlaps the eval)
and forms three kinds of pairs, scored by CSD cosine:
  same_artist_diff_char   style match, content differs     (what the style score should reward)
  same_char_diff_artist   content match, style differs     (what it should NOT reward)
  random                  neither
If same_char_diff_artist >= same_artist_diff_char, CSD rewards content and the style score is inflated.
Also reports DINOv2 on the same pairs for contrast (DINOv2 should be content-heavy).
"""
import argparse
import collections
import json
import os
import random
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "datagen"))
sys.path.insert(0, str(HERE))
from artist_v2_sizing import AI_TAGS, IMAGE_EXT, LAYOUT_TAGS, NOT_ARTIST  # noqa: E402

URL = "https://huggingface.co/datasets/nyanko-devs/danbooru2026/resolve/main/original/data-{:04d}.tar"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("danbooru", type=Path)
    ap.add_argument("heldout", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--n", type=int, default=300, help="pairs per kind")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--pairs-only", action="store_true", help="rebuild and dump the pair lists, no download/scoring")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    held = set(json.loads(args.heldout.read_text(encoding="utf-8"))["artists"])
    pf = pq.ParquetFile(args.danbooru / "metadata" / "posts-snapshot.parquet")
    cols = ["id", "rating", "image_width", "image_height", "tag_string_general", "tag_string_character", "tag_string_meta",
            "tag_string_artist", "file_ext", "is_deleted", "is_banned"]
    posts = []
    for g in range(0, pf.metadata.num_row_groups, 3):   # a third of the rows is plenty
        t = pf.read_row_group(g, columns=cols).to_pydict()
        for i in range(len(t["id"])):
            if (t["rating"][i] or "") != "g" or (t["file_ext"][i] or "").lower() not in IMAGE_EXT or t["is_deleted"][i] or t["is_banned"][i]:
                continue
            artists = [a for a in (t["tag_string_artist"][i] or "").split() if a not in NOT_ARTIST]
            chars = (t["tag_string_character"][i] or "").split()
            tags = set((t["tag_string_general"][i] or "").split()) | set((t["tag_string_meta"][i] or "").split())
            if len(artists) != 1 or len(chars) != 1 or artists[0] in held or "solo" not in tags or tags & (LAYOUT_TAGS | AI_TAGS):
                continue
            if min(t["image_width"][i] or 0, t["image_height"][i] or 0) < 768:
                continue
            posts.append((t["id"][i], artists[0], chars[0]))
    by_a, by_c = collections.defaultdict(list), collections.defaultdict(list)
    for p in posts:
        by_a[p[1]].append(p)
        by_c[p[2]].append(p)
    pairs = {"same_artist_diff_char": [], "same_char_diff_artist": [], "random": []}
    arts = [a for a, ps in by_a.items() if len({p[2] for p in ps}) >= 2]
    chars = [c for c, ps in by_c.items() if len({p[1] for p in ps}) >= 2 and len(ps) < 3000]   # skip mega-characters
    while len(pairs["same_artist_diff_char"]) < args.n:
        ps = by_a[rng.choice(arts)]
        a, b = rng.sample(ps, 2)
        if a[2] != b[2]:
            pairs["same_artist_diff_char"].append((a[0], b[0]))
    while len(pairs["same_char_diff_artist"]) < args.n:
        ps = by_c[rng.choice(chars)]
        a, b = rng.sample(ps, 2)
        if a[1] != b[1]:
            pairs["same_char_diff_artist"].append((a[0], b[0]))
    while len(pairs["random"]) < args.n:
        a, b = rng.sample(posts, 2)
        if a[1] != b[1] and a[2] != b[2]:
            pairs["random"].append((a[0], b[0]))
    (args.out.parent / "csd_pairs.json").write_text(json.dumps(pairs), encoding="utf-8")   # reused by other metrics
    if args.pairs_only:
        return 0
    ids = sorted({i for v in pairs.values() for p in v for i in p})
    idx = pq.read_table(args.danbooru / "index" / "tar-index.parquet")
    rows = {r["id"]: r for r in idx.filter(pc.is_in(idx["id"], value_set=pa.array(ids))).to_pylist()}
    token = os.environ["HF_TOKEN"].strip()
    img_dir = args.out.parent / "csd_check_images"
    img_dir.mkdir(parents=True, exist_ok=True)

    def get(pid):
        dst = img_dir / f"{pid}.webp"
        if dst.exists():
            return True
        r = rows.get(pid)
        if r is None:
            return False
        for _ in range(5):
            try:
                req = urllib.request.Request(URL.format(r["shard"]), headers={"Authorization": f"Bearer {token}",
                    "Range": f"bytes={r['offset_bytes']}-{r['offset_bytes'] + r['length_bytes'] - 1}"})
                data = urllib.request.urlopen(req, timeout=120).read()
                dst.write_bytes(data)
                return True
            except Exception:
                import time
                time.sleep(3)
        return False

    with ThreadPoolExecutor(24) as ex:
        list(ex.map(get, ids))
    dev = "cuda"
    from style_score import csd_scorer
    csd = csd_scorer(dev)
    from transformers import AutoImageProcessor, AutoModel
    proc = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    dino = AutoModel.from_pretrained("facebook/dinov2-base").to(dev).eval().half()
    cache = {}

    @torch.no_grad()
    def emb(pid):
        if pid not in cache:
            im = Image.open(img_dir / f"{pid}.webp").convert("RGB")
            x = proc(images=im, return_tensors="pt")["pixel_values"].to(dev).half()
            cache[pid] = (csd(im), F.normalize(dino(pixel_values=x).pooler_output.float(), dim=-1)[0])
        return cache[pid]
    res = {}
    for kind, ps in pairs.items():
        cs, ds = [], []
        for a, b in ps:
            if not ((img_dir / f"{a}.webp").exists() and (img_dir / f"{b}.webp").exists()):
                continue
            ea, eb = emb(a), emb(b)
            cs.append(float(ea[0] @ eb[0]))
            ds.append(float(ea[1] @ eb[1]))
        res[kind] = {"n": len(cs), "csd": round(sum(cs) / len(cs), 4), "dino": round(sum(ds) / len(ds), 4)}
    args.out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
