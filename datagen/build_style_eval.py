"""Style-reference eval set from held-out artists (Danbooru2026, tags only; images fetched by HTTP range).

    ~/venv-data/bin/python datagen/build_style_eval.py ~/danbooru2026 ~/style_data [--artists 40 --per-artist 4]

For each of the first N held-out artists (heldout_artists.json from artist_v2_sizing.py): up to K SFW posts
with distinct characters, short side >= 768 px (rules as artist_v2_sizing.py; rating g/s so the eval is SFW).
Posts are fetched by HTTP range (tar-index.parquet) into OUT/style_eval/<artist>/<id>.webp. OUT/style_eval/cases.json:
  ref: one post (the best-scored); style_targets: the artist's other posts (other characters) for the style score;
  scenes: fixed scene prompts, the same for every method.
"""
import argparse
import hashlib
import json
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq

import pyarrow as pa
from artist_v2_sizing import AI_TAGS, IMAGE_EXT, LAYOUT_TAGS, NOT_ARTIST

URL = "https://huggingface.co/datasets/nyanko-devs/danbooru2026/resolve/main/original/data-{:04d}.tar"
SCENES = [
    "solo, upper body, looking at viewer, smile, simple background",
    "solo, full body, standing, street, city, day",
    "solo, cowboy shot, sitting, cafe, indoors, holding cup",
    "solo, upper body, night, outdoors, wind, looking afar",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("danbooru", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--artists", type=int, default=40)
    ap.add_argument("--per-artist", type=int, default=4)
    args = ap.parse_args()
    held = json.loads((args.out / "heldout_artists.json").read_text(encoding="utf-8"))["artists"]
    held = sorted(held, key=lambda a: hashlib.sha256(a.encode()).hexdigest())[: args.artists]
    want = set(held)
    pf = pq.ParquetFile(args.danbooru / "metadata" / "posts-snapshot.parquet")
    cols = ["id", "rating", "image_width", "image_height", "tag_string_general", "tag_string_character",
            "tag_string_meta", "tag_string_artist", "file_ext", "is_deleted", "is_banned", "score"]
    posts = {a: [] for a in held}
    for g in range(pf.metadata.num_row_groups):
        t = pf.read_row_group(g, columns=cols).to_pydict()
        for i in range(len(t["id"])):
            artists = [a for a in (t["tag_string_artist"][i] or "").split() if a not in NOT_ARTIST]
            if len(artists) != 1 or artists[0] not in want:
                continue
            if (t["file_ext"][i] or "").lower() not in IMAGE_EXT or t["is_deleted"][i] or t["is_banned"][i]:
                continue
            if (t["rating"][i] or "") not in ("g", "s"):
                continue
            tags = set((t["tag_string_general"][i] or "").split()) | set((t["tag_string_meta"][i] or "").split())
            chars = (t["tag_string_character"][i] or "").split()
            if "solo" not in tags or len(chars) > 1 or tags & LAYOUT_TAGS or tags & AI_TAGS:
                continue
            if min(t["image_width"][i] or 0, t["image_height"][i] or 0) < 768:
                continue
            gender = "1boy" if "1boy" in tags else "1girl" if "1girl" in tags else None
            if gender is None:
                continue
            posts[artists[0]].append({"id": t["id"][i], "character": chars[0] if chars else f"oc:{t['id'][i]}",
                                      "gender": gender, "score": t["score"][i] or 0})
    cases, fetch = [], []
    for a in held:
        seen, chosen = set(), []
        for p in sorted(posts[a], key=lambda p: -p["score"]):   # well-received posts first, one per character
            if p["character"] in seen:
                continue
            seen.add(p["character"])
            chosen.append(p)
            if len(chosen) == args.per_artist:
                break
        if len(chosen) < 3:
            continue
        cases.append({"artist": a, "ref": chosen[0], "style_targets": chosen[1:]})
        fetch += [(a, p["id"]) for p in chosen]
    idx = pq.read_table(args.danbooru / "index" / "tar-index.parquet")
    ids = [i for _, i in fetch]
    rows = {r["id"]: r for r in idx.filter(pc.is_in(idx["id"], value_set=pa.array(ids))).to_pylist()}
    token = os.environ["HF_TOKEN"]

    def get(job):
        a, i = job
        dst = args.out / "style_eval" / a / f"{i}.webp"
        if dst.exists():
            return True
        r = rows.get(i)
        if r is None:
            return False
        req = urllib.request.Request(URL.format(r["shard"]), headers={
            "Authorization": f"Bearer {token}",
            "Range": f"bytes={r['offset_bytes']}-{r['offset_bytes'] + r['length_bytes'] - 1}"})
        data = urllib.request.urlopen(req, timeout=120).read()
        if data[:4] != b"RIFF":
            return False
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(data)
        return True

    with ThreadPoolExecutor(16) as ex:
        ok = list(ex.map(get, fetch))
    got = {j for j, o in zip(fetch, ok) if o}
    for c in cases:
        c["ref"]["path"] = f"style_eval/{c['artist']}/{c['ref']['id']}.webp"
        for p in c["style_targets"]:
            p["path"] = f"style_eval/{c['artist']}/{p['id']}.webp"
    cases = [c for c in cases if (c["artist"], c["ref"]["id"]) in got
             and all((c["artist"], p["id"]) in got for p in c["style_targets"])]
    (args.out / "style_eval" / "cases.json").write_text(json.dumps({"scenes": SCENES, "cases": cases}, indent=1),
                                                        encoding="utf-8")
    print(json.dumps({"artists": len(cases), "images": len(got), "failed": len(fetch) - len(got)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
