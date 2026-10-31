"""Pilot data from Danbooru2026 only (no generated images): cs / char / style reference pairs, tags only.

    ~/venv-data/bin/python datagen/build_danbooru_pilot.py ~/danbooru2026 ~/style_data OUT [--images 60000]

Post filter (decisions 10/11, tags only, no image inspection):
  still image, not deleted/banned, short side >= 768 px, exactly one artist tag, solo, exactly one character tag,
  no layout tags, not ai-generated; NSFW (q/e) without minor tags, and minor characters (old 435 list, an
  over-strict bound) SFW only; held-out characters (32) and held-out artists (200) are never used.
Pairs (target, reference):
  cs     same artist, same character, other post
  char   same character, other artist          (target prompt names the target artist)
  style  same artist, other character          (target prompt names the target character and its look)
Selection: artists with >= 2 kept characters, posts capped per artist and per character, so pairs of every mode
exist without a few artists or characters dominating.
Writes OUT/records.jsonl (one post per line: id, artist, character, rating, gender, size, prompts, ref_caption),
OUT/pairs.jsonl ({mode, target, ref}), OUT/images/<id>.webp (HTTP range from the tar index), OUT/summary.json.
Prompt pieces (Danbooru tags -> Anima spelling via catalog.anima_tag; taxonomy Mexes/danbooru-tag-taxonomy):
  scene      = pose / expression / camera / scenery / props tags (no appearance, clothing, character, artist)
  look       = appearance (hair, eyes, anatomy marks) + clothing tags
  prompts:   stripped = quality, rating, gender, solo, scene
             with_artist = quality, @artist, rating, gender, solo, scene
             with_character = quality, rating, gender, solo, character, look, scene
  ref_caption = gender, solo, look
"""
import argparse
import collections
import hashlib
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from artist_v2_sizing import AI_TAGS, IMAGE_EXT, LAYOUT_TAGS, MINOR_TAGS, NOT_ARTIST  # noqa: E402
from catalog import anima_tag  # noqa: E402

URL = "https://huggingface.co/datasets/nyanko-devs/danbooru2026/resolve/main/original/data-{:04d}.tar"
QUALITY = "masterpiece, best quality, score_7, newest, highres, absurdres"
RATING = {"g": "safe", "s": "sensitive", "q": "nsfw", "e": "explicit"}
SCENE_CATS = {"pose", "scenery"}
SCENE_SLOTS = {"equipment:handheld", "creature:animal"}
LOOK_CATS = {"appearance", "clothing"}
DROP_TAGS = {"solo", "1girl", "1boy", "1other", "male_focus", "female_focus", "highres", "absurdres"}


def norm(name: str) -> str:
    return name.replace("\\(", "(").replace("\\)", ")").replace(" ", "_").lower()


def taxonomy(path: Path) -> dict[str, tuple[str, str]]:
    t = pq.read_table(path, columns=["tag", "danbooru_category", "category", "slot"]).to_pydict()
    return {tag: (cat, slot) for tag, dc, cat, slot in zip(t["tag"], t["danbooru_category"], t["category"], t["slot"])
            if dc == "general"}


def split_tags(tags: list[str], tax) -> tuple[list[str], list[str]]:
    scene, look = [], []
    for tag in tags:
        if tag in DROP_TAGS or tag in MINOR_TAGS:
            continue
        cat, slot = tax.get(tag, (None, None))
        if cat in LOOK_CATS and slot != "appearance:demographics":
            look.append(tag)
        elif cat in SCENE_CATS or slot in SCENE_SLOTS:
            scene.append(tag)
    return scene, look


def download(ids: list[int], index_path: Path, img_dir: Path, workers: int) -> dict:
    """Fetch posts by HTTP range from the tar index; resumable (existing files are skipped). Retries with backoff
    on HTTP 429/5xx and network errors; ids missing from the index are counted, not retried."""
    import time
    import urllib.error
    idx = pq.read_table(index_path)
    rows = {r["id"]: r for r in idx.filter(pc.is_in(idx["id"], value_set=pa.array(ids))).to_pylist()}
    token = os.environ["HF_TOKEN"].strip()
    img_dir.mkdir(parents=True, exist_ok=True)
    stats = collections.Counter(not_in_index=len(set(ids) - set(rows)))

    def get(pid):
        dst = img_dir / f"{pid}.webp"
        if dst.exists():
            return "exists"
        r = rows[pid]
        delay = 2.0
        for _ in range(8):
            try:
                req = urllib.request.Request(URL.format(r["shard"]), headers={
                    "Authorization": f"Bearer {token}",
                    "Range": f"bytes={r['offset_bytes']}-{r['offset_bytes'] + r['length_bytes'] - 1}"})
                data = urllib.request.urlopen(req, timeout=120).read()
                if data[:4] != b"RIFF":
                    return "bad_bytes"
                tmp = dst.with_suffix(".part")
                tmp.write_bytes(data)
                tmp.rename(dst)
                return "ok"
            except urllib.error.HTTPError as e:
                if e.code not in (429, 500, 502, 503, 504):
                    return f"http_{e.code}"
                stats[f"retry_http_{e.code}"] += 1
            except Exception as e:  # timeouts, resets
                stats[f"retry_{type(e).__name__}"] += 1
            time.sleep(delay)
            delay = min(60.0, delay * 2)
        return "gave_up"

    todo = [i for i in ids if i in rows]
    with ThreadPoolExecutor(workers) as ex:
        for n, res in enumerate(ex.map(get, todo), 1):
            stats[res] += 1
            if n % 5000 == 0:
                print(f"download {n}/{len(todo)} {dict(stats)}", file=sys.stderr, flush=True)
    return {"download_" + k: v for k, v in stats.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("danbooru", type=Path)
    ap.add_argument("style_data", type=Path, help="artist_v2_sizing output (heldout_artists.json)")
    ap.add_argument("out", type=Path)
    ap.add_argument("--images", type=int, default=60000)
    ap.add_argument("--per-artist", type=int, default=12)
    ap.add_argument("--per-character", type=int, default=40)
    ap.add_argument("--seed", type=int, default=20261030)
    ap.add_argument("--no-download", action="store_true")
    ap.add_argument("--download-only", action="store_true", help="skip selection; fetch the images of OUT/records.jsonl")
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()
    if args.download_only:
        ids = [json.loads(l)["id"] for l in (args.out / "records.jsonl").read_text(encoding="utf-8").splitlines()]
        print(json.dumps(download(ids, args.danbooru / "index" / "tar-index.parquet", args.out / "images", args.workers)), flush=True)
        return 0
    rng = random.Random(args.seed)
    home = Path.home()
    used = json.loads((home / "gelbooru2026" / "characters_used.json").read_text(encoding="utf-8"))
    held_chars = {norm(c) for c in used["held_out"]}
    minors = {norm(l.strip()) for l in (home / "gelbooru2026" / "minor_characters.txt").read_text(encoding="utf-8").splitlines() if l.strip()}
    held_artists = set(json.loads((args.style_data / "heldout_artists.json").read_text(encoding="utf-8"))["artists"])
    tax = taxonomy(home / "hf" / "taxonomy" / "tags.parquet")

    pf = pq.ParquetFile(args.danbooru / "metadata" / "posts-snapshot.parquet")
    cols = ["id", "rating", "image_width", "image_height", "tag_string_general", "tag_string_character",
            "tag_string_meta", "tag_string_artist", "file_ext", "is_deleted", "is_banned", "score"]
    posts, c = [], collections.Counter()
    for g in range(pf.metadata.num_row_groups):
        t = pf.read_row_group(g, columns=cols).to_pydict()
        for i in range(len(t["id"])):
            if (t["file_ext"][i] or "").lower() not in IMAGE_EXT or t["is_deleted"][i] or t["is_banned"][i]:
                continue
            artists = [a for a in (t["tag_string_artist"][i] or "").split() if a not in NOT_ARTIST]
            chars = (t["tag_string_character"][i] or "").split()
            if len(artists) != 1 or len(chars) != 1:
                continue
            general = (t["tag_string_general"][i] or "").split()
            tags = set(general) | set((t["tag_string_meta"][i] or "").split())
            if "solo" not in tags or tags & LAYOUT_TAGS or tags & AI_TAGS:
                continue
            w, h = t["image_width"][i] or 0, t["image_height"][i] or 0
            if min(w, h) < 768:
                continue
            a, ch, r = artists[0], chars[0], t["rating"][i] or "g"
            if a in held_artists or ch in held_chars:
                c["held_out_skipped"] += 1
                continue
            if r in ("q", "e") and (tags & MINOR_TAGS or ch in minors):
                c["nsfw_minor_skipped"] += 1
                continue
            gender = "1boy" if "1boy" in tags else "1girl" if "1girl" in tags else "1other"
            posts.append((t["id"][i], a, ch, r, gender, w, h, t["score"][i] or 0, general))
        print(f"scan {g + 1}/{pf.metadata.num_row_groups}: {len(posts)}", file=sys.stderr, flush=True)
    c["eligible"] = len(posts)

    # artists that drew >= 2 characters (style pairs); characters drawn by >= 2 artists (char pairs)
    by_artist = collections.defaultdict(list)
    for p in posts:
        by_artist[p[1]].append(p)
    chars_of = {a: {p[2] for p in ps} for a, ps in by_artist.items()}
    artists_of = collections.defaultdict(set)
    for p in posts:
        artists_of[p[2]].add(p[1])
    good = [a for a, cs in chars_of.items() if len(cs) >= 2]
    rng.shuffle(good)
    chosen, per_char = [], collections.Counter()
    for a in good:   # per artist: best-scored posts, at most 2 per character so cs pairs exist, spread characters
        ps = sorted(by_artist[a], key=lambda p: -p[7])
        taken, per_ac = [], collections.Counter()
        for p in ps:
            if per_ac[p[2]] >= 2 or per_char[p[2]] >= args.per_character:
                continue
            taken.append(p)
            per_ac[p[2]] += 1
            if len(taken) == args.per_artist:
                break
        if len({p[2] for p in taken}) < 2:
            continue
        for p in taken:
            per_char[p[2]] += 1
        chosen += taken
        if len(chosen) >= args.images:
            break
    c["chosen"] = len(chosen)
    c["chosen_artists"] = len({p[1] for p in chosen})
    c["chosen_characters"] = len({p[2] for p in chosen})

    records = {}
    for pid, a, ch, r, gender, w, h, _, general in chosen:
        scene, look = split_tags(general, tax)
        rng.shuffle(scene)
        head = [QUALITY, RATING[r], gender, "solo"]
        sc = [anima_tag(x) for x in scene]
        records[pid] = {"id": pid, "artist": a, "character": ch, "rating": r, "gender": gender, "width": w, "height": h,
                        "prompts": {"stripped": ", ".join(head + sc),
                                    "with_artist": ", ".join([QUALITY, "@" + anima_tag(a), RATING[r], gender, "solo"] + sc),
                                    "with_character": ", ".join(head + [anima_tag(ch)] + [anima_tag(x) for x in look] + sc)},
                        "ref_caption": ", ".join([gender, "solo"] + [anima_tag(x) for x in look])}
    ids_by_artist, ids_by_char = collections.defaultdict(list), collections.defaultdict(list)
    for pid, rec in records.items():
        ids_by_artist[rec["artist"]].append(pid)
        ids_by_char[rec["character"]].append(pid)
    pairs = []
    for pid, rec in records.items():
        same_a = [q for q in ids_by_artist[rec["artist"]] if q != pid]
        for q in same_a:
            pairs.append({"mode": "cs" if records[q]["character"] == rec["character"] else "style", "target": pid, "ref": q})
        for q in ids_by_char[rec["character"]]:
            if q != pid and records[q]["artist"] != rec["artist"]:
                pairs.append({"mode": "char", "target": pid, "ref": q})
    c.update({"pairs_" + m: n for m, n in collections.Counter(p["mode"] for p in pairs).items()})
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "records.jsonl").open("w", encoding="utf-8") as f:
        for rec in records.values():
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with (args.out / "pairs.jsonl").open("w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")
    print(json.dumps(dict(c)), flush=True)
    if args.no_download:
        return 0

    c.update(download(list(records), args.danbooru / "index" / "tar-index.parquet", args.out / "images", args.workers))
    (args.out / "summary.json").write_text(json.dumps(dict(c), indent=1), encoding="utf-8")
    print(json.dumps(dict(c)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
