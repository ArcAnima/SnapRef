"""Artist statistics for style-reference pairs (metadata only, Danbooru2026).

    ~/venv-data/bin/python datagen/artist_v2_sizing.py ~/danbooru2026/metadata/posts-snapshot.parquet OUT_DIR

Gelbooru2026 metadata has no separate artist field (artist tags are mixed into `tags`), so it is not used here.
A usable post for style pairs: still image, short side >= 512 px, exactly one artist tag, solo, exactly one
character tag (or no character tag: original characters count as their own "character" per post), no layout
tags, not ai-generated / ai-assisted, not deleted/banned, NSFW posts without minor tags (decision 10/11, tags only).
Per artist: posts, distinct characters, year span. A style pair = two posts of one artist with different
characters, so an artist is useful with >= 2 characters.
Writes OUT_DIR/artist_stats.json (counts) and OUT_DIR/artists.parquet (per-artist rows) and picks a held-out
artist set (hash of the name, among artists with >= 40 posts and >= 8 characters) into OUT_DIR/heldout_artists.json.
"""
import collections
import hashlib
import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

MINOR_TAGS = {"loli", "shota", "child", "toddler", "baby", "kindergarten_uniform", "aged_down", "female_child",
              "male_child", "elementary_school", "randoseru", "child_on_child", "onee-shota", "oppai_loli"}
LAYOUT_TAGS = {"comic", "4koma", "2koma", "multiple_views", "split_screen", "character_sheet", "reference_sheet",
               "text-only_page", "variations"}
AI_TAGS = {"ai-generated", "ai-assisted", "ai_generated", "ai_assisted"}
IMAGE_EXT = {"jpg", "jpeg", "png", "webp", "avif", "bmp"}
NOT_ARTIST = {"banned_artist", "unknown_artist", "anonymous", "artist_request", "third-party_edit", "third-party_source"}


def add(stats, artist, char, year, nsfw):
    s = stats[artist]
    s["posts"] += 1
    s["nsfw"] += nsfw
    s["chars"].add(char)
    s["years"].add(year)


def danbooru(path, stats, c):
    pf = pq.ParquetFile(path)
    cols = ["id", "rating", "image_width", "image_height", "tag_string_general", "tag_string_character",
            "tag_string_meta", "tag_string_artist", "file_ext", "is_deleted", "is_banned", "created_at"]
    for g in range(pf.metadata.num_row_groups):
        t = pf.read_row_group(g, columns=cols).to_pydict()
        for i in range(len(t["id"])):
            c["d_posts"] += 1
            if (t["file_ext"][i] or "").lower() not in IMAGE_EXT or t["is_deleted"][i] or t["is_banned"][i]:
                continue
            artists = [a for a in (t["tag_string_artist"][i] or "").split() if a not in NOT_ARTIST]
            if len(artists) != 1:
                continue
            tags = set((t["tag_string_general"][i] or "").split()) | set((t["tag_string_meta"][i] or "").split())
            chars = (t["tag_string_character"][i] or "").split()
            if "solo" not in tags or len(chars) > 1 or tags & LAYOUT_TAGS or tags & AI_TAGS:
                continue
            if min(t["image_width"][i] or 0, t["image_height"][i] or 0) < 512:
                continue
            nsfw = (t["rating"][i] or "") in ("q", "e")
            if nsfw and tags & MINOR_TAGS:
                continue
            char = chars[0] if chars else f"oc:{t['id'][i]}"
            year = t["created_at"][i].year if t["created_at"][i] else 0
            add(stats, artists[0], char, year, nsfw)
            c["d_usable"] += 1
        print(f"danbooru {g + 1}/{pf.metadata.num_row_groups}", file=sys.stderr, flush=True)


def main(dan, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stats = collections.defaultdict(lambda: {"posts": 0, "nsfw": 0, "chars": set(), "years": set()})
    c = collections.Counter()
    danbooru(dan, stats, c)
    rows = {"artist": [], "posts": [], "nsfw": [], "characters": [], "year_min": [], "year_max": []}
    for a, s in stats.items():
        ys = sorted(y for y in s["years"] if y)
        for k, v in (("artist", a), ("posts", s["posts"]), ("nsfw", s["nsfw"]), ("characters", len(s["chars"])),
                     ("year_min", ys[0] if ys else 0), ("year_max", ys[-1] if ys else 0)):
            rows[k].append(v)
    pq.write_table(pa.table(rows), out / "artists.parquet")
    arts = list(zip(rows["artist"], rows["posts"], rows["characters"]))

    def tier(min_posts, min_chars, cap=None):
        sel = [(a, p, ch) for a, p, ch in arts if p >= min_posts and ch >= min_chars]
        return {"artists": len(sel), "posts": sum(p for _, p, _ in sel),
                "posts_capped": sum(min(p, cap) for _, p, _ in sel) if cap else None}

    eligible = sorted(a for a, p, ch in arts if p >= 40 and ch >= 8)
    held = sorted(eligible, key=lambda a: hashlib.sha256(a.encode()).hexdigest())[:200]
    (out / "heldout_artists.json").write_text(json.dumps({"rule": "sha256(name) smallest 200 among posts>=40, characters>=8",
                                                          "artists": held}, indent=1), encoding="utf-8")
    summary = {"counts": dict(c), "artists_total": len(arts),
               "tiers": {f"posts>={p},chars>={ch}": tier(p, ch, cap) for p, ch, cap in
                         ((10, 2, None), (20, 3, 100), (40, 5, 100), (40, 8, 100), (100, 10, 200), (100, 10, 100))},
               "heldout_artists": len(held),
               "heldout_posts": sum(p for a, p, _ in arts if a in set(held)),
               "top10": sorted(arts, key=lambda x: -x[1])[:10]}
    (out / "artist_stats.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:3])
