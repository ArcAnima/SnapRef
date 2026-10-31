"""Select the SnapRef v2 image pool from Danbooru2026 (tags only, no image inspection).

    ~/venv-data/bin/python datagen/select_v2_images.py ~/danbooru2026 ~/style_data OUT [--target 250000]

Post filter: as build_danbooru_pilot.py (decisions 10/11): still image, not deleted/banned, short side >= 768 px,
exactly one artist tag, solo, exactly one character tag, no layout tags, not ai-generated; NSFW (q/e) without
minor tags, minor characters (old 435 list, an over-strict bound) SFW only; held-out characters (32) and
held-out artists (200) never used.
Selection, per artist (artists with >= 2 kept characters, shuffled):
  * characters the artist drew several times first (cs pairs need two posts of one artist and one character),
    at most --per-artist-char posts per (artist, character), chosen greedily for overlapping outfits
    (cs/char pairs whose clothing tags barely overlap are dropped later by build_pairs_v2.py);
  * at most --per-artist posts per artist, --per-char per character (--per-char-pool for our 2,367 pool
    characters, which the identity eval covers), until --target posts.
Writes OUT/records.jsonl (id, artist, character, rating, gender, size, year, parent_id, look and scene tags in
Danbooru spelling) and OUT/ids.txt for deploy/modal_prep.py, plus OUT/summary.json.
"""
import argparse
import collections
import json
import random
import sys
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from artist_v2_sizing import AI_TAGS, IMAGE_EXT, LAYOUT_TAGS, MINOR_TAGS, NOT_ARTIST  # noqa: E402
from build_danbooru_pilot import norm, split_tags, taxonomy  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("danbooru", type=Path)
    ap.add_argument("style_data", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--target", type=int, default=250000)
    ap.add_argument("--per-artist", type=int, default=30)
    ap.add_argument("--per-artist-char", type=int, default=4)
    ap.add_argument("--per-char", type=int, default=100)
    ap.add_argument("--per-char-pool", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20261030)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    home = Path.home()
    used = json.loads((home / "gelbooru2026" / "characters_used.json").read_text(encoding="utf-8"))
    held_chars = {norm(c) for c in used["held_out"]}
    pool = {norm(c) for c in used["train_characters"]}
    minors = {norm(l.strip()) for l in (home / "gelbooru2026" / "minor_characters.txt").read_text(encoding="utf-8").splitlines() if l.strip()}
    held_artists = set(json.loads((args.style_data / "heldout_artists.json").read_text(encoding="utf-8"))["artists"])
    tax = taxonomy(home / "hf" / "taxonomy" / "tags.parquet")
    pf = pq.ParquetFile(args.danbooru / "metadata" / "posts-snapshot.parquet")
    cols = ["id", "rating", "image_width", "image_height", "tag_string_general", "tag_string_character", "tag_string_meta",
            "tag_string_artist", "file_ext", "is_deleted", "is_banned", "score", "created_at", "parent_id"]
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
            general = t["tag_string_general"][i] or ""
            tags = set(general.split()) | set((t["tag_string_meta"][i] or "").split())
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
            year = t["created_at"][i].year if t["created_at"][i] else 0
            posts.append((t["id"][i], a, ch, r, gender, w, h, t["score"][i] or 0, year, t["parent_id"][i], general))
        print(f"scan {g + 1}/{pf.metadata.num_row_groups}: {len(posts)}", file=sys.stderr, flush=True)
    c["eligible"] = len(posts)

    by_artist = collections.defaultdict(lambda: collections.defaultdict(list))
    for p in posts:
        by_artist[p[1]][p[2]].append(p)
    artists = [a for a, chars in by_artist.items() if len(chars) >= 2]
    rng.shuffle(artists)
    cloth_cache = {}

    def cloth(p):
        if p[0] not in cloth_cache:
            cloth_cache[p[0]] = {x for x in p[10].split() if tax.get(x, (None,))[0] == "clothing"}
        return cloth_cache[p[0]]

    def pick_group(ps, k):
        """k posts of one (artist, character): the best-scored, then the ones whose outfits overlap the picked most."""
        ps = sorted(ps, key=lambda p: -p[7])
        if len(ps) <= k:
            return ps
        picked, rest = [ps[0]], ps[1:]
        while len(picked) < k and rest:
            union = set().union(*(cloth(p) for p in picked))
            best = max(rest, key=lambda p: (len(cloth(p) & union) / max(1, len(cloth(p) | union)), p[7]))
            picked.append(best)
            rest.remove(best)
        return picked

    per_char, chosen = collections.Counter(), []
    for a in artists:
        chars = by_artist[a]
        order = sorted(chars, key=lambda ch: (-min(len(chars[ch]), args.per_artist_char), -max(p[7] for p in chars[ch])))
        taken = []
        for ch in order:
            cap = args.per_char_pool if ch in pool else args.per_char
            room = min(args.per_artist_char, cap - per_char[ch], args.per_artist - len(taken))
            if room <= 0:
                continue
            taken += pick_group(chars[ch], room)
            if len(taken) >= args.per_artist:
                break
        if len({p[2] for p in taken}) < 2:
            continue
        for p in taken:
            per_char[p[2]] += 1
        chosen += taken
        if len(chosen) >= args.target:
            break
    groups = collections.Counter((p[1], p[2]) for p in chosen)
    c.update({"chosen": len(chosen), "artists": len({p[1] for p in chosen}), "characters": len({p[2] for p in chosen}),
              "pool_characters": len({p[2] for p in chosen} & pool), "artist_char_groups_ge2": sum(n >= 2 for n in groups.values()),
              "cs_ordered_pairs_upper_bound": sum(n * (n - 1) for n in groups.values()),
              "nsfw": sum(p[3] in ("q", "e") for p in chosen)})
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "records.jsonl").open("w", encoding="utf-8") as f:
        for p in chosen:
            scene, look = split_tags(p[10].split(), tax)
            f.write(json.dumps({"id": p[0], "artist": p[1], "character": p[2], "rating": p[3], "gender": p[4],
                                "width": p[5], "height": p[6], "score": p[7], "year": p[8], "parent_id": p[9],
                                "look": look, "scene": scene}, ensure_ascii=False) + "\n")
    (args.out / "ids.txt").write_text("\n".join(str(p[0]) for p in chosen) + "\n", encoding="utf-8")
    (args.out / "summary.json").write_text(json.dumps(dict(c), indent=1), encoding="utf-8")
    print(json.dumps(dict(c)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
