"""Per-artist style profile from Danbooru tags (tags only, no image inspection).

    ~/venv-data/bin/python datagen/artist_style_profile.py ~/danbooru2026 OUT.json [--min-posts 10]

An artist's style changes little across their work, so one post's style/medium tags are noisy (taggers forget
"painterly" on most posts) but the share over all of the artist's posts is a stable label. For every artist with
>= --min-posts eligible posts (single artist tag, still image, not deleted/banned; any rating, any character
count: the style is the artist's, not the post's), counts the posts carrying each tag of the taxonomy's
other:style slot plus the medium meta tags (watercolor_(medium), oil_painting_(medium), traditional_media, ...).
Writes OUT.json = {artist: {"posts": n, "share": {tag: fraction >= MIN_SHARE}, "tier": t, "texture": x}} where
  texture = max share over TEXTURE_TAGS (painterly media / visible strokes and grain),
  tier    = "flat" (flat_color/cel-like shading or no_lineart dominate), "texture" (texture >= 0.15),
            "mono" (monochrome/greyscale/sketch/lineart >= 0.5: drawings, not finished colour), else "standard".
Post-level use (build_pairs_v2.py --style-profile): a style pair's two posts should both show the artist's usual
look, so posts tagged with a style tag the artist rarely uses (share < 0.1: an off-style sketch, a monochrome
page, a guest medium) are not used as style references or targets.
"""
import argparse
import collections
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from artist_v2_sizing import IMAGE_EXT, NOT_ARTIST  # noqa: E402

MEDIUM_TAGS = {"watercolor_(medium)", "oil_painting_(medium)", "traditional_media", "acrylic_paint_(medium)",
               "gouache_(medium)", "colored_pencil_(medium)", "marker_(medium)", "pencil_(medium)", "graphite_(medium)",
               "ink_(medium)", "pastel_(medium)", "crayon_(medium)", "ballpoint_pen_(medium)", "pixel_art", "oekaki",
               "3d", "photo_(medium)", "photorealistic"}
TEXTURE_TAGS = {"painterly", "impasto", "watercolor_(medium)", "oil_painting_(medium)", "traditional_media",
                "acrylic_paint_(medium)", "gouache_(medium)", "colored_pencil_(medium)", "pastel_(medium)",
                "faux_traditional_media", "film_grain", "blending", "realistic", "sketch"}
FLAT_TAGS = {"flat_color", "anime_coloring", "no_lineart", "cel_shading"}
MONO_TAGS = {"monochrome", "greyscale", "sketch", "lineart", "spot_color"}
MIN_SHARE = 0.02


def style_tags(tax_path: Path) -> set[str]:
    t = pq.read_table(tax_path, columns=["tag", "slot"]).to_pylist()
    return {r["tag"] for r in t if r["slot"] == "other:style"} | MEDIUM_TAGS


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("danbooru", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--min-posts", type=int, default=10)
    args = ap.parse_args()
    vocab = style_tags(Path.home() / "hf" / "taxonomy" / "tags.parquet")
    pf = pq.ParquetFile(args.danbooru / "metadata" / "posts-snapshot.parquet")
    cols = ["tag_string_general", "tag_string_meta", "tag_string_artist", "file_ext", "is_deleted", "is_banned"]
    posts, counts = collections.Counter(), collections.defaultdict(collections.Counter)
    for g in range(pf.metadata.num_row_groups):
        t = pf.read_row_group(g, columns=cols).to_pydict()
        for i in range(len(t["file_ext"])):
            if (t["file_ext"][i] or "").lower() not in IMAGE_EXT or t["is_deleted"][i] or t["is_banned"][i]:
                continue
            artists = [a for a in (t["tag_string_artist"][i] or "").split() if a not in NOT_ARTIST]
            if len(artists) != 1:
                continue
            a = artists[0]
            posts[a] += 1
            tags = set((t["tag_string_general"][i] or "").split()) | set((t["tag_string_meta"][i] or "").split())
            counts[a].update(tags & vocab)
        print(f"scan {g + 1}/{pf.metadata.num_row_groups}: {len(posts)} artists", file=sys.stderr, flush=True)
    out, tiers = {}, collections.Counter()
    for a, n in posts.items():
        if n < args.min_posts:
            continue
        share = {k: round(v / n, 3) for k, v in counts[a].most_common() if v / n >= MIN_SHARE}
        texture = max([share.get(k, 0.0) for k in TEXTURE_TAGS] + [0.0])
        if sum(share.get(k, 0.0) for k in ("monochrome", "greyscale")) >= 0.5 or share.get("sketch", 0) + share.get("lineart", 0) >= 0.5:
            tier = "mono"
        elif texture >= 0.15:
            tier = "texture"
        elif max([share.get(k, 0.0) for k in FLAT_TAGS] + [0.0]) >= 0.3:
            tier = "flat"
        else:
            tier = "standard"
        tiers[tier] += 1
        out[a] = {"posts": n, "share": share, "tier": tier, "texture": texture}
    args.out.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"artists": len(out), "tiers": dict(tiers), "vocab": len(vocab)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
