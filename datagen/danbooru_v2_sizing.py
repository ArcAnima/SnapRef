"""Danbooru2026 counts for sizing the v2 dataset (metadata only).

    ~/venv-data/bin/python danbooru_v2_sizing.py ~/danbooru2026/metadata/posts-snapshot.parquet \
        ~/gelbooru2026/characters_used.json ~/gelbooru2026/superseded/minor_characters.txt > sizing.json

Same rules as gelbooru_v2_sizing.py (decision 10): solo + exactly one character tag; SFW as is; NSFW without
minor tags; layout posts and short side < 512 px not counted. Danbooru2026 holds still images only (no
animations, videos), so posts with other file types are skipped; deleted/banned posts are counted separately
because the image tars may or may not contain them.
Reports our 2,367-character pool and, for real-to-real pairs, every character with enough images.
"""
import collections
import json
import sys

import pyarrow.parquet as pq

MINOR_TAGS = {"loli", "shota", "child", "toddler", "baby", "kindergarten_uniform", "aged_down", "female_child",
              "male_child", "elementary_school", "randoseru", "child_on_child", "onee-shota", "oppai_loli"}
LAYOUT_TAGS = {"comic", "4koma", "2koma", "multiple_views", "split_screen", "character_sheet", "reference_sheet",
               "text-only_page", "variations"}
IMAGE_EXT = {"jpg", "jpeg", "png", "webp", "avif", "bmp"}
CAPS = (50, 60, 100, 200)


def norm(name: str) -> str:
    return name.replace("\\(", "(").replace("\\)", ")").replace(" ", "_").lower()


def main(posts: str, used_json: str, minors_txt: str) -> int:
    used = json.load(open(used_json, encoding="utf-8"))
    pool = {norm(c) for c in used["train_characters"]}
    held = {norm(c) for c in used["held_out"]}
    minors = {norm(l.strip()) for l in open(minors_txt, encoding="utf-8") if l.strip()}
    pf = pq.ParquetFile(posts)
    cols = ["rating", "image_width", "image_height", "tag_string_general", "tag_string_character", "tag_string_meta",
            "file_ext", "is_deleted", "is_banned"]
    c = collections.Counter()
    per = collections.defaultdict(collections.Counter)   # character -> sfw / nsfw / nsfw_minor_tag / sfw768 / nsfw768
    held_imgs = collections.Counter()
    for g in range(pf.metadata.num_row_groups):
        t = pf.read_row_group(g, columns=cols).to_pydict()
        for i in range(len(t["rating"])):
            c["posts"] += 1
            if (t["file_ext"][i] or "").lower() not in IMAGE_EXT:
                c["not_still_image"] += 1
                continue
            gone = bool(t["is_deleted"][i]) or bool(t["is_banned"][i])
            chars = (t["tag_string_character"][i] or "").split()
            tags = set((t["tag_string_general"][i] or "").split()) | set((t["tag_string_meta"][i] or "").split())
            if "solo" not in tags or len(chars) != 1:
                continue
            c["solo_one_char"] += 1
            if tags & LAYOUT_TAGS:
                c["layout"] += 1
                continue
            side = min(t["image_width"][i] or 0, t["image_height"][i] or 0)
            if side < 512:
                c["lt512"] += 1
                continue
            ch = chars[0]
            nsfw = (t["rating"][i] or "") in ("q", "e")
            if gone:
                c["deleted_or_banned"] += 1
                per[ch]["gone"] += 1
                continue
            if ch in held:
                if not nsfw:
                    held_imgs[ch] += 1
                continue
            if nsfw and tags & MINOR_TAGS:
                per[ch]["nsfw_minor_tag"] += 1
                continue
            k = "nsfw" if nsfw else "sfw"
            per[ch][k] += 1
            if side >= 768:
                per[ch][k + "768"] += 1
        print(f"row group {g + 1}/{pf.metadata.num_row_groups}", file=sys.stderr, flush=True)

    def usable(ch, v):   # minor characters (old over-strict list as an upper bound): SFW only
        return v["sfw"] + (0 if ch in minors else v["nsfw"])

    def summary(chars):
        vals = {ch: usable(ch, per[ch]) for ch in chars if ch in per}
        counts = sorted(vals.values())
        return {"characters_with_images": sum(n > 0 for n in counts),
                "images": sum(counts), "sfw": sum(per[ch]["sfw"] for ch in vals),
                "nsfw": sum(0 if ch in minors else per[ch]["nsfw"] for ch in vals),
                "ge768": sum(per[ch]["sfw768"] + (0 if ch in minors else per[ch]["nsfw768"]) for ch in vals),
                "ge5": sum(n >= 5 for n in counts), "ge20": sum(n >= 20 for n in counts),
                "median": counts[len(counts) // 2] if counts else 0,
                "capped": {cap: sum(min(n, cap) for n in counts) for cap in CAPS}}

    others = [ch for ch in per if ch not in pool and ch not in held]
    others_ge20 = [ch for ch in others if usable(ch, per[ch]) >= 20]
    out = {
        "counts": dict(c),
        "pool": summary(pool),
        "outside_pool_all": summary(others),
        "outside_pool_ge20": summary(others_ge20),
        "nsfw_dropped_minor_tag": sum(v["nsfw_minor_tag"] for v in per.values()),
        "deleted_or_banned_solo": sum(v["gone"] for v in per.values()),
        "held_out_sfw": {"characters": len(held_imgs), "images": sum(held_imgs.values())},
        "top10_pool": sorted(((ch, usable(ch, per[ch])) for ch in pool if ch in per), key=lambda x: -x[1])[:10],
    }
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:4]))
