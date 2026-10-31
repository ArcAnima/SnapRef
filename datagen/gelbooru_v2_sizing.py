"""Gelbooru2026 counts for sizing the v2 dataset (metadata only, nothing downloaded or written except a JSON).

    ~/venv-data/bin/python gelbooru_v2_sizing.py ~/gelbooru2026 > sizing.json

Rules (user decision 2026-10-29, second version):
  * solo, exactly one character tag, character in our pool; the 32 held-out characters are counted for eval only;
  * SFW (general, sensitive): as is;
  * NSFW (questionable, explicit): without minor tags; minor characters SFW only. The minor-character list is not
    built yet, so NSFW is reported for all characters and for the old over-strict 435-character list as a bound;
  * layout (comic, 4koma, multiple views, sheets) and size bands are counted, not dropped.
"""
import collections
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

MINOR_TAGS = {"loli", "shota", "child", "toddler", "baby", "kindergarten_uniform", "aged_down", "female_child",
              "male_child", "elementary_school", "randoseru", "child_on_child", "onee-shota", "oppai_loli"}
LAYOUT_TAGS = {"comic", "4koma", "2koma", "multiple_views", "split_screen", "character_sheet", "reference_sheet",
               "text-only_page", "variations"}
SFW = {"general", "sensitive", "safe"}
CAPS = (20, 50, 100, 200, 400)


def norm(name: str) -> str:
    return name.replace("\\(", "(").replace("\\)", ")").replace(" ", "_").lower()


def main(root: str) -> int:
    root = Path(root)
    used = json.loads((root / "characters_used.json").read_text(encoding="utf-8"))
    train = {norm(c) for c in used["train_characters"]}
    held = {norm(c) for c in used["held_out"]}
    old_minors = {norm(l.strip()) for l in (root / "superseded" / "minor_characters.txt").read_text(encoding="utf-8").splitlines() if l.strip()}
    pf = pq.ParquetFile(root / "metadata" / "posts.parquet")
    c = collections.Counter()
    per_char = collections.defaultdict(collections.Counter)   # character -> {sfw, nsfw, nsfw_minor_tag}
    held_imgs = collections.Counter()
    for g in range(pf.metadata.num_row_groups):
        for r in pf.read_row_group(g, columns=["extra", "width", "height"]).to_pylist():
            e = r["extra"] or {}
            tags = set(e.get("tags") or [])
            chars = [norm(x) for x in (e.get("tags_character") or [])]
            if "solo" not in tags or len(chars) != 1:
                continue
            ch = chars[0]
            rating = (e.get("rating") or "").lower()
            nsfw = rating not in SFW
            if tags & LAYOUT_TAGS:
                c["layout_" + ("held" if ch in held else "pool" if ch in train else "outside")] += 1
                continue
            side = min(r["width"] or 0, r["height"] or 0)
            band = "lt512" if side < 512 else "512_768" if side < 768 else "ge768"
            if ch in held:
                if not nsfw and side >= 512:
                    held_imgs[ch] += 1
                continue
            if ch not in train:
                c["outside_pool_" + ("nsfw" if nsfw else "sfw")] += 1
                continue
            c[f"pool_{band}"] += 1
            if side < 512:
                continue
            if not nsfw:
                per_char[ch]["sfw"] += 1
            elif tags & MINOR_TAGS:
                per_char[ch]["nsfw_minor_tag"] += 1
            else:
                per_char[ch]["nsfw"] += 1
    def total(key, chars=None):
        return sum(v[key] for ch, v in per_char.items() if chars is None or ch in chars)
    adult_like = set(per_char) - old_minors
    sfw_counts = sorted(v["sfw"] for v in per_char.values())
    all_counts = {ch: v["sfw"] + v["nsfw"] for ch, v in per_char.items()}
    capped = {}
    for cap in CAPS:
        capped[cap] = {
            "sfw_only": sum(min(v["sfw"], cap) for v in per_char.values()),
            "sfw+nsfw_all_chars": sum(min(v["sfw"] + v["nsfw"], cap) for v in per_char.values()),
            "sfw+nsfw_old_minor_list_sfw_only": sum(min(v["sfw"] + (0 if ch in old_minors else v["nsfw"]), cap) for ch, v in per_char.items()),
        }
    out = {
        "counts": dict(c),
        "pool_side_ge512": {"sfw": total("sfw"), "nsfw": total("nsfw"), "nsfw_dropped_minor_tag": total("nsfw_minor_tag"),
                            "nsfw_of_old_minor_list": total("nsfw", old_minors), "old_minor_list_size": len(old_minors)},
        "characters": {"with_any": len(per_char), "pool": len(train),
                       "sfw_ge5": sum(v["sfw"] >= 5 for v in per_char.values()),
                       "sfw_ge20": sum(v["sfw"] >= 20 for v in per_char.values()),
                       "all_ge20": sum(n >= 20 for n in all_counts.values()),
                       "sfw_median": sfw_counts[len(sfw_counts) // 2],
                       "all_median": sorted(all_counts.values())[len(all_counts) // 2],
                       "top10_all": sorted(all_counts.items(), key=lambda x: -x[1])[:10]},
        "capped_per_character": capped,
        "held_out_sfw_ge512": {"characters_with_images": len(held_imgs), "images": sum(held_imgs.values()),
                               "with_ge5": sum(v >= 5 for v in held_imgs.values()), "min": min(held_imgs.values()) if held_imgs else 0},
    }
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
