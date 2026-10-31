"""Pilot v2 pairs: per-pair target prompts from the original Danbooru tags (tags only, no image inspection).

    ~/venv-data/bin/python datagen/build_pairs_v2.py ~/danbooru2026 ~/pilot_data OUT_DIR [--max-pairs 400000]

Fixes over build_danbooru_pilot.py's pairs, whose prompts dropped every look tag although Danbooru references and
targets usually wear different clothes:
  * target prompt per (target, reference) pair = quality, rating, gender, solo, [mode part], the target's look tags
    (appearance + clothing) that the REFERENCE does not have, and the target's scene tags. Look tags both posts
    share are left out (the reference must supply them);
      cs     mode part: -                      (same artist, same character)
      char   mode part: @artist of the target  (style from text, identity from the reference)
      style  mode part: character of the target (identity from text; the target's full look is "not in the
             reference" because the reference is another character, so it is all kept)
  * pairs dropped: parent/child variants and posts sharing a parent (near copies); cs/char pairs whose clothing
    tag Jaccard < MIN_CLOTH_J (outfits too different for the tags to describe); style pairs posted more than
    MAX_YEAR_GAP years apart (an artist's style drifts);
  * at most --per-target pairs per target and mode, so a few dense characters do not dominate.
Reads OUT_DIR's sibling pilot_data/records.jsonl (the 55,812 downloaded posts). Writes OUT_DIR/pairs.jsonl
({mode, target, ref, prompt}) and OUT_DIR/summary.json.
"""
import argparse
import collections
import json
import random
import sys
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_danbooru_pilot import DROP_TAGS, LOOK_CATS, QUALITY, RATING, SCENE_CATS, SCENE_SLOTS, taxonomy  # noqa: E402
from artist_v2_sizing import MINOR_TAGS  # noqa: E402
from catalog import anima_tag  # noqa: E402

MIN_CLOTH_J = 0.3
MAX_YEAR_GAP = 3


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("danbooru", type=Path)
    ap.add_argument("data", type=Path, help="pilot_data with records.jsonl and images/")
    ap.add_argument("out", type=Path)
    ap.add_argument("--per-target", type=int, default=4)
    ap.add_argument("--max-pairs", type=int, default=400000)
    ap.add_argument("--seed", type=int, default=20261031)
    ap.add_argument("--have", type=Path, help="text file of downloaded post ids (when images/ is on another machine)")
    args = ap.parse_args()
    have = {int(x) for x in args.have.read_text().split()} if args.have else None
    rng = random.Random(args.seed)
    recs = {}
    for line in (args.data / "records.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if (r["id"] in have) if have is not None else (args.data / "images" / f"{r['id']}.webp").exists():
            recs[r["id"]] = r
    tax = taxonomy(Path.home() / "hf" / "taxonomy" / "tags.parquet")
    t = pq.read_table(args.danbooru / "metadata" / "posts-snapshot.parquet",
                      columns=["id", "tag_string_general", "parent_id", "created_at"],
                      filters=[("id", "in", list(recs))]).to_pylist()
    look, scene, cloth, parent, year = {}, {}, {}, {}, {}
    for row in t:
        pid = row["id"]
        tags = [x for x in (row["tag_string_general"] or "").split() if x not in DROP_TAGS and x not in MINOR_TAGS]
        lk, sc = [], []
        for x in tags:
            cat, slot = tax.get(x, (None, None))
            if cat in LOOK_CATS and slot != "appearance:demographics":
                lk.append(x)
            elif cat in SCENE_CATS or slot in SCENE_SLOTS:
                sc.append(x)
        look[pid], scene[pid] = lk, sc
        cloth[pid] = {x for x in lk if tax.get(x, (None,))[0] == "clothing"}
        parent[pid] = row["parent_id"]
        year[pid] = row["created_at"].year if row["created_at"] else 0
    ids = [i for i in recs if i in look]

    def related(a, b):
        return parent.get(a) == b or parent.get(b) == a or (parent.get(a) and parent.get(a) == parent.get(b))

    def cloth_j(a, b):
        u = cloth[a] | cloth[b]
        return len(cloth[a] & cloth[b]) / len(u) if u else 1.0

    def prompt(target, ref, mode):
        r = recs[target]
        head = [QUALITY, RATING[r["rating"]], r["gender"], "solo"]
        if mode == "char":
            head.append("@" + anima_tag(r["artist"]))
        if mode == "style":
            head.append(anima_tag(r["character"]))
        # style: the reference is another character, so none of its look may be taken; the target's whole look
        # goes into the prompt. cs/char: only what the reference does not show.
        ref_look = set() if mode == "style" else set(look[ref])
        diff = [x for x in look[target] if x not in ref_look]
        sc = list(scene[target])
        rng.shuffle(sc)
        return ", ".join(head + [anima_tag(x) for x in diff] + [anima_tag(x) for x in sc])

    by_artist, by_char = collections.defaultdict(list), collections.defaultdict(list)
    for i in ids:
        by_artist[recs[i]["artist"]].append(i)
        by_char[recs[i]["character"]].append(i)
    stats = collections.Counter()
    pairs = []
    for tgt in ids:
        a, ch = recs[tgt]["artist"], recs[tgt]["character"]
        cands = {"cs": [], "style": [], "char": []}
        for ref in by_artist[a]:
            if ref == tgt:
                continue
            if related(tgt, ref):
                stats["drop_variant"] += 1
                continue
            if recs[ref]["character"] == ch:
                if cloth_j(tgt, ref) < MIN_CLOTH_J:
                    stats["drop_cs_cloth"] += 1
                    continue
                cands["cs"].append(ref)
            else:
                if abs(year[tgt] - year[ref]) > MAX_YEAR_GAP:
                    stats["drop_style_years"] += 1
                    continue
                cands["style"].append(ref)
        for ref in by_char[ch]:
            if ref == tgt or recs[ref]["artist"] == a:
                continue
            if cloth_j(tgt, ref) < MIN_CLOTH_J:
                stats["drop_char_cloth"] += 1
                continue
            cands["char"].append(ref)
        for mode, refs in cands.items():
            rng.shuffle(refs)
            for ref in refs[: args.per_target]:
                pairs.append({"mode": mode, "target": tgt, "ref": ref, "prompt": prompt(tgt, ref, mode)})
    rng.shuffle(pairs)
    pairs = pairs[: args.max_pairs]
    by_mode = collections.Counter(p["mode"] for p in pairs)
    stats.update({"pairs_" + m: n for m, n in by_mode.items()})
    stats["records"] = len(ids)
    stats["records_in_pairs"] = len({p["target"] for p in pairs} | {p["ref"] for p in pairs})
    diffs = collections.defaultdict(list)
    for p in pairs:
        diffs[p["mode"]].append(p["prompt"].count(","))
    stats.update({f"prompt_tags_median_{m}": sorted(v)[len(v) // 2] for m, v in diffs.items()})
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "pairs.jsonl").open("w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    (args.out / "summary.json").write_text(json.dumps(dict(stats), indent=1), encoding="utf-8")
    print(json.dumps(dict(stats)))
    for m in ("cs", "char", "style"):
        ex = next((p for p in pairs if p["mode"] == m), None)
        if ex:
            print(m, ex["target"], "<-", ex["ref"], "|", ex["prompt"][:300])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
