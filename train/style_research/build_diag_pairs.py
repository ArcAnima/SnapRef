"""Pairs for diag_loss.py from the v2 pool, artists outside the Danbooru pilot (pilot2 never trained on them).

    python train/style_research/build_diag_pairs.py --records ~/style_agent/v2_records.jsonl \
        --pilot ~/pilot_data/records.jsonl --images ~/v2_data/hf/images --latents ~/v2_data/latents --out PAIRS.json

Per pair: target T (artist a, character X), right reference R (a, another character), wrong reference W (another
artist, another character), cs reference C (a, X, another post) when one exists. SFW (g/s) only, no parent/child
relations, at most one pair per artist, only posts whose latent shard is on this machine. Prompts follow
build_pairs_v2.py's style prompt: quality, rating, gender, solo, character, look, scene; "prompt_art" is the same
with @artist after the quality tags (Anima's own text route to the artist's style).
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "datagen"))
from catalog import anima_tag  # noqa: E402

QUALITY = "masterpiece, best quality, score_7, newest, highres, absurdres"
RATING = {"g": "safe", "s": "sensitive"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=Path, required=True)
    ap.add_argument("--pilot", type=Path, required=True)
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--latents", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=240)
    ap.add_argument("--seed", type=int, default=20261030)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    pilot_artists = {json.loads(l)["artist"] for l in args.pilot.read_text(encoding="utf-8").splitlines() if l.strip()}
    where = {}
    for js in sorted(args.images.glob("shard_*.json")):
        lat = args.latents / f"latents_{js.stem[6:]}.safetensors"
        if lat.exists():
            for pid in json.loads(js.read_text()):
                where[int(pid)] = str(lat)
    recs = {}
    for line in args.records.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r["id"] in where and r["rating"] in RATING and r["artist"] not in pilot_artists:
            recs[r["id"]] = r
    by_ac = collections.defaultdict(list)
    for pid, r in recs.items():
        by_ac[(r["artist"], r["character"])].append(pid)
    chars_of = collections.defaultdict(set)
    for a, c in by_ac:
        chars_of[a].add(c)
    artists = sorted(a for a, cs in chars_of.items() if len(cs) >= 2)
    rng.shuffle(artists)

    def related(x, y):
        px, py = recs[x].get("parent_id"), recs[y].get("parent_id")
        return px == y or py == x or (px is not None and px == py)

    def prompt(t, with_artist):
        r = recs[t]
        head = [QUALITY] + (["@" + anima_tag(r["artist"])] if with_artist else []) + [RATING[r["rating"]], r["gender"], "solo"]
        sc = list(r["scene"])
        rng.shuffle(sc)
        return ", ".join(head + [anima_tag(r["character"])] + [anima_tag(x) for x in r["look"]] + [anima_tag(x) for x in sc])

    pairs = []
    for a in artists:
        x, y = rng.sample(sorted(chars_of[a]), 2)
        t = rng.choice(by_ac[(a, x)])
        r = rng.choice(by_ac[(a, y)])
        if related(t, r):
            continue
        cs = [p for p in by_ac[(a, x)] if p != t and not related(t, p)]
        c = rng.choice(cs) if cs else None
        while True:
            w = rng.choice(list(recs))
            if recs[w]["artist"] != a and recs[w]["character"] != x:
                break
        pairs.append({"artist": a, "t": t, "r": r, "w": w, "c": c, "t_file": where[t], "r_file": where[r],
                      "w_file": where[w], "c_file": where[c] if c else None,
                      "prompt": prompt(t, False), "prompt_art": prompt(t, True), "artist_characters": len(chars_of[a])})
        if len(pairs) >= args.n:
            break
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(pairs, ensure_ascii=False, indent=0), encoding="utf-8")
    print(json.dumps({"pairs": len(pairs), "with_cs": sum(p["c"] is not None for p in pairs), "records_usable": len(recs),
                      "artists_usable": len(artists)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
