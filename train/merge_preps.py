"""Merge a scenes3x4 prep (prepare_asym.py output, generated data) into a Danbooru pilot prep (prepare_danbooru.py),
taking only a share of the generated groups, so one train_asym.py run samples both.

    python train/merge_preps.py --danbooru ~/train/pilot_prep_v2 --synthetic ~/train/asym_prep --out ~/train/pilot_mix_prep \
        --synthetic-groups 3000

  * generated records keep their prompts (stripped/partial/minimal/with_artist/ref_caption, re-indexed into one
    prompt table) and become pairs of index["pairs"]: cs = two images of one group (same character, same style),
    char = images of two groups of one character (target prompt with_artist), sampling modes "cs_syn" / "char_syn" so
    their share is set separately in adapter.modes and their prompt column in index["prompt_of_mode"]; they train the
    "cs" / "char" mode embeddings (index["embed_of_mode"]), the ones inference uses;
  * held-out characters of the generated prep stay out (they are the identity eval);
  * latents are not copied: index["files"]["latents"] lists both preps' shard files with relative paths.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import random
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--danbooru", type=Path, required=True)
    ap.add_argument("--synthetic", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--synthetic-groups", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    d = json.loads((args.danbooru / "index.json").read_text(encoding="utf-8"))
    s = json.loads((args.synthetic / "index.json").read_text(encoding="utf-8"))
    dctx, sctx = load_file(str(args.danbooru / "contexts.safetensors")), load_file(str(args.synthetic / "contexts.safetensors"))
    d_off, s_off = dctx["offsets"].tolist(), sctx["offsets"].tolist()
    n_d = len(d_off) - 1
    held = set(s["held_out_characters"])
    groups = [g for g, v in s["groups"].items() if v["split"] == "train" and v["character"] not in held]
    rng.shuffle(groups)
    groups = groups[: args.synthetic_groups]
    by_char = {}
    for g in groups:
        by_char.setdefault(s["groups"][g]["character"], []).append(g)
    used_prompts = set()
    records = dict(d["records"])
    pairs = {k: list(v) for k, v in d["pairs"].items()}
    pairs["cs_syn"], pairs["char_syn"] = [], []
    for g in groups:
        ids = s["groups"][g]["record_ids"]
        for rid in ids:
            r = s["records"][rid]
            records[rid] = {"character": r["character"], "artist": "|".join(s["groups"][g].get("artists", [])),
                            "width": r["width"], "height": r["height"], "focus": r.get("focus"), "framing": r.get("framing"),
                            "prompts": {k: n_d + v for k, v in r["prompts"].items()}}
            used_prompts.update(r["prompts"].values())
        for t, r in itertools.permutations(ids, 2):
            pairs["cs_syn"].append([t, r])
        for other in by_char[s["groups"][g]["character"]]:
            if other != g:
                pairs["char_syn"].extend([t, r] for t in ids for r in s["groups"][other]["record_ids"])
    # one context table: Danbooru prompts first, then every generated prompt (indices shifted by n_d)
    data = torch.cat([dctx["data"], sctx["data"]])
    offsets = torch.tensor(d_off + [d_off[-1] + o for o in s_off[1:]])
    args.out.mkdir(parents=True, exist_ok=True)
    save_file({"data": data.contiguous(), "offsets": offsets}, str(args.out / "contexts.safetensors"))
    files = [os.path.relpath(args.danbooru / f, args.out) for f in d["files"]["latents"]] + \
            [os.path.relpath(args.synthetic / f, args.out) for f in s["files"]["latents"]]
    pom = dict(d["prompt_of_mode"])
    pom.update({"cs_syn": "stripped", "char_syn": "with_artist"})
    index = {**d, "records": records, "pairs": pairs, "prompt_of_mode": pom, "files": {"latents": files},
             "embed_of_mode": {"cs_syn": "cs", "char_syn": "char"},
             "merged_from": [str(args.danbooru), str(args.synthetic)], "synthetic_groups": len(groups),
             "eval_cases": s.get("eval_cases", []), "held_out_characters": sorted(held)}
    (args.out / "index.json").write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"records": len(records), "pairs": {k: len(v) for k, v in pairs.items()}, "groups": len(groups)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
