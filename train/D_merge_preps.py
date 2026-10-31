"""train/merge_preps.py for a Danbooru prep whose contexts.safetensors is larger than RAM (v2_prep: 308 GB).

    python train/D_merge_preps.py --danbooru ~/v2_pool/hf/preps/v2_prep --synthetic ~/v2_pool/hf/preps/asym_prep \
        --out ~/v2_pool/hf/preps/v2_mix_prep [--synthetic-groups 3000] [--contexts-only]

Same records, pairs (cs_syn / char_syn), prompt_of_mode, embed_of_mode and relative latent paths as merge_preps.py
(same seed, same group choice). Only the context table is written differently: merge_preps.py torch.cat's both
tables in memory; here the header is written first and the two data blocks are copied as bytes (Danbooru first,
then synthetic, so prompt indices are the same as merge_preps.py's). --contexts-only rebuilds contexts.safetensors
of an existing --out (its index.json is not touched), e.g. after downloading index.json from HF.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import shutil
from pathlib import Path

import torch


def read_table(path: Path) -> tuple[dict, int, list[int]]:
    """Header, data start and offsets of a {data, offsets} safetensors file, without mmap (the file may exceed RAM).
    Tensor order inside the file varies (save_file puts offsets first, D_build_v2_prep.py data first)."""
    with path.open("rb") as f:
        n = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(n))
        a, b = header["offsets"]["data_offsets"]
        f.seek(8 + n + a)
        offsets = torch.frombuffer(bytearray(f.read(b - a)), dtype=torch.int64).tolist()
    assert header["data"]["dtype"] == "BF16"
    assert header["data"]["shape"][0] == offsets[-1]
    return header, 8 + n + header["data"]["data_offsets"][0], offsets


def merge_contexts(d_path: Path, s_path: Path, out_path: Path) -> int:
    (dh, ds, d_off), (sh, ss, s_off) = read_table(d_path), read_table(s_path)
    dim = dh["data"]["shape"][1]
    assert sh["data"]["shape"][1] == dim
    offsets = d_off + [d_off[-1] + o for o in s_off[1:]]
    n_tok = offsets[-1]
    data_bytes = n_tok * dim * 2
    header = {"data": {"dtype": "BF16", "shape": [n_tok, dim], "data_offsets": [0, data_bytes]},
              "offsets": {"dtype": "I64", "shape": [len(offsets)], "data_offsets": [data_bytes, data_bytes + 8 * len(offsets)]}}
    h = json.dumps(header).encode()
    h += b" " * (-len(h) % 8)
    tmp = out_path.with_suffix(".tmp")
    with tmp.open("wb") as out:
        out.write(len(h).to_bytes(8, "little"))
        out.write(h)
        for path, start, nbytes in ((d_path, ds, d_off[-1] * dim * 2), (s_path, ss, s_off[-1] * dim * 2)):
            with path.open("rb") as f:
                f.seek(start)
                left = nbytes
                while left:
                    buf = f.read(min(left, 1 << 28))
                    out.write(buf)
                    left -= len(buf)
        out.write(torch.tensor(offsets, dtype=torch.int64).numpy().tobytes())
    tmp.rename(out_path)
    return len(offsets) - 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--danbooru", type=Path, required=True)
    ap.add_argument("--synthetic", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--synthetic-groups", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--contexts-only", action="store_true")
    ap.add_argument("--index-only", action="store_true", help="rewrite index.json only; reuse the existing "
                    "contexts.safetensors of --out (the table holds every synthetic prompt, whatever --synthetic-groups is)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.contexts_only:
        n = merge_contexts(args.danbooru / "contexts.safetensors", args.synthetic / "contexts.safetensors",
                           args.out / "contexts.safetensors")
        print(json.dumps({"contexts": n}))
        return 0
    rng = random.Random(args.seed)
    d = json.loads((args.danbooru / "index.json").read_text(encoding="utf-8"))
    s = json.loads((args.synthetic / "index.json").read_text(encoding="utf-8"))
    n_d = len(read_table(args.danbooru / "contexts.safetensors")[2]) - 1
    held = set(s["held_out_characters"])
    groups = [g for g, v in s["groups"].items() if v["split"] == "train" and v["character"] not in held]
    rng.shuffle(groups)
    groups = groups[: args.synthetic_groups]
    by_char = {}
    for g in groups:
        by_char.setdefault(s["groups"][g]["character"], []).append(g)
    records = dict(d["records"])
    pairs = {k: list(v) for k, v in d["pairs"].items()}
    pairs["cs_syn"], pairs["char_syn"] = [], []
    for g in groups:
        ids = s["groups"][g]["record_ids"]
        for rid in ids:
            r = s["records"][rid]
            records[rid] = {"character": r["character"], "artist": "|".join(a["tag"] if isinstance(a, dict) else a
                                                                             for a in s["groups"][g].get("artists", [])),
                            "width": r["width"], "height": r["height"], "focus": r.get("focus"), "framing": r.get("framing"),
                            "prompts": {k: n_d + v for k, v in r["prompts"].items()}}
        for t, r in itertools.permutations(ids, 2):
            pairs["cs_syn"].append([t, r])
        for other in by_char[s["groups"][g]["character"]]:
            if other != g:
                pairs["char_syn"].extend([t, r] for t in ids for r in s["groups"][other]["record_ids"])
    if args.index_only:
        n = len(read_table(args.out / "contexts.safetensors")[2]) - 1
        n_s = len(read_table(args.synthetic / "contexts.safetensors")[2]) - 1
        assert n == n_d + n_s, f"existing table has {n} contexts, expected {n_d} + {n_s}"
    else:
        n = merge_contexts(args.danbooru / "contexts.safetensors", args.synthetic / "contexts.safetensors",
                           args.out / "contexts.safetensors")
    files = [os.path.relpath(args.danbooru / f, args.out) for f in d["files"]["latents"]] + \
            [os.path.relpath(args.synthetic / f, args.out) for f in s["files"]["latents"]]
    pom = dict(d["prompt_of_mode"])
    pom.update({"cs_syn": "stripped", "char_syn": "with_artist"})
    index = {**d, "records": records, "pairs": pairs, "prompt_of_mode": pom, "files": {"latents": files},
             "embed_of_mode": {"cs_syn": "cs", "char_syn": "char"},
             "merged_from": [str(args.danbooru), str(args.synthetic)], "synthetic_groups": len(groups),
             "eval_cases": s.get("eval_cases", []), "held_out_characters": sorted(held)}
    (args.out / "index.json").write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")
    shutil.copy(Path(__file__), args.out / "D_merge_preps.py")
    print(json.dumps({"records": len(records), "contexts": n, "pairs": {k: len(v) for k, v in pairs.items()},
                      "groups": len(groups)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
