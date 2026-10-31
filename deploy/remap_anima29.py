"""Remap a SnapRef (anima_ref_kv/1) adapter from Anima base (28 blocks) to Anima-2.9B (40 blocks).

Anima-2.9B (Gazingstars123/Anima-2.9B) inserts 12 new blocks between the original ones; the original blocks
keep their weights but move to new indices. LoRA keys `anima_ref_kv.<stream>.blocks.<i>.*` are renamed to the
new index of original block i. Inserted blocks get no LoRA.

    python deploy/remap_anima29.py SRC.safetensors DST.safetensors [--check-base BASE --check-29 ANIMA29]
--check-* compares the moved blocks' weights between the two base models (they should be identical).
"""
import argparse
import json
import re

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

INSERTED = [2, 5, 8, 11, 14, 17, 21, 24, 27, 30, 33, 36]   # expand_manifest.json "insertion_positions"
NEW_BLOCKS = 40
OLD_TO_NEW = [i for i in range(NEW_BLOCKS) if i not in INSERTED]
assert len(OLD_TO_NEW) == 28


def check(base: str, a29: str) -> None:
    with safe_open(base, "pt") as b, safe_open(a29, "pt") as n:
        nkeys = set(n.keys())
        pre = "net." if any(k.startswith("net.blocks.") for k in b.keys()) else ""
        worst = 0.0
        for old, new in enumerate(OLD_TO_NEW):
            for k in (f"{pre}blocks.{old}.self_attn.q_proj.weight", f"{pre}blocks.{old}.mlp.layer2.weight"):
                k29 = k.replace(f"blocks.{old}.", f"blocks.{new}.")
                assert k29 in nkeys, k29
                d = (b.get_tensor(k).float() - n.get_tensor(k29).float()).abs().max().item()
                worst = max(worst, d)
        print(json.dumps({"moved_blocks_max_abs_diff_vs_base": worst}))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--check-base")
    ap.add_argument("--check-29")
    args = ap.parse_args()
    if args.check_base and args.check_29:
        check(args.check_base, args.check_29)
    tensors = load_file(args.src)
    with safe_open(args.src, "pt") as h:
        metadata = dict(h.metadata() or {})
    pat = re.compile(r"^(anima_ref_kv\.[a-z]+\.blocks\.)(\d+)(\..*)$")
    out = {}
    for k, v in tensors.items():
        m = pat.match(k)
        out[f"{m.group(1)}{OLD_TO_NEW[int(m.group(2))]}{m.group(3)}" if m else k] = v
    assert len(out) == len(tensors)
    meta = json.loads(metadata.get("snapref") or metadata.get("anima_ref_kv") or "{}")
    meta["base"] = "Anima-2.9B-preview-v1.safetensors (remapped from anima-base-v1.0)"
    metadata["anima_ref_kv"] = json.dumps(meta)
    save_file(out, args.dst, metadata=metadata)
    print(json.dumps({"out": args.dst, "tensors": len(out)}))


if __name__ == "__main__":
    main()
