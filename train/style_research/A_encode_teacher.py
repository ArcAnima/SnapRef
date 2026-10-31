"""Experiment A: text contexts of the "@artist" teacher prompts for the style pairs of known artists.

    CUDA_VISIBLE_DEVICES=0 python train/style_research/A_encode_teacher.py --prep ~/train/pilot_prep_v2 \
        --known ~/style_agent/A/A_known_artists.json --out ~/style_agent/A/teacher_ctx

Teacher prompt = the pair's target prompt (pilot_prep_v2 new_prompts.json) with "@<artist>" inserted after the quality
tags, where build_diag_pairs.py puts it (the D1 artist direction was measured with that spelling). Encoded with
anima_env.encode_prompt, the encoder add_pair_prompts.py used for the prep contexts, so v(@a) - v(plain) carries no
encoder mismatch (fast_encode.py differs by ~1.8% relative L2, more than the artist direction itself).
Only style pairs whose target artist is in --known (Danbooru posts >= 100, none of the 33 eval artists).
Writes OUT/contexts.safetensors (data, offsets) and OUT/map.json {prep prompt index: teacher index}. Resumable by parts.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import save_file

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent / "datagen"))
import anima_env  # noqa: E402
from catalog import anima_tag  # noqa: E402

QUALITY = "masterpiece, best quality, score_7, newest, highres, absurdres, "


def teacher_prompt(prompt: str, artist: str) -> str:
    assert prompt.startswith(QUALITY), prompt[:80]
    return QUALITY + "@" + anima_tag(artist) + ", " + prompt[len(QUALITY):]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", type=Path, required=True)
    ap.add_argument("--known", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--comfy-root", default=str(Path.home() / "ComfyUI"))
    ap.add_argument("--part-size", type=int, default=5000)
    args = ap.parse_args()
    index = json.loads((args.prep / "index.json").read_text(encoding="utf-8"))
    texts = json.loads((args.prep / "new_prompts.json").read_text(encoding="utf-8"))
    recs = index["records"]
    offset = max(p[2] for ps in index["pairs"].values() for p in ps) + 1 - len(texts)   # add_pair_prompts.py layout
    known = set(json.loads(args.known.read_text(encoding="utf-8"))["artists"])
    jobs = sorted({(p[2], recs[p[0]]["artist"]) for p in index["pairs"]["style"] if recs[p[0]]["artist"] in known})
    assert len({j[0] for j in jobs}) == len(jobs), "one prompt index with two artists"
    print(f"teacher prompts: {len(jobs)} (prompt offset {offset})", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    cfg = index["config"]["comfy"]
    anima_env.setup_comfy(args.comfy_root)
    dev = torch.device("cuda", 0)
    _, dm = anima_env.load_diffusion_model(cfg["unet_name"], dev)
    clip = anima_env.load_text_encoder(cfg["clip_name"])
    started = time.perf_counter()
    parts = []
    for k in range(0, len(jobs), args.part_size):
        f = args.out / f"part_{k // args.part_size:03d}.pt"
        parts.append(f)
        if f.exists():
            continue
        out = {pi: anima_env.encode_prompt(clip, dm, teacher_prompt(texts[pi - offset], a), dev)
               for pi, a in jobs[k:k + args.part_size]}
        torch.save(out, f)
        print(f"{k + len(out)}/{len(jobs)} {time.perf_counter() - started:.0f}s", flush=True)
    chunks, mapping = [], {}
    for f in parts:
        for pi, c in torch.load(f, weights_only=True).items():
            mapping[int(pi)] = len(chunks)
            chunks.append(c)
    offsets = torch.tensor([0] + [c.shape[0] for c in chunks]).cumsum(0)
    save_file({"data": torch.cat(chunks).contiguous(), "offsets": offsets}, str(args.out / "contexts.safetensors"))
    (args.out / "map.json").write_text(json.dumps({"prompt_offset": offset, "map": mapping,
                                                   "example": teacher_prompt(texts[jobs[0][0] - offset], jobs[0][1])},
                                                  ensure_ascii=False), encoding="utf-8")
    print(f"TEACHER_CTX_DONE {len(chunks)} {time.perf_counter() - started:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
