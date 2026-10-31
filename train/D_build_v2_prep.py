"""SnapRef v2 prep (train_asym.py --prep format, as add_pair_prompts.py's pilot_prep_v2) from the v2 pool and the
HF latents of 0ppxnhximxr/anima-ref-danbooru-v2.

    python train/D_build_v2_prep.py index  --pool ~/v2_pool --hf ~/v2_pool/hf --out ~/v2_pool/hf/preps/v2_prep
    python train/D_build_v2_prep.py encode --out ~/v2_pool/hf/preps/v2_prep      # GPU, resumable
    python train/D_build_v2_prep.py check  --out ~/v2_pool/hf/preps/v2_prep      # GPU, FastEncoder vs encode_prompt

  * latents are not re-encoded or copied: index["files"]["latents"] lists ../../latents/latents_NNNNN.safetensors
    (the repo layout), {post id: fp16 raw VAE latent [16, 1, h, w]} of the bucketed WebP = prepare_danbooru.py's
    format (raw, fp16, its 7 buckets; train_asym.py applies process_in when it loads them). `index` checks every
    shape against images/shard_NNNNN.json and the bucket list;
  * context table: "" first, then each record's ref_caption (gender, solo, look; build_danbooru_pilot.py), then every
    distinct pair prompt; index["pairs"][mode] = [[target, ref, prompt index], ...] as add_pair_prompts.py;
  * text encoder: train/fast_encode.py FastEncoder (BF16, 16-token buckets, masked adapter). encode_prompt would take
    1.44M / 24 p/s = 16.7 h on the A10; FastEncoder differs from it by ~1.8% rel L2 (bench_te/REPORT.md, `check`);
  * the table (~308 GB) does not fit in RAM: `index` tokenizes every text first (context length = T5 token count),
    writes the safetensors header and offsets, and `encode` writes each chunk in place (chunks done are logged in
    encode_done.txt, so a restart resumes).
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import random
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "datagen"))
import anima_env  # noqa: E402

BUCKETS = [(1024, 1024), (896, 1152), (1152, 896), (832, 1216), (1216, 832), (768, 1344), (1344, 768)]  # prepare_danbooru
PROMPT_OF_MODE = {"cs": "stripped", "char": "with_artist", "style": "with_character"}
DIM, CHUNK = 1024, 16384
_clip = None


def _ntok(text: str) -> int:
    return len(_clip.tokenize(text)["t5xxl"][0])


def write_header(path: Path, n_tokens: int, offsets: list[int]) -> int:
    """Preallocate a safetensors file {data: bf16 [n_tokens, DIM], offsets: i64 [n+1]}; return the data start."""
    data_bytes = n_tokens * DIM * 2
    header = {"data": {"dtype": "BF16", "shape": [n_tokens, DIM], "data_offsets": [0, data_bytes]},
              "offsets": {"dtype": "I64", "shape": [len(offsets)], "data_offsets": [data_bytes, data_bytes + 8 * len(offsets)]}}
    h = json.dumps(header).encode()
    h += b" " * (-len(h) % 8)
    with path.open("wb") as f:
        f.write(len(h).to_bytes(8, "little"))
        f.write(h)
        f.seek(8 + len(h) + data_bytes)
        f.write(torch.tensor(offsets, dtype=torch.int64).numpy().tobytes())
    return 8 + len(h)


def cmd_index(args) -> None:
    global _clip
    from safetensors import safe_open
    from catalog import anima_tag
    t0 = time.time()
    recs = {str(r["id"]): r for r in map(json.loads, (args.pool / "records.jsonl").open(encoding="utf-8"))}
    sizes, files, bad = {}, [], []
    allowed = {(h // 8, w // 8) for w, h in BUCKETS}
    for js in sorted((args.hf / "images").glob("shard_*.json")):
        num = js.stem[6:]
        lat = args.hf / "latents" / f"latents_{num}.safetensors"
        want = json.loads(js.read_text())
        with safe_open(str(lat), "pt") as f:
            for k in f.keys():
                sl = f.get_slice(k)
                shape, dtype = sl.get_shape(), sl.get_dtype()
                w, h = want[k]
                if dtype != "F16" or len(shape) != 4 or shape[:2] != [16, 1] or (shape[2], shape[3]) != (h // 8, w // 8) \
                        or (shape[2], shape[3]) not in allowed:
                    bad.append(k)
                    continue
                sizes[k] = (shape[3] * 8, shape[2] * 8)
        files.append(os.path.relpath(lat, args.out))
    print(f"latents {len(sizes)}, bad {len(bad)}, shards {len(files)}, {time.time() - t0:.0f}s", flush=True)
    assert not bad, bad[:5]
    keep = [k for k in recs if k in sizes]
    caption = {k: ", ".join([recs[k]["gender"], "solo"] + [anima_tag(x) for x in recs[k]["look"]]) for k in keep}
    pairs_in = [p for p in map(json.loads, args.pairs.open(encoding="utf-8"))
                if str(p["target"]) in sizes and str(p["ref"]) in sizes]
    prompts: dict[str, int] = {"": 0}
    for k in keep:
        prompts.setdefault(caption[k], len(prompts))
    n_caps = len(prompts)
    for p in sorted({p["prompt"] for p in pairs_in}):
        prompts.setdefault(p, len(prompts))
    texts = list(prompts)
    print(f"records {len(keep)}, pairs {len(pairs_in)}, texts {len(texts)} (captions+'' {n_caps})", flush=True)
    anima_env.setup_comfy(args.comfy_root)
    _clip = anima_env.load_text_encoder("qwen_3_06b_base.safetensors")
    with mp.get_context("fork").Pool(args.workers) as pool:
        lens = pool.map(_ntok, texts, chunksize=2000)
    offsets = [0]
    for n in lens:
        offsets.append(offsets[-1] + n)
    print(f"tokens {offsets[-1]} ({offsets[-1] * DIM * 2 / 1e9:.1f} GB), {time.time() - t0:.0f}s", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "texts.json").write_text(json.dumps(texts, ensure_ascii=False), encoding="utf-8")
    write_header(args.out / "contexts.safetensors", offsets[-1], offsets)
    (args.out / "encode_done.txt").write_text("")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    config["data"]["source"] = "danbooru2026 v2 pool (select_v2_images.py), pairs v2 (build_pairs_v2.py)"
    records = {k: {"character": recs[k]["character"], "artist": recs[k]["artist"], "rating": recs[k]["rating"],
                   "prompts": {"ref_caption": prompts[caption[k]]}, "focus": None, "framing": None,
                   "width": sizes[k][0], "height": sizes[k][1]} for k in keep}
    pairs = {}
    for p in pairs_in:
        pairs.setdefault(p["mode"], []).append([str(p["target"]), str(p["ref"]), prompts[p["prompt"]]])
    index = {"created": time.time(), "config": config, "data_dir": str(args.hf), "records": records, "groups": {},
             "pairs": pairs, "prompt_of_mode": PROMPT_OF_MODE, "prompts": None, "eval_cases": [],
             "held_out_characters": [], "files": {"latents": files}, "pair_prompts": True,
             "records_file": str(args.pool / "records.jsonl"), "pairs_file": str(args.pairs),
             "text_encoder": "train/fast_encode.py FastEncoder (BF16, 16-token buckets, masked llm_adapter); "
                             "~1.8% rel L2 from anima_env.encode_prompt"}
    (args.out / "index.json").write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"records": len(records), "texts": len(texts), "ref_captions": n_caps - 1,
                      "pair_prompts": len(texts) - n_caps, "pairs": {m: len(v) for m, v in pairs.items()},
                      "seconds": round(time.time() - t0)}), flush=True)


def _load_models(comfy_root: str):
    anima_env.setup_comfy(comfy_root)
    device = torch.device("cuda", 0)
    _, dm = anima_env.load_diffusion_model("anima-base-v1.0.safetensors", device)
    clip = anima_env.load_text_encoder("qwen_3_06b_base.safetensors")
    return clip, dm, device


def _table(out: Path):
    with (out / "contexts.safetensors").open("rb") as f:
        n = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(n))
        start = 8 + n
        a, b = header["offsets"]["data_offsets"]
        f.seek(start + a)   # read directly: safetensors/torch mmap of a file larger than RAM fails (overcommit)
        offsets = torch.frombuffer(bytearray(f.read(b - a)), dtype=torch.int64).tolist()
    assert header["data"]["shape"][0] == offsets[-1]
    return start, offsets


def cmd_encode(args) -> None:
    from fast_encode import FastEncoder
    texts = json.loads((args.out / "texts.json").read_text(encoding="utf-8"))
    start, offsets = _table(args.out)
    done_path = args.out / "encode_done.txt"
    done = {int(x) for x in done_path.read_text().split()}
    chunks = [c for c in range(-(-len(texts) // CHUNK)) if c not in done]
    clip, dm, device = _load_models(args.comfy_root)
    enc = FastEncoder(clip, dm, device)
    t0, n_done = time.time(), 0
    with (args.out / "contexts.safetensors").open("r+b") as f, done_path.open("a") as log:
        for c in chunks:
            a, b = c * CHUNK, min(len(texts), (c + 1) * CHUNK)
            ctxs = enc(texts[a:b])
            for i, x in enumerate(ctxs):
                assert x.shape == (offsets[a + i + 1] - offsets[a + i], DIM) and x.dtype == torch.bfloat16, (a + i, x.shape)
            f.seek(start + offsets[a] * DIM * 2)
            f.write(torch.cat(ctxs).contiguous().view(torch.int16).numpy().tobytes())
            f.flush()
            log.write(f"{c}\n")
            log.flush()
            n_done += b - a
            el = time.time() - t0
            print(f"chunk {c}: {n_done}/{sum(min(len(texts), (k + 1) * CHUNK) - k * CHUNK for k in chunks)} "
                  f"{n_done / el:.1f} p/s, {el:.0f}s", flush=True)
    print("ENCODE_DONE", flush=True)


def cmd_check(args) -> None:
    """FastEncoder vs encode_prompt on a sample of this table's texts, and the written rows vs a fresh FastEncoder."""
    from fast_encode import FastEncoder
    texts = json.loads((args.out / "texts.json").read_text(encoding="utf-8"))
    done = {int(x) for x in (args.out / "encode_done.txt").read_text().split()}
    assert len(done) == -(-len(texts) // CHUNK), f"{len(done)} chunks done"
    start, off = _table(args.out)
    idx = [0] + random.Random(0).sample(range(1, len(texts)), args.n)
    clip, dm, device = _load_models(args.comfy_root)
    enc = FastEncoder(clip, dm, device)
    sample = [texts[i] for i in idx]
    m = enc.self_check(sample)
    stored = []
    with (args.out / "contexts.safetensors").open("rb") as f:
        for i in idx:
            f.seek(start + off[i] * DIM * 2)
            raw = bytearray(f.read((off[i + 1] - off[i]) * DIM * 2))
            stored.append(torch.frombuffer(raw, dtype=torch.bfloat16).view(-1, DIM).float())
    ref = [anima_env.encode_prompt(clip, dm, t, device).float() for t in sample]
    rel = [float((s - r).norm() / r.norm()) for s, r in zip(stored, ref)]
    cos = torch.cat([torch.nn.functional.cosine_similarity(s, r, dim=-1) for s, r in zip(stored, ref)])
    res = {"fast_vs_encode_prompt": m, "stored_vs_encode_prompt": {"rel_l2": sum(rel) / len(rel), "rel_l2_max": max(rel),
           "cos_mean": float(cos.mean()), "cos_min": float(cos.min()), "n": len(idx)},
           "rows_zero": sum(int(s.abs().sum() == 0) for s in stored)}
    (args.out / "encoder_check.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res), flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["index", "encode", "check"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--pool", type=Path, default=Path.home() / "v2_pool")
    ap.add_argument("--hf", type=Path, default=Path.home() / "v2_pool" / "hf")
    ap.add_argument("--pairs", type=Path, default=Path.home() / "v2_pool" / "pairs" / "pairs.jsonl")
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "train_pilot2_ft.json")
    ap.add_argument("--comfy-root", default=str(Path.home() / "ComfyUI"))
    ap.add_argument("--workers", type=int, default=28)
    ap.add_argument("--n", type=int, default=500, help="check: sample size")
    args = ap.parse_args()
    {"index": cmd_index, "encode": cmd_encode, "check": cmd_check}[args.cmd](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
