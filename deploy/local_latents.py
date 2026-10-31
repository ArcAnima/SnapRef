"""VAE-encode the v2 image shards on our own GPUs (same output as deploy/modal_prep.py's VaeEncoder: one
latents/latents_NNNNN.safetensors per shard, {post id: fp16 raw latent [16, 1, h, w]}), and upload them to HF.

    CUDA_VISIBLE_DEVICES=0 ~/comfy-venv/bin/python deploy/local_latents.py IMAGES_DIR OUT_DIR --part 0/2 [--repo R]

IMAGES_DIR holds images/shard_NNNNN.tar (snapshot of the HF dataset). Shards are split over --part k/n by index,
so two GPUs (or machines) each take every n-th shard. A shard already on HF (listed once at start: HF rate-limits
per-file checks) or already in OUT_DIR is skipped; an existing local file is only uploaded.
"""
import argparse
import io
import os
import sys
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))
import anima_env  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("images", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--part", default="0/1")
    ap.add_argument("--repo", default="0ppxnhximxr/anima-ref-danbooru-v2")
    ap.add_argument("--comfy-root", default=str(Path.home() / "ComfyUI"))
    args = ap.parse_args()
    k, n = map(int, args.part.split("/"))
    from huggingface_hub import HfApi
    from safetensors.torch import save_file
    api = HfApi(token=os.environ["HF_TOKEN"].strip())
    on_hf = {s.rfilename for s in api.dataset_info(args.repo).siblings}
    shards = sorted(args.images.glob("images/shard_*.tar"))
    todo = [p for i, p in enumerate(shards) if i % n == k and f"latents/latents_{p.stem[6:]}.safetensors" not in on_hf]
    print(f"part {k}/{n}: {len(todo)} shards to do", flush=True)
    anima_env.setup_comfy(args.comfy_root)
    vae = anima_env.load_vae("qwen_image_vae.safetensors")
    args.out.mkdir(parents=True, exist_ok=True)

    def decode(item):   # CPU threads decode while the GPU encodes
        name, data = item
        a = np.asarray(Image.open(io.BytesIO(data)).convert("RGB"), dtype=np.float32) / 255.0
        return name[: -len(".webp")], torch.from_numpy(a).unsqueeze(0)
    for tar in todo:
        num = tar.stem[6:]
        dst = args.out / f"latents_{num}.safetensors"
        t0 = time.time()
        if not dst.exists():
            with tarfile.open(tar) as tf:
                members = [(m.name, tf.extractfile(m).read()) for m in tf.getmembers()]
            out = {}
            with ThreadPoolExecutor(8) as ex, torch.inference_mode():
                for pid, px in ex.map(decode, members):
                    lat = vae.encode(px)
                    if lat.ndim == 4:
                        lat = lat.unsqueeze(2)
                    out[pid] = lat[0].to(torch.float16).cpu().contiguous()
            save_file(out, str(dst.with_suffix(".tmp")))
            dst.with_suffix(".tmp").rename(dst)
        for attempt in range(5):
            try:
                api.upload_file(path_or_fileobj=str(dst), path_in_repo=f"latents/{dst.name}", repo_id=args.repo, repo_type="dataset")
                break
            except Exception as e:   # HF 5xx / 429: wait and retry; the local file stays either way
                print(f"upload {dst.name} failed ({type(e).__name__}), retry {attempt + 1}", flush=True)
                time.sleep(30 * (attempt + 1))
        print(f"shard {num}: {time.time() - t0:.0f}s", flush=True)
    print("PART_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
