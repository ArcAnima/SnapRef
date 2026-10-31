"""Danbooru preprocessing on Modal: fetch -> resize/crop to the training buckets -> WebP -> HF shard upload, and
VAE latents per shard, in parallel containers (Modal).

    modal run --detach deploy/modal_prep.py --ids-file ids.txt --repo 0ppxnhximxr/anima-ref-danbooru-v2 --shard-size 2000         [--latents] [--latents-only]
--detach keeps the run going when this terminal disconnects; progress = shard files in the repo.

Per shard (one container, CPU for the images; a GPU for the VAE when --latents: the first free of L4, A10, L40S):
  1. HTTP range reads of the posts from nyanko-devs/danbooru2026 (tar index petersunde/danbooru2026-index),
     32 threads, straight from the CDN URL each tar resolves to (HF limits resolve calls, not CDN reads);
     a shard is a few whole tars, so the run needs ~1,000 resolve calls; ids missing from the index are skipped;
  2. resize + centre crop (centering 0.5/0.4) to the nearest of the 7 ~1 MP buckets of train/prepare_danbooru.py,
     WebP q=95 (about 4x smaller than the originals, still sharper than what the 1 MP latents keep);
  3. one tar per shard -> <repo>/images/shard_NNNNN.tar (members <id>.webp) + sizes.json;
  4. with --latents: Anima's VAE (qwen_image_vae, Wan 2.1 family) on each image -> latents_NNNNN.safetensors
     (fp16, key = post id) in the same repo.
Shards already in the repo are skipped, so a failed run is resumed by running it again.
Secrets: a Modal secret "anima-hf" holding HF_TOKEN (created from ~/.config/anima-ref/secrets.env by --setup).
"""
import io
import json
import math
import os
import tarfile
import time
from pathlib import Path

import modal

APP = modal.App("anima-danbooru-prep")
BUCKETS = [(1024, 1024), (896, 1152), (1152, 896), (832, 1216), (1216, 832), (768, 1344), (1344, 768)]
DANBOORU = "https://huggingface.co/datasets/nyanko-devs/danbooru2026/resolve/main/original/data-{:04d}.tar"
COMFY_COMMIT = "79cf64c21a1dba50749945ba9fe2e3a692f45438"

cpu_image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "pillow>=11", "huggingface_hub>=0.34", "pyarrow<18", "numpy<2")
gpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install("torch==2.7.0", "torchvision==0.22.0", extra_index_url="https://download.pytorch.org/whl/cu126")
    .run_commands(f"git clone -q https://github.com/comfyanonymous/ComfyUI.git /comfy && git -C /comfy checkout -q {COMFY_COMMIT}",
                  "pip install -q -r /comfy/requirements.txt")
    .pip_install("pillow>=11", "huggingface_hub>=0.34", "pyarrow<18", "safetensors")
)
hf_secret = modal.Secret.from_name("anima-hf")
cache = modal.Volume.from_name("anima-prep-cache", create_if_missing=True)


def bucket_for(w: int, h: int) -> tuple[int, int]:
    return min(BUCKETS, key=lambda b: abs(math.log((w / h) / (b[0] / b[1]))))


def resolve_cdn(tar: int, token: str) -> str:
    """The signed CDN URL behind huggingface.co/.../resolve/main/<tar> (valid ~1 h, accepts Range without auth).
    HF limits resolve calls (12,000 per 5 min per user), so each Danbooru tar is resolved once per shard."""
    import urllib.error
    import urllib.request

    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *a, **k):
            return None
    opener = urllib.request.build_opener(NoRedirect)
    delay = 5.0
    for _ in range(10):
        try:
            opener.open(urllib.request.Request(DANBOORU.format(tar), method="HEAD",
                                               headers={"Authorization": f"Bearer {token}"}), timeout=60)
        except urllib.error.HTTPError as e:
            if e.code in (301, 302, 307, 308) and e.headers.get("Location"):
                return e.headers["Location"]
            if e.code != 429 and e.code < 500:
                raise
        except Exception:
            pass
        time.sleep(delay)
        delay = min(120.0, delay * 2)
    raise RuntimeError(f"could not resolve tar {tar}")


def fetch_shard(rows: list[dict], token: str) -> tuple[dict[int, bytes], dict]:
    import collections
    import urllib.error
    import urllib.request
    from concurrent.futures import ThreadPoolExecutor
    stats = {"ok": 0, "failed": 0, "retries": 0, "resolves": 0}
    by_tar = collections.defaultdict(list)
    for r in rows:
        by_tar[r["shard"]].append(r)
    urls = {}

    def url_of(tar, fresh=False):
        if fresh or tar not in urls:
            urls[tar] = resolve_cdn(tar, token)
            stats["resolves"] += 1
        return urls[tar]

    def get(r):
        delay = 2.0
        for attempt in range(8):
            try:
                req = urllib.request.Request(url_of(r["shard"], fresh=attempt >= 2 and attempt % 2 == 0), headers={
                    "Range": f"bytes={r['offset_bytes']}-{r['offset_bytes'] + r['length_bytes'] - 1}"})
                data = urllib.request.urlopen(req, timeout=120).read()
                if len(data) == r["length_bytes"]:
                    return r["id"], data
            except urllib.error.HTTPError as e:
                if e.code not in (403, 429, 500, 502, 503, 504):   # 403: signed URL expired -> re-resolve
                    return r["id"], None
            except Exception:
                pass
            stats["retries"] += 1
            time.sleep(delay)
            delay = min(60.0, delay * 2)
        return r["id"], None

    out = {}
    for tar in sorted(by_tar):
        url_of(tar)
    with ThreadPoolExecutor(32) as ex:
        for pid, data in ex.map(get, rows):
            if data:
                out[pid] = data
                stats["ok"] += 1
            else:
                stats["failed"] += 1
    return out, stats


def to_webp(data: bytes) -> tuple[bytes, tuple[int, int]] | None:
    from PIL import Image, ImageOps
    try:
        im = Image.open(io.BytesIO(data))
        im = im.convert("RGBA") if im.mode in ("P", "LA", "RGBA") else im.convert("RGB")
        if im.mode == "RGBA":   # alpha onto white, as the reference images in training
            bg = Image.new("RGB", im.size, (255, 255, 255))
            bg.paste(im, mask=im.split()[-1])
            im = bg
        size = bucket_for(*im.size)
        im = ImageOps.fit(im, size, Image.Resampling.LANCZOS, centering=(0.5, 0.4))
        buf = io.BytesIO()
        im.save(buf, "WEBP", quality=95, method=4)
        return buf.getvalue(), size
    except Exception:
        return None


@APP.function(image=cpu_image, secrets=[hf_secret], cpu=8, memory=8192, timeout=3600, max_containers=40, retries=2)
def image_shard(k: int, rows: list[dict], repo: str, done: bool = False) -> dict:
    from huggingface_hub import HfApi
    token = os.environ["HF_TOKEN"].strip()
    api = HfApi(token=token)
    name = f"images/shard_{k:05d}.tar"
    if done:   # the entrypoint lists the repo once; per-shard file_exists calls count against HF's rate limit
        return {"shard": k, "skipped": True}
    t0 = time.time()
    raw, stats = fetch_shard(rows, token)
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(8) as ex:
        conv = dict(zip(raw.keys(), ex.map(to_webp, raw.values(), chunksize=16)))
    sizes, buf = {}, io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for pid, res in conv.items():
            if res is None:
                stats["bad_image"] = stats.get("bad_image", 0) + 1
                continue
            data, size = res
            info = tarfile.TarInfo(f"{pid}.webp")
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
            sizes[pid] = size
    api.upload_file(path_or_fileobj=io.BytesIO(json.dumps(sizes).encode()), path_in_repo=f"images/shard_{k:05d}.json",
                    repo_id=repo, repo_type="dataset")
    api.upload_file(path_or_fileobj=buf.getvalue(), path_in_repo=name, repo_id=repo, repo_type="dataset")
    return {"shard": k, **stats, "images": len(sizes), "mb": round(len(buf.getvalue()) / 2**20), "s": round(time.time() - t0)}


@APP.cls(image=gpu_image, secrets=[hf_secret], gpu=["L4", "A10", "L40S"], timeout=3600, max_containers=10,
         volumes={"/cache": cache}, retries=2)
class VaeEncoder:
    @modal.enter()
    def load(self):
        import sys
        import torch
        from huggingface_hub import hf_hub_download
        sys.path.insert(0, "/comfy")
        os.chdir("/comfy")
        import comfy.sd
        import comfy.utils
        p = hf_hub_download("circlestone-labs/Anima", "split_files/vae/qwen_image_vae.safetensors", cache_dir="/cache/hf")
        self.vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(p))
        self.torch = torch

    @modal.method()
    def shard(self, k: int, repo: str) -> dict:
        import numpy as np
        from huggingface_hub import HfApi, hf_hub_download
        from PIL import Image
        from safetensors.torch import save
        torch = self.torch
        api = HfApi(token=os.environ["HF_TOKEN"].strip())
        name = f"latents/latents_{k:05d}.safetensors"
        if api.file_exists(repo, name, repo_type="dataset"):
            return {"shard": k, "skipped": True}
        t0 = time.time()
        tar_path = hf_hub_download(repo, f"images/shard_{k:05d}.tar", repo_type="dataset", token=api.token, cache_dir="/tmp/hf")
        from concurrent.futures import ThreadPoolExecutor
        with tarfile.open(tar_path) as tf:
            members = [(m.name, tf.extractfile(m).read()) for m in tf.getmembers()]

        def decode(item):   # CPU threads decode while the GPU encodes the previous image
            n, data = item
            a = np.asarray(Image.open(io.BytesIO(data)).convert("RGB"), dtype=np.float32) / 255.0
            return n[: -len(".webp")], torch.from_numpy(a).unsqueeze(0)
        out = {}
        with ThreadPoolExecutor(6) as ex, torch.inference_mode():
            for pid, px in ex.map(decode, members):
                lat = self.vae.encode(px)
                if lat.ndim == 4:
                    lat = lat.unsqueeze(2)
                out[pid] = lat[0].to(torch.float16).cpu().contiguous()
        api.upload_file(path_or_fileobj=save(out), path_in_repo=name, repo_id=repo, repo_type="dataset")
        return {"shard": k, "latents": len(out), "s": round(time.time() - t0)}


@APP.local_entrypoint()
def main(ids_file: str = "", repo: str = "0ppxnhximxr/anima-ref-danbooru-v2", shard_size: int = 2000,
         latents: bool = False, latents_only: bool = False, limit_shards: int = 0, setup: bool = False):
    if setup:
        env = dict(l.strip().split("=", 1) for l in Path(os.path.expanduser("~/.config/anima-ref/secrets.env")).read_text().splitlines() if "=" in l)
        import subprocess
        subprocess.run(["modal", "secret", "create", "anima-hf", f"HF_TOKEN={env['HF_TOKEN'].strip()}", "--force"], check=True,
                       stdout=subprocess.DEVNULL)
        print("secret anima-hf created")
        return
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi()
    api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
    assert api.repo_info(repo, repo_type="dataset").private
    ids = [int(x) for x in Path(ids_file).read_text().split()]
    idx = pq.read_table(hf_hub_download("petersunde/danbooru2026-index", "tar-index.parquet", repo_type="dataset"))
    rows = idx.filter(pc.is_in(idx["id"], value_set=pa.array(ids))).to_pylist()
    rows.sort(key=lambda r: r["id"])
    print(f"ids {len(ids)}, in the tar index {len(rows)}")
    # a shard = a few whole Danbooru tars (posts are spread over 1,000 tars by id % 1000), so each tar is resolved
    # once in one container: ~1,000 resolve calls in total instead of one per image
    by_tar = {}
    for r in rows:
        by_tar.setdefault(r["shard"], []).append(r)
    per = max(1, round(shard_size / (len(rows) / max(1, len(by_tar)))))
    tars = sorted(by_tar)
    shards = [[r for t in tars[i: i + per] for r in by_tar[t]] for i in range(0, len(tars), per)]
    print(f"{len(shards)} shards of {per} tars (~{len(rows) // max(1, len(shards))} images)")
    if limit_shards:
        shards = shards[:limit_shards]
    t0 = time.time()
    existing = {s.rfilename for s in api.dataset_info(repo).siblings}
    if not latents_only:
        todo = [(k, s, repo, f"images/shard_{k:05d}.tar" in existing) for k, s in enumerate(shards)]
        print(f"{sum(not t[3] for t in todo)} image shards to do")
        results = list(image_shard.starmap(todo))
        print(json.dumps(results[:5]), "...", flush=True)
        print(f"images: {sum(r.get('images', 0) for r in results)} in {time.time() - t0:.0f}s", flush=True)
    if latents or latents_only:
        t0 = time.time()
        enc = VaeEncoder()
        lres = list(enc.shard.starmap([(k, repo) for k in range(len(shards))]))
        print(json.dumps(lres[:5]), "...")
        print(f"latents: {sum(r.get('latents', 0) for r in lres)} in {time.time() - t0:.0f}s")
