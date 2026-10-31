"""DINOv2 layer statistics (style_probe.py's recipe: 448 px centre crop, layers 2..12 mean and std) for image files.

    CUDA_VISIBLE_DEVICES=1 python train/style_research/ref_feats.py --images ~/pilot_data/images \
        --ids-from ~/train/pilot_prep_v2/index.json --out ~/style_agent/feat_pilot_dinov2.npz

--ids-from: a prep index.json (its records) or a text file of ids; --images holds <id>.webp. Transparent images are
put on white, as prepare_danbooru.py does for the latents. Output: npz ids (int64), feat (fp16 [N, 9216]).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from style_probe import BACKBONES, preprocess  # noqa: E402

SPEC = BACKBONES["dinov2"]


def open_rgb(path) -> Image.Image:
    img = Image.open(path)
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[-1])
        return bg
    return img.convert("RGB")


class Files(torch.utils.data.Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, k):
        pid, path = self.items[k]
        return pid, preprocess(open_rgb(path), SPEC["size"], SPEC["mean"], SPEC["std"])


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--ids-from", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.ids_from.suffix == ".json":
        ids = [int(k) for k in json.loads(args.ids_from.read_text(encoding="utf-8"))["records"]]
    else:
        ids = [int(x) for x in args.ids_from.read_text().split()]
    items = [(i, args.images / f"{i}.webp") for i in ids if (args.images / f"{i}.webp").exists()]
    from transformers import AutoModel
    model = AutoModel.from_pretrained(SPEC["repo"]).to("cuda").eval().half()
    dl = torch.utils.data.DataLoader(Files(items), batch_size=64, num_workers=16, pin_memory=True)
    out_ids, feats = [], []
    for k, (pid, x) in enumerate(dl):
        hs = model(pixel_values=x.to("cuda", torch.float16), output_hidden_states=True).hidden_states
        parts = []
        for layer in SPEC["layers"]:
            t = hs[layer][:, SPEC["skip_tokens"]:].float()
            parts += [t.mean(1), t.std(1)]
        feats.append(torch.cat(parts, 1).half().cpu().numpy())
        out_ids += pid.tolist()
        if k % 100 == 0:
            print(f"{len(out_ids)}/{len(items)}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, ids=np.array(out_ids, dtype=np.int64), feat=np.concatenate(feats))
    print(f"FEATS_DONE {len(out_ids)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
