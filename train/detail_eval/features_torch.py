"""Torch features of head crops for the detail eval (run in ~/comfy-venv, after features_imgutils.py on the same DIR).

    ~/comfy-venv/bin/python train/detail_eval/features_torch.py --set ~/detail_eval/set [--renders DIR]

Head crop = the box in DIR/feat_imgutils.json (whole image when no head was found). Per image:
  dino_patch  DINOv2-base patch tokens of the crop resized to 224 x 224 (16 x 16 tokens, L2-normalised, fp16)
  siglip      SigLIP2 so400m-patch14-384 image embedding of the crop (L2-normalised)
Writes DIR/feat_torch.npz; images already there are skipped.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from features_imgutils import items  # noqa: E402

SIGLIP = "google/siglip2-so400m-patch14-384"
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


@torch.no_grad()
def main() -> int:
    from transformers import AutoImageProcessor, AutoModel
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", type=Path, required=True)
    ap.add_argument("--renders", type=Path)
    args = ap.parse_args()
    out_dir = args.renders or args.set
    todo = items(args.set, args.renders)
    boxes = json.loads((out_dir / "feat_imgutils.json").read_text(encoding="utf-8"))
    npath = out_dir / "feat_torch.npz"
    feats = dict(np.load(npath)) if npath.exists() else {}
    new = [k for k in todo if f"{k}|siglip" not in feats]
    if not new:
        print(json.dumps({"dir": str(out_dir), "new": 0}))
        return 0
    device = torch.device("cuda", 0)
    dino = AutoModel.from_pretrained("facebook/dinov2-base").to(device).eval().half()
    sig_proc = AutoImageProcessor.from_pretrained(SIGLIP)
    sig = AutoModel.from_pretrained(SIGLIP, torch_dtype=torch.float16).to(device).eval()
    mean, std = torch.tensor(MEAN).view(3, 1, 1), torch.tensor(STD).view(3, 1, 1)
    for i, k in enumerate(new):
        img = Image.open(todo[k]).convert("RGB")
        box = boxes[k]["head_box"]
        crop = img.crop(box) if box else img
        x = torch.from_numpy(np.asarray(crop.resize((224, 224), Image.BICUBIC), dtype=np.float32) / 255).permute(2, 0, 1)
        x = ((x - mean) / std)[None].to(device).half()
        tokens = dino(pixel_values=x).last_hidden_state[0, 1:].float()   # [256, 768] patch tokens (CLS dropped)
        feats[f"{k}|dino_patch"] = F.normalize(tokens, dim=-1).half().cpu().numpy()
        px = sig_proc(images=crop, return_tensors="pt")["pixel_values"].to(device).half()
        e = sig.get_image_features(pixel_values=px)
        e = e.pooler_output if hasattr(e, "pooler_output") else e
        feats[f"{k}|siglip"] = F.normalize(e.float(), dim=-1)[0].cpu().numpy()
        if (i + 1) % 64 == 0 or i + 1 == len(new):
            np.savez(npath, **feats)
            print(f"{i + 1}/{len(new)}", flush=True)
    print(json.dumps({"dir": str(out_dir), "new": len(new)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
