"""imgutils features of the detail eval (run in ~/ccip-venv): WD tags, head box, CCIP of the full image and the head crop.

    ~/ccip-venv/bin/python train/detail_eval/features_imgutils.py --set ~/detail_eval/set            # ref + truth
    ~/ccip-venv/bin/python train/detail_eval/features_imgutils.py --set ~/detail_eval/set --renders ~/detail_eval/renders/v1a

Writes DIR/feat_imgutils.json (per image: general tags with p >= 0.1 from WD SwinV2_v3, head box from detect_heads, or
face box grown 1.6x when no head is found) and DIR/feat_ccip.npz (full / head CCIP features). Images already in the
outputs are skipped. Keys: "<case>/ref", "<case>/truth" for the set, "<case>_s<seed>" for renders.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import head_box, load_manifest  # noqa: E402


def items(set_dir: Path, renders: Path | None) -> dict[str, Path]:
    man = load_manifest(set_dir)
    if renders is None:
        return {f"{c['name']}/{k}": set_dir / c["name"] / f for c in man["cases"] for k, f in (("ref", "ref.png"), ("truth", "truth.jpg"))}
    return {f"{c['name']}_s{s}": renders / f"{c['name']}_s{s}.png" for c in man["cases"] for s in c["seeds"]}


def main() -> int:
    import onnxruntime
    onnxruntime.preload_dlls()   # see train/ccip_eval.py
    from imgutils.detect import detect_faces, detect_heads
    from imgutils.metrics import ccip_batch_extract_features
    from imgutils.tagging import get_wd14_tags
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", type=Path, required=True)
    ap.add_argument("--renders", type=Path)
    args = ap.parse_args()
    out_dir = args.renders or args.set
    todo = items(args.set, args.renders)
    missing = [k for k, p in todo.items() if not p.exists()]
    if missing:
        raise SystemExit(f"{len(missing)} images missing, e.g. {missing[0]}")
    jpath, npath = out_dir / "feat_imgutils.json", out_dir / "feat_ccip.npz"
    feats = json.loads(jpath.read_text(encoding="utf-8")) if jpath.exists() else {}
    ccip = dict(np.load(npath)) if npath.exists() else {}
    new = [k for k in todo if k not in feats]
    for i, k in enumerate(new):
        img = Image.open(todo[k]).convert("RGB")
        general = get_wd14_tags(img, general_threshold=0.1, fmt=("general",))[0]
        box = head_box(detect_heads(img), img.size)
        source = "head"
        if box is None:
            box, source = head_box(detect_faces(img), img.size, margin=0.8), "face"
        crop = img.crop(box) if box else img
        f_full, f_head = ccip_batch_extract_features([img, crop])
        feats[k] = {"tags": {t: round(float(p), 4) for t, p in general.items()}, "head_box": box,
                    "head_source": source if box else None, "size": img.size}
        ccip[f"{k}|full"], ccip[f"{k}|head"] = f_full, f_head
        if (i + 1) % 32 == 0 or i + 1 == len(new):
            jpath.write_text(json.dumps(feats), encoding="utf-8")
            np.savez(npath, **ccip)
            print(f"{i + 1}/{len(new)}", flush=True)
    print(json.dumps({"dir": str(out_dir), "images": len(todo), "new": len(new),
                      "no_head": sum(feats[k]["head_box"] is None for k in todo)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
