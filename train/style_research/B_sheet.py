"""Experiment B: style sheet (PARALLEL.md §3): reference | method renders ... | another post of the artist, scene 0.

    python train/style_research/B_sheet.py --out SHEET.jpg --methods v1a=DIR,baseline=DIR,B=DIR [--n 12]

DIR holds eval_style_v2.py renders <artist>__s<k>.png; artists missing from a DIR get a blank panel.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=Path, default=Path.home() / "eval" / "style_eval" / "cases.json")
    ap.add_argument("--methods", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--scene", type=int, default=0)
    args = ap.parse_args()
    root = args.cases.parent.parent
    cases = json.loads(args.cases.read_text(encoding="utf-8"))["cases"][: args.n]
    methods = [m.split("=", 1) for m in args.methods.split(",")]
    th, pad = 300, 4
    labels = ["reference"] + [k for k, _ in methods] + ["same artist"]
    rows = []
    for c in cases:
        ims = [Image.open(root / c["ref"]["path"]).convert("RGB")]
        for _, d in methods:
            p = Path(d) / f"{c['artist']}__s{args.scene}.png"
            ims.append(Image.open(p).convert("RGB") if p.exists() else Image.new("RGB", (205, th), "gray"))
        ims.append(Image.open(root / c["style_targets"][0]["path"]).convert("RGB"))
        rows.append([im.resize((max(1, round(im.width * th / im.height)), th)) for im in ims])
    col_w = [max(r[k].width for r in rows) for k in range(len(labels))]
    W = sum(w + pad for w in col_w)
    sheet = Image.new("RGB", (W, 24 + (th + pad) * len(rows)), "white")
    draw = ImageDraw.Draw(sheet)
    x = 0
    for k, lab in enumerate(labels):
        draw.text((x + 4, 4), lab, fill="black")
        x += col_w[k] + pad
    for i, (c, r) in enumerate(zip(cases, rows)):
        x = 0
        for k, im in enumerate(r):
            sheet.paste(im, (x, 24 + i * (th + pad)))
            x += col_w[k] + pad
        draw.text((4, 24 + i * (th + pad) + th - 16), c["artist"], fill="red")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.out, quality=82)
    print(f"SHEET {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
