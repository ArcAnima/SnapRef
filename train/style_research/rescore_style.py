"""Score style-eval renders with the learned style metric (style_metric.StyleDL) instead of CSD / MegaStyle.

    python train/style_research/rescore_style.py --renders DIR[,DIR...] --out OUT.json [--baseline noref]

Renders: files <artist>__s<k>.png in method folders (eval_style_v2.py layout: <run>/img/<method>/...).
Gallery per artist: the eval case's style_targets (other characters) plus the extended posts of
--extra (build_style_eval.py --per-artist 13 output; the reference post itself is never in the gallery).
Per render: style = mean cos to the artist's gallery - mean cos to the other artists' galleries; rank of the artist
among the 33 by mean cos (1 = best); ref_cos = cos to the reference image (style space).
Per method: means, top-1 / top-5 share, and the paired difference to --baseline (mean over the 2 scenes per artist,
bootstrap over artists, 95% CI). The reference images themselves are scored as method "reference" (ceiling).
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from style_metric import StyleDL, artist_scores, paired_ci  # noqa: E402


def galleries(m: StyleDL, cases_path: Path, extra: Path | None) -> tuple[dict, dict, list]:
    sd = json.loads(cases_path.read_text(encoding="utf-8"))
    root = cases_path.parent.parent
    gal, refs = {}, {}
    for c in sd["cases"]:
        a = c["artist"]
        ref_id = str(c["ref"]["id"])
        paths = {Path(p["path"]).name: root / p["path"] for p in c["style_targets"]}
        if extra is not None and (extra / a).exists():
            for p in sorted((extra / a).glob("*.webp")):
                paths.setdefault(p.name, p)
        paths.pop(f"{ref_id}.webp", None)
        gal[a] = m.emb_many([Image.open(p).convert("RGB") for p in paths.values()])
        refs[a] = m.emb(Image.open(root / c["ref"]["path"]).convert("RGB"))
    return gal, refs, sd["cases"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--renders", required=True, help="comma-separated method folders")
    ap.add_argument("--cases", type=Path, default=Path.home() / "eval" / "style_eval" / "cases.json")
    ap.add_argument("--extra", type=Path, default=Path.home() / "style_agent" / "style_eval_ext")
    ap.add_argument("--probe", type=Path, default=Path.home() / "style_agent" / "probe" / "probe_dinov2_linear.pt")
    ap.add_argument("--baseline", default="noref")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    m = StyleDL(args.probe)
    gal, refs, cases = galleries(m, args.cases, args.extra if args.extra.exists() else None)
    rows = []
    for a in gal:   # ceiling: the reference post itself
        s = artist_scores(refs[a], a, gal)
        rows.append({"method": "reference", "artist": a, "scene": 0, **s, "ref_cos": 1.0})
    for d in args.renders.split(","):
        d = Path(d)
        files = sorted(d.glob("*__s*.png"))
        embs = m.emb_many([Image.open(f).convert("RGB") for f in files])
        for f, e in zip(files, embs):
            a, s_ = f.stem.rsplit("__s", 1)
            if a not in gal:
                continue
            rows.append({"method": d.name, "artist": a, "scene": int(s_), **artist_scores(e, a, gal),
                         "ref_cos": float(e @ refs[a])})
    by = collections.defaultdict(list)
    for r in rows:
        by[r["method"]].append(r)
    summary = {}
    base = {(r["artist"]): [] for r in by.get(args.baseline, [])}
    for r in by.get(args.baseline, []):
        base[r["artist"]].append(r["style"])
    for meth, rs in by.items():
        summary[meth] = {"style": round(float(np.mean([r["style"] for r in rs])), 4),
                         "top1": round(float(np.mean([r["rank"] == 1 for r in rs])), 3),
                         "top5": round(float(np.mean([r["rank"] <= 5 for r in rs])), 3),
                         "ref_cos": round(float(np.mean([r["ref_cos"] for r in rs])), 4), "n": len(rs)}
        if base and meth != args.baseline and meth != "reference":
            per = collections.defaultdict(list)
            for r in rs:
                per[r["artist"]].append(r["style"])
            arts = [a for a in per if a in base]
            diff = np.array([np.mean(per[a]) - np.mean(base[a]) for a in arts])
            summary[meth][f"diff_vs_{args.baseline}"] = round(float(diff.mean()), 4)
            summary[meth]["ci95"] = paired_ci(diff, arts)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": summary, "rows": rows,
                                    "gallery_sizes": {a: int(g.shape[0]) for a, g in gal.items()}}, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
