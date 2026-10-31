"""CCIP character-identity scores for an A_eval_identity.py output (52 cases), from its saved panels.

    ~/ccip-venv/bin/python train/ccip_eval.py --panels RUN/eval_005000/identity --cases ~/eval/v1ab --out ccip.json

CCIP (deepghs/imgutils, ccip-caformer-24-randaug-pruned) is trained to tell whether two single-character anime images
show the same character, across pose, outfit and artist. Lower difference = more alike; below the model threshold
(0.178) counts as "same character". The generated image is cut back out of the panel row (reference | output | truth,
each resized to 512 px high by sample.labeled_row); reference and truth come from the full-size case files.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

PANEL_HEIGHT, LABEL, GAP = 512, 28, 8   # sample.labeled_row


def cut_output(panel: Image.Image, ref_size: tuple[int, int], out_size: tuple[int, int], has_truth: bool) -> Image.Image:
    w = lambda s: round(s[0] * PANEL_HEIGHT / s[1])   # noqa: E731
    x0 = w(ref_size) + GAP
    expected = x0 + w(out_size) + (GAP + w(out_size) if has_truth else 0)
    assert panel.size == (expected, PANEL_HEIGHT + LABEL), (panel.size, expected)
    return panel.crop((x0, LABEL, x0 + w(out_size), LABEL + PANEL_HEIGHT))


def main() -> int:
    import onnxruntime
    onnxruntime.preload_dlls()   # pip CUDA/cuDNN wheels; without this the CUDA provider silently falls back to CPU
    from imgutils.metrics import ccip_batch_extract_features, ccip_default_threshold, ccip_difference
    ap = argparse.ArgumentParser()
    ap.add_argument("--panels", type=Path, required=True)
    ap.add_argument("--cases", type=Path, default=Path.home() / "eval" / "v1ab")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    th = ccip_default_threshold()
    cases = json.loads((args.cases / "summary.json").read_text(encoding="utf-8"))["cases"][:52]
    imgs, rows = [], []
    for c in cases:
        cd = args.cases / c["name"]
        ref = Image.open(cd / "ref.png").convert("RGB")
        truth = Image.open(cd / "truth.jpg").convert("RGB") if (cd / "truth.jpg").exists() else None
        out = cut_output(Image.open(args.panels / f"{c['name']}.jpg").convert("RGB"), ref.size,
                         (int(c["width"]), int(c["height"])), truth is not None)
        rows.append({"case": c["name"], "kind": c["name"].split("_")[0].replace("held", "held_out")})
        imgs += [out, ref, truth or ref]
    feats = ccip_batch_extract_features(imgs)
    for i, r in enumerate(rows):
        f_out, f_ref, f_truth = feats[3 * i], feats[3 * i + 1], feats[3 * i + 2]
        r["diff_ref"] = float(ccip_difference(f_out, f_ref))
        if r["kind"] != "ext0" and r["kind"] != "ext1":
            r["diff_truth"] = float(ccip_difference(f_out, f_truth))
            r["diff_truth_vs_ref"] = float(ccip_difference(f_truth, f_ref))   # how far the truth itself is from the ref

    held = [i for i, r in enumerate(rows) if r["kind"] == "held_out"]
    for a, b in zip(held, held[1:] + held[:1]):
        rows[a]["diff_other_truth"] = float(ccip_difference(feats[3 * a], feats[3 * b + 2]))

    def agg(sel, key):
        v = [r[key] for r in rows if sel(r) and key in r]
        return {"mean_diff": round(sum(v) / len(v), 4), "same_rate": round(sum(x < th for x in v) / len(v), 4), "n": len(v)}
    res = {"model": "ccip-caformer-24-randaug-pruned", "threshold": th,
           "held_out": agg(lambda r: r["kind"] == "held_out", "diff_truth"),
           "seen": agg(lambda r: r["kind"] == "seen", "diff_truth"),
           "external_ref": agg(lambda r: r["kind"].startswith("ext"), "diff_ref"),
           "truth_vs_ref_held_out": agg(lambda r: r["kind"] == "held_out", "diff_truth_vs_ref"),
           "other_character_held_out": agg(lambda r: r["kind"] == "held_out", "diff_other_truth"),
           "rows": rows}
    args.out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("held_out", "seen", "external_ref", "truth_vs_ref_held_out", "other_character_held_out")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
