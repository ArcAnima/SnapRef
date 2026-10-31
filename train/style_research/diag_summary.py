"""Summarise diag_loss.py output: loss gains by sigma and band, style gap, artist-direction projection, attention.

    python train/style_research/diag_summary.py ~/style_agent/diag/loss.jsonl > ~/style_agent/diag/summary.json

Per condition X and sigma: mean MSE; relative change vs a reference condition in % (bootstrap 95% CI over pairs);
the same per frequency band (low / mid / fine). Gaps reported:
  text_gain      base - base_art         what Anima's own @artist tag removes from the loss (the style signal FM sees)
  v1a/p2s gain   noref - R               what the same-artist reference removes
  style gap      W - R                   right vs wrong artist reference (> 0: the reference's style is used)
  oracle gain    noref - T               the target itself as reference (upper bound of the path)
proj / cos: mean over pairs of <v_X - v_noref, d_art> / |d_art|^2 and cosine (total and fine band).
attn: per block, mean reference mass / entropy / locality over the stats pairs, for each tag and sigma.
"""
from __future__ import annotations

import json
import sys

import numpy as np


def boot_ci(x: np.ndarray, n: int = 2000, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    m = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return [round(float(v), 3) for v in np.percentile(m, [2.5, 97.5])]


def main(path: str) -> int:
    recs = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    sig = recs[0]["sigmas"]
    names = sorted({k for r in recs for k in r["loss"]})
    out = {"n_pairs": len(recs), "sigmas": sig, "mse": {}, "gains_pct": {}, "band_gains_pct": {}, "proj": {}, "cos": {},
           "proj_fine": {}, "cos_fine": {}}
    for nm in names:
        rs = [r for r in recs if nm in r["loss"]]
        out["mse"][nm] = [round(float(np.mean([r["loss"][nm][j] for r in rs])), 5) for j in range(len(sig))]
    comps = {"text_gain": ("base", "base_art"), "v1a_noref_vs_base": ("base", "v1a_noref"),
             "v1a_R_gain": ("v1a_noref", "v1a_R"), "v1a_style_gap": ("v1a_W", "v1a_R"), "v1a_oracle": ("v1a_noref", "v1a_T"),
             "v1a_cs_gain": ("v1a_noref", "v1a_C"),
             "p2s_R_gain": ("p2s_noref", "p2s_R"), "p2s_style_gap": ("p2s_W", "p2s_R"), "p2s_oracle": ("p2s_noref", "p2s_T"),
             "p2c_cs_gain": ("p2s_noref", "p2c_C")}
    for key, (a, b) in comps.items():
        rs = [r for r in recs if a in r["loss"] and b in r["loss"]]
        if not rs:
            continue
        rows, brows = [], []
        for j in range(len(sig)):
            x = np.array([100 * (r["loss"][a][j] - r["loss"][b][j]) / r["loss"][a][j] for r in rs])
            rows.append({"mean": round(float(x.mean()), 3), "ci95": boot_ci(x)})
            bb = []
            for band in range(3):
                y = np.array([100 * (r["band"][a][j][band] - r["band"][b][j][band]) / max(r["band"][a][j][band], 1e-12) for r in rs])
                bb.append(round(float(y.mean()), 3))
            brows.append(bb)
        out["gains_pct"][key] = {"n": len(rs), "by_sigma": rows}
        out["band_gains_pct"][key] = brows
    # band shares of the base error and of the artist direction energy
    out["base_band_share"] = [[round(float(np.mean([r["band"]["base"][j][b] / sum(r["band"]["base"][j]) for r in recs])), 3)
                               for b in range(3)] for j in range(len(sig))]
    out["d_art_rms"] = [round(float(np.mean([r["d_art_norm"][j] for r in recs])), 4) for j in range(len(sig))]
    for field in ("proj", "cos", "proj_fine", "cos_fine"):
        for nm in sorted({k for r in recs for k in r[field]}):
            rs = [r for r in recs if nm in r[field]]
            out[field][nm] = [round(float(np.median([r[field][nm][j] for r in rs])), 4) for j in range(len(sig))]
    att = [r for r in recs if "attn" in r]
    if att:
        out["attn"] = {}
        for tag in att[0]["attn"]:
            out["attn"][tag] = {}
            for s in att[0]["attn"][tag]:
                rows = [r["attn"][tag][s] for r in att]
                blocks = len(rows[0])
                mass = [round(float(np.mean([np.mean(rw[b][1]) for rw in rows])), 4) for b in range(blocks)]
                ent = [round(float(np.mean([np.mean(rw[b][2]) for rw in rows])), 3) for b in range(blocks)]
                loc = [round(float(np.mean([np.mean(rw[b][3]) for rw in rows])), 3) for b in range(blocks)]
                hmax = [round(float(np.mean([np.max(rw[b][1]) for rw in rows])), 3) for b in range(blocks)]
                out["attn"][tag][s] = {"mass": mass, "entropy": ent, "local": loc, "max_head_mass": hmax}
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
