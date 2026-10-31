"""Detail-eval scores, controls, paired model differences with case-bootstrap CIs, worst-case sheets (~/ccip-venv).

    ~/ccip-venv/bin/python train/detail_eval/score.py --set ~/detail_eval/set --renders ~/detail_eval/renders \
        --models noref v1a v2a_001000 v2a_005000 --base v1a --out ~/detail_eval/results/first
    python train/detail_eval/score.py --self-check

Needs feat_imgutils.json / feat_ccip.npz / feat_torch.npz in the set dir and in every renders/<model> dir.
Per image (output O of case c, seed s), with R = reference, T = truth (synthetic: the generator's image of another
scene; real: another post of the same artist and character):
  tag_recall   share of K = look(R) & look(T) that O shows (WD p >= 0.35). Look tags: hair, eyes, accessories,
               clothing items, persistent body marks (common.look_category on the taxonomy slot).
  tag_false    share of O's look tags that R does not show even weakly (p < 0.1). Lower is better.
  head_siglip  cos of SigLIP2 so400m embeddings, head crop of O vs head crop of T.
  head_dino    DINOv2 patch matching of the head crops: mean over patches of the best cos in the other crop, both ways.
  head_ccip    CCIP difference of the head crops (lower = more alike).
  full_ccip    CCIP difference of the full images (lower = more alike; the old metric, kept as a reference).
Controls: the same metric against the truth (and K, R for the tag metrics) of another case of the same kind
("other": next case of that kind in manifest order, cyclic). ceiling: the metric for T vs R.
A case's value is the mean over its seeds; CIs resample cases (4,000 bootstrap draws).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import load_manifest, look_category  # noqa: E402

ON, OFF = 0.35, 0.1
LOWER_BETTER = {"tag_false", "head_ccip", "full_ccip"}
METRICS = ["tag_recall", "tag_false", "head_siglip", "head_dino", "head_ccip", "full_ccip"]
KINDS = ["synthetic", "real", "all"]


def taxonomy_slots(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8") as f:
        return {r["tag"]: r["slot"] for r in csv.DictReader(f) if r["danbooru_category"] == "general"}


def look(tags: dict[str, float], slots: dict[str, str], th: float = ON) -> set[str]:
    return {t for t, p in tags.items() if p >= th and look_category(t, slots.get(t))}


def tag_scores(o: dict, r: dict, t: dict, slots: dict) -> dict:
    """o, r, t: WD tag -> prob of output, reference, truth."""
    k = look(r, slots) & look(t, slots)
    o_on = look(o, slots)
    r_weak = {x for x, p in r.items() if p >= OFF}
    res = {"n_keep": len(k), "tag_false": len(o_on - r_weak) / max(1, len(o_on))}
    if k:
        res["tag_recall"] = len(k & o_on) / len(k)
        res["misses"] = sorted(k - o_on)
    return res


def chamfer(a: np.ndarray, b: np.ndarray) -> float:
    s = a.astype(np.float32) @ b.astype(np.float32).T
    return float((s.max(1).mean() + s.max(0).mean()) / 2)


def bootstrap(per_case: np.ndarray, n: int = 4000, seed: int = 0) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    means = per_case[rng.integers(0, len(per_case), size=(n, len(per_case)))].mean(1)
    return float(per_case.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


class Feats:
    def __init__(self, d: Path):
        self.tags = json.loads((d / "feat_imgutils.json").read_text(encoding="utf-8"))
        self.ccip = dict(np.load(d / "feat_ccip.npz"))
        self.torch = dict(np.load(d / "feat_torch.npz"))


def ccip_matrix(keys_a: list, fa: Feats, keys_b: list, fb: Feats, part: str) -> np.ndarray:
    from imgutils.metrics import ccip_batch_differences
    feats = [fa.ccip[f"{k}|{part}"] for k in keys_a] + [fb.ccip[f"{k}|{part}"] for k in keys_b]
    return ccip_batch_differences(feats)[: len(keys_a), len(keys_a):]


def score_model(fo: Feats, fs: Feats, cases: list, other: dict, slots: dict) -> list[dict]:
    """Rows per (case, seed) with metrics vs truth, vs reference and vs another case's truth."""
    okeys = [f"{c['name']}_s{s}" for c in cases for s in c["seeds"]]
    skeys = [f"{c['name']}/{x}" for c in cases for x in ("ref", "truth")]
    col = {k: i for i, k in enumerate(skeys)}
    ccip = {p: ccip_matrix(okeys, fo, skeys, fs, p) for p in ("full", "head")}
    rows, i = [], 0
    for c in cases:
        o_c = other[c["name"]]
        for s in c["seeds"]:
            ko = okeys[i]
            r = {"case": c["name"], "kind": c["kind"], "seed": s}
            for tag, against in (("", c["name"]), ("_other", o_c)):
                kt, kr = f"{against}/truth", f"{against}/ref"
                ts = tag_scores(fo.tags[ko]["tags"], fs.tags[kr]["tags"], fs.tags[kt]["tags"], slots)
                r["tag_false" + tag] = ts["tag_false"]
                if "tag_recall" in ts:
                    r["tag_recall" + tag] = ts["tag_recall"]
                if not tag:
                    r["n_keep"], r["misses"] = ts["n_keep"], ts.get("misses", [])
                r["head_siglip" + tag] = float(fo.torch[f"{ko}|siglip"] @ fs.torch[f"{kt}|siglip"])
                r["head_dino" + tag] = chamfer(fo.torch[f"{ko}|dino_patch"], fs.torch[f"{kt}|dino_patch"])
                r["head_ccip" + tag] = float(ccip["head"][i, col[kt]])
                r["full_ccip" + tag] = float(ccip["full"][i, col[kt]])
            kr = f"{c['name']}/ref"
            r["head_siglip_ref"] = float(fo.torch[f"{ko}|siglip"] @ fs.torch[f"{kr}|siglip"])
            r["head_ccip_ref"] = float(ccip["head"][i, col[kr]])
            r["full_ccip_ref"] = float(ccip["full"][i, col[kr]])
            r["no_head"] = fo.tags[ko]["head_box"] is None
            rows.append(r)
            i += 1
    return rows


def per_case(rows: list[dict], key: str, names: list[str]) -> np.ndarray:
    """Mean over seeds per case, NaN where the metric is undefined (e.g. no shared look tags)."""
    out = []
    for n in names:
        v = [r[key] for r in rows if r["case"] == n and key in r]
        out.append(np.mean(v) if v else np.nan)
    return np.array(out)


def ci(v: np.ndarray, seed: int = 0) -> dict:
    v = v[~np.isnan(v)]
    m, lo, hi = bootstrap(v, seed=seed)
    return {"mean": round(m, 4), "lo": round(lo, 4), "hi": round(hi, 4), "n": int(len(v))}


def seed_split(rows: list[dict], key: str, names: list[str]) -> np.ndarray:
    """Per case: seed 0 minus seed 1 (a model compared with itself; a null difference)."""
    out = []
    for n in names:
        v = {r["seed"]: r[key] for r in rows if r["case"] == n and key in r}
        s = sorted(v)
        out.append(v[s[0]] - v[s[1]] if len(s) >= 2 else np.nan)
    return np.array(out)


def ceiling(fs: Feats, cases: list, slots: dict) -> list[dict]:
    keys = [f"{c['name']}/truth" for c in cases]
    refs = [f"{c['name']}/ref" for c in cases]
    hc, fc = ccip_matrix(keys, fs, refs, fs, "head"), ccip_matrix(keys, fs, refs, fs, "full")
    rows = []
    for i, c in enumerate(cases):
        kt, kr = keys[i], refs[i]
        lr, lt = look(fs.tags[kr]["tags"], slots), look(fs.tags[kt]["tags"], slots)
        rows.append({"case": c["name"], "kind": c["kind"], "n_keep": len(lr & lt),
                     "look_ref": len(lr), "look_jaccard": len(lr & lt) / max(1, len(lr | lt)),
                     "head_siglip": float(fs.torch[f"{kt}|siglip"] @ fs.torch[f"{kr}|siglip"]),
                     "head_dino": chamfer(fs.torch[f"{kt}|dino_patch"], fs.torch[f"{kr}|dino_patch"]),
                     "head_ccip": float(hc[i, i]), "full_ccip": float(fc[i, i])})
    return rows


def sheet(rows: list[dict], model: str, cases: dict, set_dir: Path, renders: Path, fs: Feats, fo: Feats, path: Path,
          key: str = "head_siglip", n: int = 12) -> None:
    """Worst n cases of a model by key (per-case mean): ref | ref head | output | output head | truth | truth head."""
    from PIL import Image, ImageDraw
    names = list(dict.fromkeys(r["case"] for r in rows))
    vals = per_case(rows, key, names)
    order = [names[i] for i in np.argsort(vals if key not in LOWER_BETTER else -vals) if not np.isnan(vals[i])][:n]
    th, out_rows = 256, []

    def crop(img, feats, k):
        box = feats.tags[k]["head_box"]
        return img.crop(box) if box else img

    for name in order:
        rs = [r for r in rows if r["case"] == name]
        r = min(rs, key=lambda x: x[key] if key not in LOWER_BETTER else -x[key])
        ko = f"{name}_s{r['seed']}"
        ref, truth = Image.open(set_dir / name / "ref.png").convert("RGB"), Image.open(set_dir / name / "truth.jpg").convert("RGB")
        out = Image.open(renders / model / f"{ko}.png").convert("RGB")
        ims = [ref, crop(ref, fs, f"{name}/ref"), out, crop(out, fo, ko), truth, crop(truth, fs, f"{name}/truth")]
        ims = [im.resize((max(1, round(im.width * th / im.height)), th)) for im in ims]
        row = Image.new("RGB", (sum(im.width + 4 for im in ims), th + 34), "white")
        d = ImageDraw.Draw(row)
        d.text((4, 2), f"{name} s{r['seed']} {model}  siglip {r['head_siglip']:.3f}  recall {r.get('tag_recall', float('nan')):.2f}  "
                        f"headccip {r['head_ccip']:.3f}  missed: {', '.join(r['misses'][:8])}", fill="black")
        d.text((4, 16), "reference | ref head | output | output head | truth | truth head", fill="grey")
        x = 0
        for im in ims:
            row.paste(im, (x, 34)); x += im.width + 4
        out_rows.append(row)
    if not out_rows:
        return
    W = max(r.width for r in out_rows)
    img = Image.new("RGB", (W, sum(r.height + 6 for r in out_rows)), "white")
    y = 0
    for r in out_rows:
        img.paste(r, (0, y)); y += r.height + 6
    img.save(path, quality=85)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", type=Path, required=True)
    ap.add_argument("--renders", type=Path, required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--base", default="v1a", help="paired differences are model - base")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--taxonomy", type=Path, default=Path.home() / "hf" / "taxonomy" / "tags.csv")
    ap.add_argument("--sheets", nargs="*", default=None, help="models to draw worst-case sheets for (default: all)")
    args = ap.parse_args()
    import onnxruntime
    onnxruntime.preload_dlls()
    man = load_manifest(args.set)
    cases = man["cases"]
    names = [c["name"] for c in cases]
    other = {}
    for kind in ("synthetic", "real"):
        ks = [c["name"] for c in cases if c["kind"] == kind]
        other.update({a: b for a, b in zip(ks, ks[1:] + ks[:1])})
    slots = taxonomy_slots(args.taxonomy)
    fs = Feats(args.set)
    args.out.mkdir(parents=True, exist_ok=True)
    rows, feats = {}, {}
    for m in args.models:
        feats[m] = Feats(args.renders / m)
        rows[m] = score_model(feats[m], fs, cases, other, slots)
    ceil = ceiling(fs, cases, slots)
    sel = {k: [n for n, c in zip(names, cases) if k == "all" or c["kind"] == k] for k in KINDS}
    res = {"set": str(args.set), "models": args.models, "base": args.base, "n_cases": len(cases),
           "seeds": len(cases[0]["seeds"]), "absolute": {}, "control": {}, "seed_null": {}, "paired": {}, "ceiling": {},
           "rank_own_vs_other": {}}
    for kind in KINDS:
        nm = sel[kind]
        res["ceiling"][kind] = {k: ci(per_case(ceil, k, nm)) for k in ("n_keep", "look_jaccard", "head_siglip", "head_dino", "head_ccip", "full_ccip")}
        for m in args.models:
            res["absolute"].setdefault(m, {})[kind] = {k: ci(per_case(rows[m], k, nm)) for k in METRICS + ["head_siglip_ref", "head_ccip_ref", "full_ccip_ref"]}
            res["control"].setdefault(m, {})[kind] = {k: ci(per_case(rows[m], k + "_other", nm)) for k in METRICS}
            res["seed_null"].setdefault(m, {})[kind] = {k: ci(seed_split(rows[m], k, nm)) for k in METRICS}
            # own minus other, per case: how far a wrong character falls (sign: positive = own is better)
            res["rank_own_vs_other"].setdefault(m, {})[kind] = {
                k: ci((per_case(rows[m], k, nm) - per_case(rows[m], k + "_other", nm)) * (-1 if k in LOWER_BETTER else 1))
                for k in METRICS}
            if m != args.base and args.base in rows:
                res["paired"].setdefault(m, {})[kind] = {
                    k: ci(per_case(rows[m], k, nm) - per_case(rows[args.base], k, nm)) for k in METRICS}
        # every pair of models, for the validation order checks
    res["paired_all"] = {f"{a}-{b}": {kind: {k: ci(per_case(rows[a], k, sel[kind]) - per_case(rows[b], k, sel[kind])) for k in METRICS}
                                     for kind in KINDS} for i, a in enumerate(args.models) for b in args.models[i + 1:]}
    (args.out / "scores.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    (args.out / "rows.json").write_text(json.dumps({"rows": rows, "ceiling": ceil}), encoding="utf-8")
    for m in (args.sheets if args.sheets is not None else args.models):
        for kind in ("synthetic", "real"):
            sheet([r for r in rows[m] if r["kind"] == kind], m, {}, args.set, args.renders, fs, feats[m],
                  args.out / f"worst_{m}_{kind}.jpg")
    (args.out / "table.md").write_text(table(res), encoding="utf-8")
    print(table(res))
    return 0


def fmt(d: dict) -> str:
    return f"{d['mean']:.3f} [{d['lo']:.3f}, {d['hi']:.3f}]"


def table(res: dict) -> str:
    lines = []
    for kind in KINDS:
        lines += [f"\n### {kind}: absolute (mean [95% CI], per case = mean of seeds)\n",
                  "| model | " + " | ".join(METRICS) + " |", "|---|" + "---:|" * len(METRICS)]
        for m in res["models"]:
            lines.append(f"| {m} | " + " | ".join(fmt(res["absolute"][m][kind][k]) for k in METRICS) + " |")
        lines.append("| other-character control (" + res["models"][-1] + ") | " +
                     " | ".join(fmt(res["control"][res["models"][-1]][kind][k]) for k in METRICS) + " |")
        lines.append("| truth vs reference (ceiling) | | | " + " | ".join(fmt(res["ceiling"][kind][k]) for k in METRICS[2:]) + " |")
        lines += [f"\n### {kind}: paired difference vs {res['base']} (model - base)\n",
                  "| model | " + " | ".join(METRICS) + " |", "|---|" + "---:|" * len(METRICS)]
        for m, d in res["paired"].items():
            lines.append(f"| {m} | " + " | ".join(fmt(d[kind][k]) for k in METRICS) + " |")
        lines.append(f"| seed 0 - seed 1 ({res['base']}) | " + " | ".join(fmt(res["seed_null"][res["base"]][kind][k]) for k in METRICS) + " |")
    return "\n".join(lines) + "\n"


def self_check() -> None:
    slots = {"long_hair": "appearance:hair", "red_eyes": "appearance:eyes", "hat": "clothing:headwear",
             "smile": "pose:expression", "closed_eyes": "appearance:eyes", "blue_hair": "appearance:hair"}
    r = {"long_hair": 0.9, "red_eyes": 0.8, "hat": 0.5, "smile": 0.9}
    t = {"long_hair": 0.7, "red_eyes": 0.6, "closed_eyes": 0.9}   # hat not visible in the truth: not counted
    o = {"long_hair": 0.8, "red_eyes": 0.2, "blue_hair": 0.6, "hat": 0.05}
    s = tag_scores(o, r, t, slots)
    assert s["n_keep"] == 2 and s["tag_recall"] == 0.5 and s["misses"] == ["red_eyes"], s
    assert s["tag_false"] == 0.5, s   # blue_hair is new; long_hair is in R
    assert "tag_recall" not in tag_scores(o, {"smile": 1.0}, t, slots)
    a = np.eye(4, dtype=np.float32)
    assert abs(chamfer(a, a) - 1) < 1e-6 and abs(chamfer(a[:2], a[2:])) < 1e-6
    m, lo, hi = bootstrap(np.array([1.0, 2.0, 3.0, 4.0]))
    assert m == 2.5 and lo < 2.5 < hi
    rows = [{"case": "a", "seed": 1, "x": 1.0}, {"case": "a", "seed": 2, "x": 3.0}, {"case": "b", "seed": 1, "x": 5.0}]
    pc = per_case(rows, "x", ["a", "b", "c"])
    assert pc[0] == 2.0 and pc[1] == 5.0 and np.isnan(pc[2])
    ss = seed_split(rows, "x", ["a", "b"])
    assert ss[0] == -2.0 and np.isnan(ss[1])
    print("score self-check ok")


if __name__ == "__main__":
    if "--self-check" in sys.argv:
        self_check()
        raise SystemExit(0)
    raise SystemExit(main())
