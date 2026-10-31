"""Multiple references of ONE character with the v1a adapter, without retraining (2026-10-31).

    python train/multiref_test.py --adapter ~/adapters/v1a/snapref_v1a.safetensors --index ~/multiref/eval_index.json \
        --images ~/multiref/images --out ~/multiref/out [--groups 64] [--self-check]

Per held-out group (asym_prep eval split: 32 characters the adapter never saw, same character in every image):
reference A = an upper body / portrait / cowboy shot, reference B = a full-body image, target T = another image
of the group (prefer one that is neither A nor B's framing). The target prompt is T's "stripped" prompt (no look
tags, as in A_eval_identity.py); the reference text is each reference's "ref_caption" stream like training.

Conditions (same seed per group):
  noref        no reference (lower bound)
  A, B         one reference each (the current node)
  stitch       A | B side by side as one reference, ~1 MP total (what a user can do today with Image Stitch)
  multi        A and B encoded separately, per-block K/V concatenated: target reads [K_A; K_B; K_t]
  multi_bias   multi + log-scale bias -log(2) on the reference keys (keeps the total reference attention mass
               about that of one reference)
Scores (written to scores.json by --score, run with the ccip venv): CCIP difference to T (lower = same character),
DINOv2 CLS cos to T, WD-tagger head count ("2girls"/"multiple girls" = the model drew two people).
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

from PIL import Image

try:   # the render path needs torch; --score runs in the CCIP venv without it
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = F = None

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

UPPER = ("upper body", "portrait", "cowboy shot")
CONDITIONS = ("noref", "A", "B", "stitch", "multi", "multi_bias")
_BIAS = [False]   # set only around the positive-branch call of a multi_bias sample


def pick_cases(index: dict, n_groups: int, seed: int = 0) -> list[dict]:
    """One (A, B, T) per group: A upper-ish, B full body, T a third image (prefer full body ≠ B, else any)."""
    rec, cases = index["records"], []
    for gid in sorted(index["groups"]):
        ids = index["groups"][gid]["record_ids"]
        ups = [i for i in ids if rec[i]["framing"] in UPPER]
        fulls = [i for i in ids if rec[i]["framing"].startswith("full body")]
        if not ups or not fulls:
            continue
        a, b = ups[0], fulls[0]
        rest = [i for i in ids if i not in (a, b)]
        if not rest:
            continue
        t = next((i for i in rest if rec[i]["framing"].startswith("full body")), rest[0])
        cases.append({"group": gid, "character": index["groups"][gid]["character"], "A": a, "B": b, "T": t,
                      "width": rec[t]["width"], "height": rec[t]["height"], "seed": 1000 + len(cases)})
    random.Random(seed).shuffle(cases)
    # spread over characters first: at most ceil(n / n_chars) groups per character
    by_char: dict[str, list] = {}
    for c in cases:
        by_char.setdefault(c["character"], []).append(c)
    out = []
    while len(out) < n_groups and any(by_char.values()):
        for ch in sorted(by_char):
            if by_char[ch] and len(out) < n_groups:
                out.append(by_char[ch].pop(0))
    return out


def stitch(a: Image.Image, b: Image.Image) -> Image.Image:
    """Core Image Stitch (right, match size): B resized to A's height, white background."""
    h = a.height
    b2 = b.resize((round(b.width * h / b.height), h), Image.LANCZOS)
    out = Image.new("RGB", (a.width + b2.width, h), "white")
    out.paste(a, (0, 0))
    out.paste(b2, (a.width, 0))
    return out


def concat_caches(caches: list[list]) -> list:
    return [(torch.cat([c[i][0] for c in caches], dim=1), torch.cat([c[i][1] for c in caches], dim=1))
            for i in range(len(caches[0]))]


def biased_attention_install(dm, n_ref_keys: list, bias: float):
    """While _BIAS[0] is set, the target stream's attention adds `bias` to the logits of the first n_ref_keys[0]
    keys (the prepended reference K). Same math as comfyui_nodes/snapref._biased_attention (strength).
    Returns an uninstall function."""
    saved = []
    for block in dm.blocks:
        attn = block.self_attn
        orig = attn.compute_attention

        def ca(q, k, v, transformer_options={}, _attn=attn, _orig=orig):
            if not _BIAS[0] or k.shape[1] == q.shape[1]:   # no bias wanted / no reference prepended
                return _orig(q, k, v, transformer_options=transformer_options)
            B, S, H, D = q.shape
            kk, vv = k.transpose(1, 2), v.transpose(1, 2)
            n, n_r = kk.shape[2], n_ref_keys[0]
            pad = (-n) % 16
            if pad:
                kk, vv = F.pad(kk, (0, 0, 0, pad)), F.pad(vv, (0, 0, 0, pad))
            mask = torch.zeros(1, 1, 1, n + pad, device=q.device, dtype=q.dtype)
            mask[..., :n_r] = bias
            if pad:
                mask[..., n:] = float("-inf")
            out = F.scaled_dot_product_attention(q.transpose(1, 2), kk, vv, attn_mask=mask.expand(B, H, S, n + pad))
            return _attn.output_dropout(_attn.output_proj(out.transpose(1, 2).reshape(B, S, H * D)))
        saved.append(orig)
        attn.compute_attention = ca

    def uninstall():
        for block, o in zip(dm.blocks, saved):
            block.self_attn.compute_attention = o
    return uninstall


def self_check() -> None:
    idx = {"groups": {f"g{i}": {"character": f"c{i % 3}", "record_ids": [f"g{i}_0", f"g{i}_1", f"g{i}_2"]} for i in range(6)},
           "records": {}}
    for i in range(6):
        for j, fr in enumerate(("upper body", "full body", "full body, wide shot")):
            idx["records"][f"g{i}_{j}"] = {"framing": fr, "width": 832, "height": 1216}
    cs = pick_cases(idx, 4)
    assert len(cs) == 4 and all(c["A"].endswith("_0") and c["B"].endswith("_1") and c["T"].endswith("_2") for c in cs)
    assert len({c["character"] for c in cs[:3]}) == 3, "first round must cover every character"
    s = stitch(Image.new("RGB", (100, 200)), Image.new("RGB", (50, 50)))
    assert s.size == (300, 200)
    assert is_look("long_hair") and is_look("black_thighhighs") and is_look("hair_ribbon")
    assert not is_look("looking_at_viewer") and not is_look("hand_in_own_hair") and not is_look("closed_eyes")
    if torch is None:
        print("self-check ok (no torch: cache concat skipped)"); return
    a = [(torch.ones(1, 3, 2, 4), torch.ones(1, 3, 2, 4))]
    c = concat_caches([a, a])
    assert c[0][0].shape == (1, 6, 2, 4)
    print("self-check ok")


def render(args) -> None:
    import anima_env
    import asym
    from eval_pilot import SHIFT, load_ref
    from sample import to_pil
    index = json.loads(args.index.read_text(encoding="utf-8"))
    cases = pick_cases(index, args.groups)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "cases.json").write_text(json.dumps(cases, indent=1), encoding="utf-8")
    device = torch.device("cuda", 0)
    cfg = json.loads((args.adapter.parent / "training_config.json").read_text(encoding="utf-8"))
    anima_env.setup_comfy(args.comfy_root)
    _, dm = anima_env.load_diffusion_model(cfg["comfy"]["unet_name"], device)
    modules = asym.inject(dm, cfg["adapter"]["lora"])
    from safetensors import safe_open
    with safe_open(str(args.adapter), "pt") as h:
        modes = list(json.loads((h.metadata() or {}).get("anima_ref_kv", "{}")).get("modes", ["cs", "char"]))
    mode_embed = torch.zeros(len(modes), dm.model_channels, device=device)
    asym.load(modules, args.adapter, {"mode_embed": mode_embed})
    mode_emb = mode_embed[modes.index("cs")][None]
    clip = anima_env.load_text_encoder(cfg["comfy"]["clip_name"])
    vae = anima_env.load_vae(cfg["comfy"]["vae_name"])
    lf = anima_env.latent_format()
    P, R = index["prompts"], index["records"]
    enc_cache: dict[str, torch.Tensor] = {}

    def enc(t):
        if t not in enc_cache:
            enc_cache[t] = anima_env.encode_prompt(clip, dm, t, device)
        return enc_cache[t]

    def prompt(rid, name):
        return P[str(R[rid]["prompts"][name])]

    def img(rid):
        return Image.open(args.images / R[rid]["image"]).convert("RGB")

    n_ref = [0]
    uninstall = biased_attention_install(dm, n_ref, -math.log(2))
    started = time.perf_counter()
    for ci, c in enumerate(cases):
        d = args.out / c["group"]
        d.mkdir(exist_ok=True)
        A, B, T = img(c["A"]), img(c["B"]), img(c["T"])
        A.save(d / "ref_A.jpg", quality=92); B.save(d / "ref_B.jpg", quality=92); T.save(d / "truth.jpg", quality=92)
        shape = (1, 16, 1, int(c["height"]) // 8, int(c["width"]) // 8)
        pos, neg = enc(prompt(c["T"], "stripped")), enc(prompt(c["T"], "negative"))
        refs = {"A": (load_ref(vae, lf, A, device), prompt(c["A"], "ref_caption")),
                "B": (load_ref(vae, lf, B, device), prompt(c["B"], "ref_caption"))}
        st = stitch(A, B)
        st.save(d / "ref_stitch.jpg", quality=92)
        cap_st = prompt(c["A"], "ref_caption")
        refs["stitch"] = (load_ref(vae, lf, st, device), cap_st)

        def cache_of(key):
            lat, cap = refs[key]
            ctx_r = anima_env.pad_contexts([enc(cap)]).to(device, torch.bfloat16)
            with torch.inference_mode():
                return asym.encode_reference(dm, lat.to(device, torch.bfloat16), ctx_r, (shape[-2], shape[-1]), mode_emb=mode_emb)
        for cond in CONDITIONS:
            if (d / f"{cond}.png").exists():
                continue
            if cond == "noref":
                cache = None
            elif cond in ("A", "B", "stitch"):
                cache = cache_of(cond)
            else:
                cache = concat_caches([cache_of("A"), cache_of("B")])
            n_ref[0] = 0 if cache is None else cache[0][0].shape[1]
            x = sample_with_cache(asym, anima_env, dm, pos, neg, shape, c["seed"], SHIFT, device, cache,
                                  bias=(cond == "multi_bias"))
            to_pil(anima_env.vae_decode(vae, lf.process_out(x.to(device)).float())[0]).save(d / f"{cond}.png")
        el = time.perf_counter() - started
        print(json.dumps({"case": ci + 1, "of": len(cases), "group": c["group"], "elapsed_s": round(el),
                          "eta_min": round(el / (ci + 1) * (len(cases) - ci - 1) / 60, 1)}), flush=True)
    uninstall()


def sample_with_cache(asym, anima_env, dm, pos_ctx, neg_ctx, shape, seed, shift, device, cache, bias: bool):
    """asym.sample with a prebuilt cache (28 steps, cfg 4.5, reference only on the positive branch)."""
    with torch.inference_mode():
        return _sample(asym, anima_env, dm, pos_ctx, neg_ctx, shape, seed, shift, device, cache, bias)


def _sample(asym, anima_env, dm, pos_ctx, neg_ctx, shape, seed, shift, device, cache, bias: bool):
    noise = torch.randn(shape, generator=torch.Generator(device="cpu").manual_seed(seed), dtype=torch.float32)
    x = noise.to(device)
    sigmas = anima_env.flow_sigmas(28, shift)
    pos = anima_env.pad_contexts([pos_ctx]).to(device, torch.bfloat16)
    neg = anima_env.pad_contexts([neg_ctx]).to(device, torch.bfloat16)
    for i in range(28):
        s0, s1 = float(sigmas[i]), float(sigmas[i + 1])
        xb, s = x.to(torch.bfloat16), torch.full((1,), s0, device=device)
        _BIAS[0] = bias
        v_pos = asym.target_forward(dm, xb, s, pos, cache).float()
        _BIAS[0] = False
        v_neg = asym.target_forward(dm, xb, s, neg, None).float()
        x = x + (s1 - s0) * (v_neg + 4.5 * (v_pos - v_neg))
    return x



LOOK_WORDS = ("hair", "eyes", "bangs", "ponytail", "twintails", "braid", "bun", "ahoge", "sidelocks", "horns", "ears",
              "tail", "wings", "halo", "hat", "ribbon", "bow", "hairband", "hairclip", "hair_ornament", "headband",
              "glasses", "earrings", "necklace", "choker", "collar", "gloves", "shirt", "skirt", "dress", "jacket",
              "coat", "uniform", "sleeves", "thighhighs", "pantyhose", "socks", "boots", "shoes", "shorts", "pants",
              "vest", "cape", "scarf", "necktie", "neckerchief", "apron", "belt", "armor", "kimono", "hoodie",
              "sweater", "leotard", "bodysuit", "serafuku", "blazer", "cardigan", "frills", "lace", "striped",
              "plaid", "checkered", "polka_dot", "print", "emblem", "badge", "mole", "freckles", "scar", "tattoo",
              "fang", "skin", "pupils", "heterochromia")
NOT_LOOK = ("looking", "open_mouth", "closed_eyes", "hand", "arm", "sitting", "standing", "holding", "smile",
            "background", "sky", "from_", "_view", "blush", "tears", "wind", "_up")


def is_look(tag: str) -> bool:
    return any(w in tag for w in LOOK_WORDS) and not any(w in tag for w in NOT_LOOK)


def score(args) -> None:
    import onnxruntime
    onnxruntime.preload_dlls()
    from imgutils.metrics import ccip_batch_extract_features, ccip_default_threshold, ccip_difference
    from imgutils.tagging import get_wd14_tags
    cases = json.loads((args.out / "cases.json").read_text(encoding="utf-8"))
    th = ccip_default_threshold()
    rows = []
    for c in cases:
        d = args.out / c["group"]
        if not all((d / f"{k}.png").exists() for k in CONDITIONS):
            continue
        ims = [Image.open(d / "truth.jpg").convert("RGB")] + [Image.open(d / f"{k}.png").convert("RGB") for k in CONDITIONS]
        f = ccip_batch_extract_features(ims)
        r = {"group": c["group"], "character": c["character"]}
        for i, k in enumerate(CONDITIONS):
            r[f"ccip_{k}"] = float(ccip_difference(f[0], f[i + 1]))
            _, general, _ = get_wd14_tags(ims[i + 1])
            r[f"multi_person_{k}"] = int(any(general.get(t, 0) > 0.35 for t in ("2girls", "2boys", "multiple_girls", "multiple_boys", "1boy_1girl")))
        # look-tag recall: appearance tags the truth shows that at least one reference also shows (what the refs could
        # carry); the share of those the output shows. Pose/scene/expression tags are excluded by is_look().
        tag = lambda im: {t for t, v in get_wd14_tags(im)[1].items() if v > 0.35 and is_look(t)}   # noqa: E731
        t_truth = tag(ims[0]); t_refs = tag(Image.open(d / "ref_A.jpg")) | tag(Image.open(d / "ref_B.jpg"))
        target = t_truth & t_refs
        r["n_look_tags"] = len(target)
        for i, k in enumerate(CONDITIONS):
            out_tags = tag(ims[i + 1])
            r[f"look_recall_{k}"] = len(target & out_tags) / len(target) if target else None
        rows.append(r)
    n = len(rows)

    def ci(vals):
        m = sum(vals) / len(vals); rng = random.Random(0); bs = []
        for _ in range(4000):
            s = [vals[rng.randrange(len(vals))] for _ in vals]; bs.append(sum(s) / len(s))
        bs.sort(); return round(m, 4), round(bs[100], 4), round(bs[3899], 4)
    summ = {"n": n, "threshold": th, "conditions": {}}
    for k in CONDITIONS:
        v = [r[f"ccip_{k}"] for r in rows]
        summ["conditions"][k] = {"ccip_mean": round(sum(v) / n, 4), "same_rate": round(sum(x < th for x in v) / n, 4),
                                 "multi_person_rate": round(sum(r[f"multi_person_{k}"] for r in rows) / n, 4)}
    rec = [r for r in rows if r["look_recall_A"] is not None]
    for k in CONDITIONS:
        summ["conditions"][k]["look_recall"] = round(sum(r[f"look_recall_{k}"] for r in rec) / len(rec), 4)
    summ["n_look"] = len(rec)
    for k in ("B", "stitch", "multi", "multi_bias"):
        summ["conditions"][k]["look_recall_diff_vs_A"] = ci([r[f"look_recall_{k}"] - r["look_recall_A"] for r in rec])
        summ["conditions"][k]["diff_vs_A"] = ci([r[f"ccip_{k}"] - r["ccip_A"] for r in rows])
        summ["conditions"][k]["diff_vs_best_single"] = ci([r[f"ccip_{k}"] - min(r["ccip_A"], r["ccip_B"]) for r in rows]) if k != "B" else None
    (args.out / "scores.json").write_text(json.dumps({"summary": summ, "rows": rows}, indent=1), encoding="utf-8")
    print(json.dumps(summ, indent=1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", type=Path)
    ap.add_argument("--index", type=Path)
    ap.add_argument("--images", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--groups", type=int, default=64)
    ap.add_argument("--comfy-root", default=str(Path.home() / "ComfyUI"))
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--self-check", action="store_true")
    args = ap.parse_args()
    if args.self_check:
        self_check()
    elif args.score:
        score(args)
    else:
        render(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
