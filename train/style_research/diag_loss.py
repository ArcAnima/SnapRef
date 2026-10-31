"""Mechanism diagnosis: does the reference lower the target loss, where (sigma, frequency band), and does it move the
prediction the way Anima's own artist tag does?

    CUDA_VISIBLE_DEVICES=0 python train/style_research/diag_loss.py --pairs ~/style_agent/diag/pairs.json \
        --out ~/style_agent/diag/loss.jsonl [--stats 40]

Pairs (build_diag_pairs.py, v2-pool artists the pilot never trained on): target T (artist a, character X), right
reference R (a, other character), wrong reference W (other artist, other character), cs reference C (a, X) when one
exists, and the target itself (oracle). Fixed noise per pair; sigmas SIGMAS in one batch. Conditions:
  base / base_art        frozen Anima, prompt without / with "@artist" (the text route to the same style)
  v1a_{noref,R,W,T,C}    v1a, mode cs (v1a has no style mode)
  p2s_{noref,R,W,T}      pilot2_ft step 2000, mode style
  p2c_{R,C}              pilot2_ft step 2000, mode cs
Per condition and sigma: velocity MSE in total and in three bands of the latent spectrum (layout f < 1/16 cycles per
latent pixel, i.e. > 128 image px; mid 1/16..1/4; fine >= 1/4: detail inside one 2x2 patch token, < 32 image px).
Direction: d_art = v_base_art - v_base; for each adapter, d_X = v_X - v_noref; reported cos(d_X, d_art) and the
projection <d_X, d_art> / |d_art|^2 (share of the artist direction the reference reproduces), total and fine band.
--stats N: for the first N pairs, per block attention mass on the reference keys, its entropy and the share of it
within 3 patches of the query's aligned position (asym_style), for v1a_R, p2s_R, p2c_C, at sigmas 0.9 / 0.5 / 0.15.
Resumable: pairs already in --out are skipped.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import anima_env  # noqa: E402
import asym  # noqa: E402
import asym_style  # noqa: E402

SIGMAS = (0.05, 0.15, 0.3, 0.5, 0.7, 0.9)
STAT_SIGMAS = (0.9, 0.5, 0.15)


def band_masks(h: int, w: int, device) -> list[torch.Tensor]:
    fy = torch.fft.fftfreq(h, device=device).abs()[:, None]
    fx = torch.fft.fftfreq(w, device=device).abs()[None, :]
    r = torch.maximum(fy, fx)   # Chebyshev radius: the patch grid is square
    return [r < 1 / 16, (r >= 1 / 16) & (r < 1 / 4), r >= 1 / 4]


def bands(err: torch.Tensor, masks) -> list[list[float]]:
    """err [S, 16, 1, H, W] -> per sigma [low, mid, fine] mean squared error (sums to the total MSE)."""
    e = err[:, :, 0].float()
    p = torch.fft.fft2(e).abs().pow(2) / (e.shape[-1] * e.shape[-2])     # Parseval: mean over pixels
    n = e.shape[1] * e.shape[-1] * e.shape[-2]
    return [[float(p[s][:, m].sum() / n) for m in masks] for s in range(e.shape[0])]


def fine(x: torch.Tensor, masks) -> torch.Tensor:
    f = torch.fft.fft2(x[:, :, 0].float())
    return torch.fft.ifft2(f * masks[2]).real


def cos_proj(d: torch.Tensor, ref: torch.Tensor) -> tuple[list[float], list[float]]:
    d, ref = d.flatten(1).float(), ref.flatten(1).float()
    dot = (d * ref).sum(1)
    cos = dot / (d.norm(dim=1) * ref.norm(dim=1)).clamp_min(1e-12)
    proj = dot / ref.pow(2).sum(1).clamp_min(1e-12)
    return [round(float(x), 4) for x in cos], [round(float(x), 4) for x in proj]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--v1a", type=Path, default=Path.home() / "adapters" / "v1a" / "snapref_v1a.safetensors")
    ap.add_argument("--p2", type=Path, default=Path.home() / "train" / "runs" / "pilot2_ft" / "adapter_002000.safetensors")
    ap.add_argument("--config", type=Path, default=Path.home() / "train" / "runs" / "pilot2_ft" / "config.json")
    ap.add_argument("--comfy-root", default=str(Path.home() / "ComfyUI"))
    ap.add_argument("--stats", type=int, default=0)
    ap.add_argument("--limit", type=int, default=10 ** 9)
    args = ap.parse_args()
    dev = torch.device("cuda", 0)
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    pairs = json.loads(args.pairs.read_text(encoding="utf-8"))[: args.limit]
    done = set()
    if args.out.exists():
        done = {json.loads(l)["pair"] for l in args.out.read_text(encoding="utf-8").splitlines() if l.strip()}
    anima_env.setup_comfy(args.comfy_root)
    _, dm = anima_env.load_diffusion_model(cfg["comfy"]["unet_name"], dev)
    asym_style.install(dm)
    modules = asym.inject(dm, cfg["adapter"]["lora"])
    params = asym.lora_params(modules)
    weights, modes = {}, {}
    for name, path in (("v1a", args.v1a), ("p2", args.p2)):
        with safe_open(str(path), "pt") as h:
            modes[name] = json.loads((h.metadata() or {}).get("anima_ref_kv", "{}")).get("modes", ["cs", "char"])
        E = torch.zeros(len(modes[name]), dm.model_channels, device=dev)
        asym.load(modules, path, {"mode_embed": E})
        weights[name] = ([p.detach().clone() for p in params], E)

    def use(name):
        for p, w in zip(params, weights[name][0]):
            p.data = w
        return weights[name][1]

    def memb(name, mode):
        return weights[name][1][modes[name].index(mode)][None]

    clip = anima_env.load_text_encoder(cfg["comfy"]["clip_name"])
    enc_cache = {}

    def enc(t):
        if t not in enc_cache:
            enc_cache[t] = anima_env.encode_prompt(clip, dm, t, dev)
        return anima_env.pad_contexts([enc_cache[t]]).to(dev, torch.bfloat16)
    lf = anima_env.latent_format()
    shard_cache = {}

    def latent(pid, file):
        if file not in shard_cache:
            shard_cache.clear()   # pairs are spread over shards; keep one open at a time
            shard_cache[file] = safe_open(file, "pt")
        x = shard_cache[file].get_tensor(str(pid))
        return lf.process_in(x[None].float().to(dev)).to(torch.bfloat16)

    empty = enc("")
    S = len(SIGMAS)
    sig = torch.tensor(SIGMAS, device=dev)
    started = time.perf_counter()
    with args.out.open("a", encoding="utf-8") as fout, torch.inference_mode():
        for k, p in enumerate(pairs):
            if k in done:
                continue
            x0 = latent(p["t"], p["t_file"])
            refs = {"R": latent(p["r"], p["r_file"]), "W": latent(p["w"], p["w_file"]), "T": x0}
            if p.get("c"):
                refs["C"] = latent(p["c"], p["c_file"])
            H, W = x0.shape[-2], x0.shape[-1]
            gen = torch.Generator(device="cpu").manual_seed(1000 + k)
            noise = torch.randn(x0.shape, generator=gen).to(dev)
            s5 = sig.view(-1, 1, 1, 1, 1)
            xt = ((1 - s5) * x0.float() + s5 * noise).to(torch.bfloat16)          # [S, 16, 1, H, W]
            v_true = (noise - x0.float()).expand(S, -1, -1, -1, -1)
            ctx = enc(p["prompt"]).expand(S, -1, -1)
            ctx_art = enc(p["prompt_art"]).expand(S, -1, -1)
            masks = band_masks(H, W, dev)
            rec = {"pair": k, "artist": p["artist"], "sigmas": SIGMAS, "loss": {}, "band": {}, "cos": {}, "proj": {},
                   "cos_fine": {}, "proj_fine": {}}

            def score(name, v):
                err = v.float() - v_true
                rec["loss"][name] = [round(float(e.pow(2).mean()), 6) for e in err]
                rec["band"][name] = [[round(b, 6) for b in bb] for bb in bands(err, masks)]

            asym.set_enabled(modules, False)
            v_base = asym_style.target_forward(dm, xt, sig, ctx, None)
            v_art = asym_style.target_forward(dm, xt, sig, ctx_art, None)
            asym.set_enabled(modules, True)
            score("base", v_base)
            score("base_art", v_art)
            d_art = v_art.float() - v_base.float()
            d_art_f = fine(d_art, masks)
            plan = {"v1a": [("cs", "v1a", ["R", "W", "T", "C"])],
                    "p2": [("style", "p2s", ["R", "W", "T"]), ("cs", "p2c", ["R", "C"])]}
            for name, items in plan.items():
                use(name)
                for mode, tag, keys in items:
                    if tag != "p2c":
                        v_no = asym_style.target_forward(dm, xt, sig, ctx, None)
                        score(f"{tag}_noref", v_no)
                        no_v = v_no.float()
                    for key in keys:
                        if key not in refs:
                            continue
                        cache = asym.encode_reference(dm, refs[key], empty, (H, W), mode_emb=memb(name, mode))
                        v = asym_style.target_forward(dm, xt, sig, ctx, cache)
                        score(f"{tag}_{key}", v)
                        d = v.float() - no_v
                        rec["cos"][f"{tag}_{key}"], rec["proj"][f"{tag}_{key}"] = cos_proj(d, d_art)
                        rec["cos_fine"][f"{tag}_{key}"], rec["proj_fine"][f"{tag}_{key}"] = cos_proj(fine(d, masks), d_art_f)
            rec["d_art_norm"] = [round(float(x), 4) for x in d_art.flatten(1).norm(dim=1) / d_art[0].numel() ** 0.5]
            if k < args.stats:
                rec["attn"] = {}
                tg, rg = asym_style.grid_positions((H, W), (refs["R"].shape[-2], refs["R"].shape[-1]))
                qidx = torch.randperm(tg.shape[0], generator=torch.Generator().manual_seed(k))[:256]
                for name, mode, key, tag in (("v1a", "cs", "R", "v1a_R"), ("p2", "style", "R", "p2s_R"),
                                             ("p2", "cs", "C" if "C" in refs else "R", "p2c_C")):
                    use(name)
                    rk = refs[key]
                    tg, rg = asym_style.grid_positions((H, W), (rk.shape[-2], rk.shape[-1]))
                    cache = asym.encode_reference(dm, rk, empty, (H, W), mode_emb=memb(name, mode))
                    out = {}
                    for s in STAT_SIGMAS:
                        st = {"qidx": qidx, "ref_pos": rg, "tgt_pos": tg, "radius": 3, "rows": []}
                        j = SIGMAS.index(s) if s in SIGMAS else None
                        xs = xt[j:j + 1] if j is not None else ((1 - s) * x0.float() + s * noise).to(torch.bfloat16)
                        asym_style.target_forward(dm, xs, torch.full((1,), s, device=dev), ctx[:1], cache, stats=st)
                        out[str(s)] = [[b, m.tolist(), e.tolist(), l.tolist()] for b, m, e, l in st["rows"]]
                    rec["attn"][tag] = out
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            if k % 10 == 0:
                print(f"pair {k}/{len(pairs)} {time.perf_counter() - started:.0f}s", flush=True)
    print("DIAG_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
