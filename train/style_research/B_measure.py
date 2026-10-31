"""Experiment B, step 1: measure the x0 style loss before training (v1a init, style pairs of pilot_prep_v2).

    CUDA_VISIBLE_DEVICES=0 python train/style_research/B_measure.py --out ~/style_agent/B/measure.jsonl --batches 8

For each style microbatch (batch 2, same pairs and prompts as training) and each sigma in --sigmas, one forward with the
reference; then per crop config (latent side c, n_crops):
  g_fm, g_st     LoRA gradient norm of the FM loss and of the style loss (lam = 1), same microbatch;
  gx_ratio       |sigma g_x0| / |grad_v FM| in latent space;
  loss           1 - cos(E(x0_hat crop), E(target crop)) (the training loss);
  sep_hat        cos(E(x0_hat), E(same-artist other post)) - cos(E(x0_hat), E(other-artist post)): does the prediction
                 already carry the artist at this sigma (random crops of the same side);
  sep_true       the same for the true target crop (how much artist signal one crop of this size has);
  mem_gb, sec    peak memory and time of the style branch.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import anima_env  # noqa: E402
import asym  # noqa: E402
from train_asym import ListPairSampler  # noqa: E402
import B_xstyle  # noqa: E402
from B_train_xstyle import LazyPrep, lora_grad_norm  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", type=Path, default=Path.home() / "train" / "pilot_prep_v2")
    ap.add_argument("--config", type=Path, default=HERE.parent.parent / "configs" / "train_pilot2_ft.json")
    ap.add_argument("--init", type=Path, default=Path.home() / "adapters" / "v1a" / "snapref_v1a.safetensors")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--batches", type=int, default=8)
    ap.add_argument("--sigmas", default="0.1,0.2,0.3,0.45,0.6,0.8")
    ap.add_argument("--crops", default="32x1,52x1,64x1,26x4")
    ap.add_argument("--comfy-root", default=str(Path.home() / "ComfyUI"))
    args = ap.parse_args()
    device = torch.device("cuda", 0)
    anima_env.setup_comfy(args.comfy_root)
    prep = LazyPrep(args.prep)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    acfg = config["adapter"]
    _, dm = anima_env.load_diffusion_model(config["comfy"]["unet_name"], device)
    modules = asym.inject(dm, acfg["lora"])
    from safetensors import safe_open
    with safe_open(str(args.init), "pt") as h:
        init_modes = json.loads((h.metadata() or {}).get("anima_ref_kv", "{}"))["modes"]
    old = torch.zeros(len(init_modes), dm.model_channels, device=device)
    asym.load(modules, args.init, {"mode_embed": old})
    mode_names = list(acfg["modes"])
    mode_embed = torch.nn.Parameter(torch.zeros(len(mode_names), dm.model_channels, device=device))
    with torch.no_grad():
        for i, m in enumerate(mode_names):
            if m in init_modes:
                mode_embed[i].copy_(old[init_modes.index(m)])
    params = asym.lora_params(modules) + [mode_embed]
    lf = prep.lf
    vae_model = B_xstyle.decoder_of(anima_env.load_vae(config["comfy"]["vae_name"]), device)
    embed = B_xstyle.LocalStyle(device=device)
    recs = prep.index["records"]
    sampler = ListPairSampler(prep.index, {"style": 1.0}, 2, 777)
    by_artist = collections.defaultdict(list)
    for rid, r in recs.items():
        if prep.has(rid):
            by_artist[r["artist"]].append(rid)
    artists = sorted(by_artist)
    rng = random.Random(5)
    gen = torch.Generator().manual_seed(5)
    sigmas = [float(s) for s in args.sigmas.split(",")]
    crops = [tuple(int(v) for v in c.split("x")) for c in args.crops.split(",")]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fh = args.out.open("a", encoding="utf-8")

    def emb_of(lat, side, n):
        box = B_xstyle.crop_boxes(lat.shape[-2], lat.shape[-1], side, n, gen)
        return B_xstyle.embed_region(vae_model, lf, embed, lat, box, n)

    for bi in range(args.batches):
        _, batch = sampler.next()
        x0 = torch.stack([prep.latent(pr[0]) for pr in batch]).to(device)
        ref = torch.stack([prep.latent(pr[1]) for pr in batch]).to(device)
        ctx = anima_env.pad_contexts([prep.context(pr[2]) if len(pr) > 2 else prep.context(0) for pr in batch]).to(device, torch.bfloat16)
        rctx = anima_env.pad_contexts([prep.context(recs[pr[1]]["prompts"]["ref_caption"]) for pr in batch]).to(device, torch.bfloat16)
        same = [prep.latent(rng.choice([i for i in by_artist[recs[pr[0]]["artist"]] if i not in pr[:2]] or [pr[1]])).to(device)[None] for pr in batch]
        other = []
        for pr in batch:
            a = rng.choice(artists)
            while a == recs[pr[0]]["artist"]:
                a = rng.choice(artists)
            other.append(prep.latent(rng.choice(by_artist[a])).to(device)[None])
        noise = torch.randn(x0.shape, generator=torch.Generator().manual_seed(100 + bi)).to(device)
        memb = mode_embed[mode_names.index("style")].expand(2, -1)
        for sg in sigmas:
            sigma = torch.full((2,), sg, device=device)
            s = sigma.view(-1, 1, 1, 1, 1)
            xt = ((1 - s) * x0.float() + s * noise).to(torch.bfloat16)
            cache = asym.encode_reference(dm, ref, rctx, (x0.shape[-2], x0.shape[-1]), checkpoint_blocks=16, mode_emb=memb)
            pred = asym.target_forward(dm, xt, sigma, ctx, cache, checkpoint_blocks=16)
            target_v = noise - x0.float()
            fm = torch.nn.functional.mse_loss(pred.float(), target_v)
            gf = torch.autograd.grad(fm, params, retain_graph=True, allow_unused=True)
            gv_fm = 2 * (pred.detach().float() - target_v) / target_v.numel()
            x0_hat = xt.float() - s * pred.detach().float()
            for (c, n) in crops:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats(device)
                t0 = time.perf_counter()
                box = B_xstyle.crop_boxes(x0.shape[-2], x0.shape[-1], c, n, gen)
                e_t = B_xstyle.embed_region(vae_model, lf, embed, x0, box, n)
                l_st, g_x0 = B_xstyle.style_grad(vae_model, lf, embed, x0_hat, box, e_t, n)
                g_v = -s * g_x0 / 2
                gs = torch.autograd.grad((pred.float() * g_v).sum(), params, retain_graph=True, allow_unused=True)
                torch.cuda.synchronize()
                sec = time.perf_counter() - t0
                mem = torch.cuda.max_memory_allocated(device) / 2**30
                with torch.no_grad():
                    e_hat = B_xstyle.embed_region(vae_model, lf, embed, x0_hat, box, n)
                    side = box[2] // (2 if n == 4 else 1)
                    e_same = torch.cat([emb_of(g, side, n) for g in same])
                    e_oth = torch.cat([emb_of(g, side, n) for g in other])
                row = {"batch": bi, "sigma": sg, "crop": c, "n_crops": n, "side": box[2], "fm": float(fm),
                       "g_fm": lora_grad_norm(gf), "g_st": lora_grad_norm(gs),
                       "gx_ratio": float(g_v.norm() / gv_fm.norm()),
                       "loss": [round(float(v), 4) for v in l_st],
                       "sep_hat": [round(float(v), 4) for v in ((e_hat * e_same).sum(-1) - (e_hat * e_oth).sum(-1))],
                       "sep_true": [round(float(v), 4) for v in ((e_t * e_same).sum(-1) - (e_t * e_oth).sum(-1))],
                       "mem_gb": round(mem, 1), "sec": round(sec, 2)}
                fh.write(json.dumps(row) + "\n")
                fh.flush()
            del cache, pred, gf, fm, gs, g_x0, g_v
            torch.cuda.empty_cache()
        print(f"batch {bi} done", flush=True)
    print("MEASURE_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
