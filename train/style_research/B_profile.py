"""Experiment B: where the time of one training microbatch goes (style microbatches, batch 2, v1a init).

    CUDA_VISIBLE_DEVICES=0 python train/style_research/B_profile.py --crops 32x1,56x1 --reps 6

Phases (cuda-synchronised wall time, s): data (latents + contexts to GPU), ref (reference stream forward), fwd (target
forward), tgt_emb (decode + metric of the true target crop, no grad), sgrad (decode + metric forward and backward to
x0_hat, B_xstyle.style_grad), bwd_fm (backward of the FM loss alone), bwd_all (backward of FM + style). Per crop config
the median over --reps microbatches; the first microbatch is warm-up and dropped.
"""
from __future__ import annotations

import argparse
import json
import statistics
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
from B_train_xstyle import LazyPrep  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", type=Path, default=Path.home() / "train" / "pilot_prep_v2")
    ap.add_argument("--config", type=Path, default=HERE.parent.parent / "configs" / "train_pilot2_ft.json")
    ap.add_argument("--init", type=Path, default=Path.home() / "adapters" / "v1a" / "snapref_v1a.safetensors")
    ap.add_argument("--crops", default="32x1,56x1")
    ap.add_argument("--reps", type=int, default=6)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--out", type=Path, default=Path.home() / "style_agent" / "B" / "profile.json")
    ap.add_argument("--comfy-root", default=str(Path.home() / "ComfyUI"))
    args = ap.parse_args()
    device = torch.device("cuda", 0)
    anima_env.setup_comfy(args.comfy_root)
    prep = LazyPrep(args.prep)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    acfg, ckpt = config["adapter"], config["train"]["checkpoint_blocks"]
    _, dm = anima_env.load_diffusion_model(config["comfy"]["unet_name"], device)
    modules = asym.inject(dm, acfg["lora"])
    mode_embed = torch.zeros(len(acfg["modes"]), dm.model_channels, device=device)
    asym.load(modules, args.init, {"mode_embed": torch.zeros(2, dm.model_channels, device=device)})
    params = asym.lora_params(modules)
    vae_model = B_xstyle.decoder_of(anima_env.load_vae(config["comfy"]["vae_name"]), device)
    embed = B_xstyle.LocalStyle(device=device)
    recs = prep.index["records"]
    sampler = ListPairSampler(prep.index, {"style": 1.0}, args.batch, 99)
    nb = args.batch
    gen = torch.Generator().manual_seed(1)
    result = {}

    def tick():
        torch.cuda.synchronize()
        return time.perf_counter()

    for spec in args.crops.split(","):
        c, n = (int(v) for v in spec.split("x"))
        rows = []
        for rep in range(args.reps + 1):
            t = {}
            t0 = tick()
            _, batch = sampler.next()
            x0 = torch.stack([prep.latent(pr[0]) for pr in batch]).to(device)
            ref = torch.stack([prep.latent(pr[1]) for pr in batch]).to(device)
            ctx = anima_env.pad_contexts([prep.context(pr[2]) for pr in batch]).to(device, torch.bfloat16)
            rctx = anima_env.pad_contexts([prep.context(recs[pr[1]]["prompts"]["ref_caption"]) for pr in batch]).to(device, torch.bfloat16)
            t1 = tick()
            t["data"] = t1 - t0
            sigma = torch.full((nb,), 0.4, device=device)
            s = sigma.view(-1, 1, 1, 1, 1)
            noise = torch.randn_like(x0, dtype=torch.float32)
            xt = ((1 - s) * x0.float() + s * noise).to(torch.bfloat16)
            for mode in ("fm", "all"):
                for p in params:
                    p.grad = None
                ta = tick()
                cache = asym.encode_reference(dm, ref, rctx, (x0.shape[-2], x0.shape[-1]), checkpoint_blocks=ckpt,
                                              mode_emb=mode_embed[0].expand(nb, -1))
                tb = tick()
                pred = asym.target_forward(dm, xt, sigma, ctx, cache, checkpoint_blocks=ckpt)
                loss = torch.nn.functional.mse_loss(pred.float(), noise - x0.float())
                tc = tick()
                if mode == "fm":
                    t["ref"], t["fwd"] = tb - ta, tc - tb
                    loss.backward()
                    t["bwd_fm"] = tick() - tc
                    continue
                box = B_xstyle.crop_boxes(x0.shape[-2], x0.shape[-1], c, n, gen)
                with torch.no_grad():
                    e_t = B_xstyle.embed_region(vae_model, lf := prep.lf, embed, x0, box, n)
                    x0_hat = xt.float() - s * pred.detach().float()
                td = tick()
                _, g = B_xstyle.style_grad(vae_model, lf, embed, x0_hat, box, e_t, n)
                te = tick()
                (loss + 0.02 * (pred.float() * (-s * g / nb)).sum()).backward()
                tf = tick()
                t["tgt_emb"], t["sgrad"], t["bwd_all"] = td - tc, te - td, tf - te
                del cache, pred, loss, g
            if rep:
                rows.append(t)
        med = {k: round(statistics.median(r[k] for r in rows), 3) for k in rows[0]}
        med["fm_microbatch"] = round(med["data"] + med["ref"] + med["fwd"] + med["bwd_fm"], 3)
        med["style_extra"] = round(med["tgt_emb"] + med["sgrad"] + med["bwd_all"] - med["bwd_fm"], 3)
        med["peak_gb"] = round(torch.cuda.max_memory_allocated(device) / 2**30, 1)
        result[spec] = med
        print(spec, json.dumps(med), flush=True)
        torch.cuda.reset_peak_memory_stats(device)
    args.out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print("PROFILE_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
