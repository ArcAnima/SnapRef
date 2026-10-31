"""Experiment A self-check before training (v1a init, style pairs with a teacher context, batch 4).

    CUDA_VISIBLE_DEVICES=0 python train/style_research/A_selfcheck.py --prep ~/train/pilot_prep_v2 \
        --teacher ~/style_agent/A/teacher_ctx --config CONFIG.json --out ~/style_agent/A/selfcheck.json [--batches 6]

Checks and measures, per microbatch (A_train_distill.py's sampler, prompts, sigma draw and reference encoding):
  * teacher_terms with the LoRA off equals the plain ComfyUI forward (the teacher is frozen Anima);
  * rms of d_art per sample and sigma (D1, diag_loss.py: 0.012-0.040 at sigma 0.3-0.9);
  * FM loss, L_dist, projection and cosine of d_s on d_art (v1a: 0.05-0.30 / 0.1-0.2 in D1);
  * LoRA gradient norm of FM, of L_dist and of the normalized L_dist separately (for the weight), peak memory, seconds.
Data are read by slice from the prep files (the baseline training holds its own copy in RAM).
"""
from __future__ import annotations

import argparse
import json
import random
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
from A_train_distill import direction_stats, distill_loss, teacher_terms  # noqa: E402
from train_asym import ListPairSampler  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", type=Path, required=True)
    ap.add_argument("--teacher", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--init", type=Path, default=Path.home() / "adapters" / "v1a" / "snapref_v1a.safetensors")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--batches", type=int, default=6)
    ap.add_argument("--comfy-root", default=str(Path.home() / "ComfyUI"))
    args = ap.parse_args()
    dev = torch.device("cuda", 0)
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    tcfg, acfg = cfg["train"], cfg["adapter"]
    index = json.loads((args.prep / "index.json").read_text(encoding="utf-8"))
    hc = safe_open(str(args.prep / "contexts.safetensors"), "pt")
    data, offs = hc.get_slice("data"), hc.get_tensor("offsets").tolist()
    context = lambda i: data[offs[i]:offs[i + 1]]  # noqa: E731
    lat_of = {}
    for name in index["files"]["latents"]:
        h = safe_open(str((args.prep / name).resolve()), "pt")
        for k in h.keys():
            lat_of[k] = h
    anima_env.setup_comfy(args.comfy_root)
    lf = anima_env.latent_format()
    latent = lambda k: lf.process_in(lat_of[k].get_tensor(k)[None].float())[0].to(torch.bfloat16)  # noqa: E731
    tc = load_file(str(args.teacher / "contexts.safetensors"))
    t_off = tc["offsets"].tolist()
    t_map = {int(k): v for k, v in json.loads((args.teacher / "map.json").read_text(encoding="utf-8"))["map"].items()}
    teacher_context = lambda pi: tc["data"][t_off[t_map[pi]]:t_off[t_map[pi] + 1]]  # noqa: E731

    _, dm = anima_env.load_diffusion_model(cfg["comfy"]["unet_name"], dev)
    modules = asym.inject(dm, acfg["lora"])
    names = ["style", "char", "cs"]
    memb_all = torch.zeros(3, dm.model_channels, device=dev)
    old = torch.zeros(2, dm.model_channels, device=dev)
    asym.load(modules, args.init, {"mode_embed": old})
    memb_all[1:] = old[[1, 0]]   # v1a modes [cs, char] -> [style, char, cs]; style starts at zero (train_asym --init)
    params = asym.lora_params(modules)
    sampler = ListPairSampler(index, {"style": 1.0}, tcfg["batch_size"], tcfg["seed"])
    rng = random.Random(tcfg["seed"] + 7)
    torch.manual_seed(tcfg["seed"])
    shift, ckpt = tcfg["shift"], tcfg["checkpoint_blocks"]
    rows = []

    # (1) the teacher is the frozen model: teacher_terms' base branch vs ComfyUI's forward
    x = torch.randn(1, 16, 1, 32, 32, device=dev).to(torch.bfloat16)
    c = torch.randn(1, 512, 1024, device=dev).to(torch.bfloat16) * 0.1
    s = torch.tensor([0.6], device=dev)
    with torch.no_grad():
        _, d0 = teacher_terms(dm, modules, x, s, c, c)
        asym.set_enabled(modules, False)
        same = (asym.target_forward(dm, x, s, c, None).float() - dm(x, s, c).float()).abs().max().item()
        asym.set_enabled(modules, True)
    check = {"d_art_same_context_maxabs": d0.abs().max().item(), "lora_off_vs_comfy_maxabs": same}
    print(json.dumps(check), flush=True)

    while len(rows) < args.batches:
        _, batch = sampler.next()
        batch = [pr for pr in batch if pr[2] in t_map]
        if not batch:
            continue
        x0 = torch.stack([latent(pr[0]) for pr in batch]).to(dev)
        ctx = anima_env.pad_contexts([context(pr[2]) for pr in batch]).to(dev, torch.bfloat16)
        ca = anima_env.pad_contexts([teacher_context(pr[2]) for pr in batch]).to(dev, torch.bfloat16)
        uu = torch.rand(len(batch), device=dev)
        sigma = (shift * uu / (1 + (shift - 1) * uu)).clamp(1e-3, 1.0)
        noise = torch.randn_like(x0, dtype=torch.float32)
        s5 = sigma.view(-1, 1, 1, 1, 1)
        xt = ((1 - s5) * x0.float() + s5 * noise).to(torch.bfloat16)
        target_v = noise - x0.float()
        ref = torch.stack([latent(pr[1]) for pr in batch]).to(dev)
        rctx = anima_env.pad_contexts([context(index["records"][pr[1]]["prompts"]["ref_caption"]) if rng.random() < 0.7
                                       else context(0) for pr in batch]).to(dev, torch.bfloat16)
        torch.cuda.reset_peak_memory_stats(dev)
        t0 = time.perf_counter()
        v_noref, d_art = teacher_terms(dm, modules, xt, sigma, ctx, ca)
        t_teacher = time.perf_counter() - t0
        grads = {}
        for which in ("fm", "dist", "dist_norm"):
            for p in params:
                p.grad = None
            cache = asym.encode_reference(dm, ref, rctx, (x0.shape[-2], x0.shape[-1]), checkpoint_blocks=ckpt,
                                          mode_emb=memb_all[0].expand(len(batch), -1))
            pred = asym.target_forward(dm, xt, sigma, ctx, cache, checkpoint_blocks=ckpt).float()
            fm = torch.nn.functional.mse_loss(pred, target_v)
            dist = distill_loss(pred, v_noref, d_art, which == "dist_norm")
            (fm if which == "fm" else dist).backward()
            if which != "fm":
                grads[which + "_value"] = dist.item()
            grads[which] = float(torch.sqrt(sum((p.grad.float() ** 2).sum() for p in params if p.grad is not None)))
        proj, cos = direction_stats(pred.detach() - v_noref, d_art)
        rms = d_art.flatten(1).pow(2).mean(1).sqrt()
        row = {"sigma": [round(v, 3) for v in sigma.tolist()], "d_art_rms": [round(v, 4) for v in rms.tolist()],
               "proj": [round(v, 3) for v in proj.tolist()], "cos": [round(v, 3) for v in cos.tolist()],
               "fm": round(fm.item(), 5), "dist": round(grads["dist_value"], 6), "dist_norm": round(grads["dist_norm_value"], 4),
               "g_fm": round(grads["fm"], 5), "g_dist": round(grads["dist"], 5), "g_dist_norm": round(grads["dist_norm"], 5),
               "g_ratio": round(grads["dist"] / max(grads["fm"], 1e-12), 3),
               "g_ratio_norm": round(grads["dist_norm"] / max(grads["fm"], 1e-12), 4),
               "teacher_s": round(t_teacher, 2), "peak_gb": round(torch.cuda.max_memory_allocated(dev) / 2 ** 30, 1),
               "n": len(batch)}
        rows.append(row)
        print(json.dumps(row), flush=True)
    for p in params:
        p.grad = None
    out = {"check": check, "rows": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("SELFCHECK_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
