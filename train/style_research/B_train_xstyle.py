"""Experiment B trainer: train_asym.py's loop (single GPU, ListPairSampler preps) plus a style loss on the x0 prediction.

    CUDA_VISIBLE_DEVICES=0 python train/style_research/B_train_xstyle.py --prep ~/train/pilot_prep_v2 \
        --config configs/train_pilot2_ft.json --init ~/adapters/v1a/snapref_v1a.safetensors --steps 1000 \
        --out ~/style_agent/B/run --lam 1.0 --sigma-lo 0.15 --sigma-hi 0.6 --crop 48 --n-crops 1

Unchanged from train_asym.py: sampler, prompts, dropout, preservation loss, LR schedule, seeds; the FM loss is the same.
Added on style-mode microbatches with a reference: for samples with sigma in [sigma_lo, sigma_hi],
    L_style = 1 - cos(E(D(x0_hat)[box]), E(D(x_target)[box']))     x0_hat = x_sigma - sigma * v_hat
E = local_linear probe (B_xstyle.LocalStyle), D = VAE decoder with gradient; box = random latent crop, box' the same box
(--style-target target) or a random crop of another post by the same artist (--style-target gallery).
total = L_fm + lam * mean_b L_style; L_style's gradient reaches v_hat through a two-stage product (B_xstyle.style_grad).
Data are read lazily (contexts and latents by slice from the safetensors files), so RAM stays small for two jobs per box.
Every log step one style microbatch also measures |grad L_fm| and |lam grad L_style| on the LoRA weights separately.
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
from safetensors import safe_open

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import anima_env  # noqa: E402
import asym  # noqa: E402
from train_asym import ListPairSampler, lr_at  # noqa: E402
import B_xstyle  # noqa: E402


class LazyPrep:
    """train_asym.Prep without loading everything: context(i) and latent(id) read slices from the files."""

    def __init__(self, prep_dir: Path):
        self.dir = Path(prep_dir)
        self.index = json.loads((self.dir / "index.json").read_text(encoding="utf-8"))
        h = safe_open(str(self.dir / "contexts.safetensors"), "pt")
        self._data, self._offsets = h.get_slice("data"), h.get_tensor("offsets").tolist()
        self._ctx_handle = h
        self._lat = {}
        self._handles = []
        for name in self.index["files"]["latents"]:
            hh = safe_open(str((self.dir / name).resolve()), "pt")
            self._handles.append(hh)
            for k in hh.keys():
                self._lat[k] = hh
        self.lf = anima_env.latent_format()

    def context(self, i: int) -> torch.Tensor:
        return self._data[self._offsets[i]:self._offsets[i + 1]]

    def has(self, k: str) -> bool:
        return k in self._lat

    def latent(self, k: str) -> torch.Tensor:
        return self.lf.process_in(self._lat[k].get_tensor(k)[None].float())[0].to(torch.bfloat16)


def lora_grad_norm(grads) -> float:
    return float(torch.sqrt(sum((g.float() ** 2).sum() for g in grads if g is not None)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--comfy-root", default=str(Path.home() / "ComfyUI"))
    ap.add_argument("--steps", type=int)
    ap.add_argument("--resume", type=Path)
    ap.add_argument("--init", type=Path)
    ap.add_argument("--stop-at", type=int)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--save-every", type=int, help="override train.save_every (spot machines: 100)")
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--sigma-lo", type=float, default=0.15)
    ap.add_argument("--sigma-hi", type=float, default=0.6)
    ap.add_argument("--crop", type=int, default=48, help="latent crop side (n_crops 4: region of 2 x crop)")
    ap.add_argument("--n-crops", type=int, default=1, choices=(1, 4))
    ap.add_argument("--style-target", default="target", choices=("target", "gallery"))
    ap.add_argument("--style-p", type=float, default=1.0, help="share of eligible microbatches that run the style branch; "
                    "its gradient is scaled by 1/p so the expected update is unchanged (speed knob)")
    args = ap.parse_args()

    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    log = lambda *a: print(*a, flush=True)  # noqa: E731
    anima_env.setup_comfy(args.comfy_root)
    prep = LazyPrep(args.prep)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    tcfg, acfg = config["train"], config["adapter"]
    tcfg["batch_size"], tcfg["grad_accum"] = args.batch, args.accum
    if args.steps:
        tcfg["steps"] = args.steps
    if args.save_every:
        tcfg["save_every"] = args.save_every
    if args.smoke:
        tcfg["steps"], tcfg["log_every"] = 6, 2
    config["B"] = {k: v for k, v in vars(args).items() if k in ("lam", "sigma_lo", "sigma_hi", "crop", "n_crops", "style_target", "style_p", "batch", "accum")}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=1), encoding="utf-8")

    seed = tcfg["seed"]
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    _, dm = anima_env.load_diffusion_model(config["comfy"]["unet_name"], device)
    modules = asym.inject(dm, acfg["lora"])
    mode_names = list(acfg["modes"])
    mode_embed = torch.nn.Parameter(torch.zeros(len(mode_names), dm.model_channels, device=device))
    params = asym.lora_params(modules) + [mode_embed]
    optimizer = torch.optim.AdamW(params, lr=tcfg["lr"], betas=(0.9, 0.999), weight_decay=tcfg["weight_decay"])

    lf = prep.lf
    vae_model = B_xstyle.decoder_of(anima_env.load_vae(config["comfy"]["vae_name"]), device)
    embed = B_xstyle.LocalStyle(device=device)
    recs = prep.index["records"]
    sampler = ListPairSampler(prep.index, acfg["modes"], tcfg["batch_size"], seed)
    by_artist = collections.defaultdict(list)
    for rid, r in recs.items():
        if prep.has(rid):
            by_artist[r["artist"]].append(rid)
    log(f"pairs per mode: {sampler.counts}; artists {len(by_artist)}")
    drop = config["data"]["dropout"]
    empty_ctx = 0
    rng = random.Random(seed + 7)
    crop_gen = torch.Generator().manual_seed(seed + 11)   # own generator: data and noise stay those of train_asym.py
    crop_rng = random.Random(seed + 13)

    start_step = 0
    if args.init and not args.resume:
        with safe_open(str(args.init), "pt") as handle:
            init_modes = json.loads((handle.metadata() or {}).get("anima_ref_kv", "{}")).get("modes", mode_names)
        old = torch.zeros(len(init_modes), dm.model_channels, device=device)
        asym.load(modules, args.init, {"mode_embed": old})
        with torch.no_grad():
            for i, m in enumerate(mode_names):
                if m in init_modes:
                    mode_embed[i].copy_(old[init_modes.index(m)])
        log(f"initialised from {args.init} (modes {init_modes} -> {mode_names})")
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=True)
        asym.load(modules, Path(str(args.resume).replace("state_", "adapter_").replace(".pt", ".safetensors")),
                  {"mode_embed": mode_embed})
        optimizer.load_state_dict(state["optimizer"])
        start_step = state["step"]
        rng.setstate(state["rng"][0])
        sampler.rng.setstate(state["sampler_rng"][0])
        torch.set_rng_state(state["torch_rng"][0].cpu())
        crop_gen.set_state(state["crop_gen"])
        crop_rng.setstate(state["crop_rng"])
        log(f"resumed at step {start_step}")

    metadata = {"format": "anima_ref_kv/1", "base": config["comfy"]["unet_name"], "ref_t": asym.REF_T,
                "modes": mode_names, "lora": acfg["lora"], "ref_text": "identity caption or empty",
                "ref_offset_max": acfg["ref_offset_max"], "B": config["B"]}
    shift, ckpt_blocks, accum = tcfg["shift"], tcfg["checkpoint_blocks"], tcfg["grad_accum"]
    log_path = args.out / "train_log.jsonl"
    running = collections.defaultdict(list)
    started = window = time.perf_counter()
    end_step = min(tcfg["steps"], args.stop_at or tcfg["steps"])
    for step in range(start_step, end_step):
        for group in optimizer.param_groups:
            group["lr"] = lr_at(step, tcfg)
        optimizer.zero_grad(set_to_none=True)
        measure = (step + 1) % tcfg["log_every"] == 0
        for _ in range(accum):
            mode, batch = sampler.next()
            u = rng.random()
            no_ref = u < drop["ref"] + drop["both"]
            no_text = drop["ref"] <= u < drop["ref"] + drop["both"] + drop["text"]
            x0 = torch.stack([prep.latent(pr[0]) for pr in batch]).to(device)
            ctxs = []
            for pr in batch:
                if no_text:
                    ctxs.append(prep.context(empty_ctx))
                elif len(pr) > 2:
                    ctxs.append(prep.context(pr[2]))
                else:
                    ctxs.append(prep.context(recs[pr[0]]["prompts"][prep.index["prompt_of_mode"][mode]]))
            context = anima_env.pad_contexts(ctxs).to(device, torch.bfloat16)
            uu = torch.rand(len(batch), device=device)
            sigma = (shift * uu / (1 + (shift - 1) * uu)).clamp(1e-3, 1.0)
            noise = torch.randn_like(x0, dtype=torch.float32)
            s = sigma.view(-1, 1, 1, 1, 1)
            xt = ((1 - s) * x0.float() + s * noise).to(torch.bfloat16)
            target_v = noise - x0.float()
            cache = None
            if not no_ref:
                ref = torch.stack([prep.latent(pr[1]) for pr in batch]).to(device)
                rctx = [prep.context(recs[pr[1]]["prompts"]["ref_caption"]) if rng.random() < config["data"]["ref_caption_p"]
                        else prep.context(empty_ctx) for pr in batch]
                rcontext = anima_env.pad_contexts(rctx).to(device, torch.bfloat16)
                m = acfg["ref_offset_max"]
                offset = (rng.randint(-m, m), rng.randint(-m, m))
                memb = mode_embed[mode_names.index(mode)].expand(len(batch), -1)
                cache = asym.encode_reference(dm, ref, rcontext, (x0.shape[-2], x0.shape[-1]), offset=offset,
                                              checkpoint_blocks=ckpt_blocks, mode_emb=memb)
            pred = asym.target_forward(dm, xt, sigma, context, cache, checkpoint_blocks=ckpt_blocks)
            loss = torch.nn.functional.mse_loss(pred.float(), target_v)
            total = loss
            if no_ref and config["data"]["preserve_weight"] > 0:
                with torch.no_grad():
                    asym.set_enabled(modules, False)
                    v_base = asym.target_forward(dm, xt, sigma, context, None).float()
                    asym.set_enabled(modules, True)
                total = total + config["data"]["preserve_weight"] * torch.nn.functional.mse_loss(pred.float(), v_base)
            aux = None
            gate = ((sigma >= args.sigma_lo) & (sigma <= args.sigma_hi)).float()
            if (mode == "style" and cache is not None and gate.any() and args.lam > 0
                    and (args.style_p >= 1 or crop_rng.random() < args.style_p)):
                h, w = x0.shape[-2:]
                box = B_xstyle.crop_boxes(h, w, args.crop, args.n_crops, crop_gen)
                with torch.no_grad():
                    if args.style_target == "target":
                        e_t = B_xstyle.embed_region(vae_model, lf, embed, x0, box, args.n_crops)
                    else:   # another post by the reference artist (never the target), random crop of the same side
                        gal = []
                        for pr in batch:
                            ids = [i for i in by_artist[recs[pr[0]]["artist"]] if i != pr[0]] or [pr[1]]
                            g = prep.latent(crop_rng.choice(ids)).to(device)[None]
                            gb = B_xstyle.crop_boxes(g.shape[-2], g.shape[-1], box[2] // (2 if args.n_crops == 4 else 1),
                                                     args.n_crops, crop_gen)
                            gal.append(B_xstyle.embed_region(vae_model, lf, embed, g, gb, args.n_crops))
                        e_t = torch.cat(gal)
                    x0_hat = xt.float() - s * pred.detach().float()
                l_st, g_x0 = B_xstyle.style_grad(vae_model, lf, embed, x0_hat, box, e_t, args.n_crops)
                # d x0_hat / d v_hat = -sigma; mean over the batch, samples outside the sigma window get no gradient
                g_v = -s * g_x0 * gate.view(-1, 1, 1, 1, 1) / len(batch)
                aux = args.lam / min(1.0, args.style_p) * (pred.float() * g_v).sum()
                for k in range(len(batch)):
                    if gate[k] > 0:
                        running["xstyle"].append(float(l_st[k]))
                        running["style_sigma"].append(float(sigma[k]))
            if aux is not None and measure:
                gf = torch.autograd.grad(total / accum, params, retain_graph=True, allow_unused=True)
                gs = torch.autograd.grad(aux / accum, params, allow_unused=True)
                running["gnorm_fm"].append(lora_grad_norm(gf))
                running["gnorm_style"].append(lora_grad_norm(gs))
                for p, a, b in zip(params, gf, gs):
                    for gg in (a, b):
                        if gg is not None:
                            p.grad = gg.clone() if p.grad is None else p.grad + gg
                measure = False
            else:
                ((total + (aux if aux is not None else 0.0)) / accum).backward()
            running["loss_" + ("noref" if no_ref else mode)].append(loss.item())
        for p in params:
            if p.grad is None:
                p.grad = torch.zeros_like(p)
        grad_norm = torch.nn.utils.clip_grad_norm_(params, tcfg["grad_clip"])
        optimizer.step()

        done = step + 1
        if done % tcfg["log_every"] == 0 or done == end_step:
            now = time.perf_counter()
            line = {"step": done, **{k: round(sum(v) / len(v), 5) for k, v in running.items()},
                    "n_xstyle": len(running.get("xstyle", [])), "lr": optimizer.param_groups[0]["lr"],
                    "grad_norm": round(float(grad_norm), 4),
                    "s_per_step": round((now - window) / tcfg["log_every"], 2),
                    "elapsed_h": round((now - started) / 3600, 3),
                    "max_mem_gb": round(torch.cuda.max_memory_allocated(device) / 2**30, 1)}
            if "gnorm_fm" in running:
                line["gratio"] = round(line["gnorm_style"] / max(1e-12, line["gnorm_fm"]), 4)
            log(json.dumps(line))
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(line) + "\n")
            running, window = collections.defaultdict(list), time.perf_counter()
        if (done % tcfg["save_every"] == 0 or done == end_step) and not args.smoke:
            asym.save(modules, args.out / f"adapter_{done:06d}.safetensors", {**metadata, "step": done},
                      {"mode_embed": mode_embed})
            torch.save({"step": done, "optimizer": optimizer.state_dict(), "rng": [rng.getstate()],
                        "sampler_rng": [sampler.rng.getstate()], "torch_rng": [torch.get_rng_state()],
                        "crop_gen": crop_gen.get_state(), "crop_rng": crop_rng.getstate()},
                       args.out / f"state_{done:06d}.pt")   # all kept (no deletion rule; ~1 GB each)
    log(f"training done in {(time.perf_counter() - started) / 3600:.2f} h")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
