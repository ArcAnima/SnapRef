"""C2 checks on the real model: gate-zero equivalence and checkpointed gradients (C_style_path), and with --timing
the generation time (asym.sample, 28 steps, 1216x832, CFG, reference cache built once) for: no reference, v1a
(K/V only), v1a + C2 path (gate 1).

    CUDA_VISIBLE_DEVICES=1 python train/style_research/C_check.py [--timing] [--out result.json]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import anima_env  # noqa: E402
import asym  # noqa: E402
import C_style_path  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1a", type=Path, default=Path.home() / "adapters" / "v1a" / "snapref_v1a.safetensors")
    ap.add_argument("--config", type=Path, default=HERE.parent.parent / "configs" / "train_pilot2_ft.json")
    ap.add_argument("--comfy-root", default=str(Path.home() / "ComfyUI"))
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--timing", action="store_true")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    dev = torch.device("cuda", 0)
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    anima_env.setup_comfy(args.comfy_root)
    _, dm = anima_env.load_diffusion_model(cfg["comfy"]["unet_name"], dev)
    modules = asym.inject(dm, cfg["adapter"]["lora"])
    E = torch.zeros(2, dm.model_channels, device=dev)
    asym.load(modules, args.v1a, {"mode_embed": E})
    path = C_style_path.StylePath(len(dm.blocks), dm.model_channels, args.rank, dev)
    C_style_path.install(dm, path)
    res = {"gate_zero": C_style_path.gate_zero_check(dm, path, dev), "grad": C_style_path.grad_check(dm, path, dev)}
    print(json.dumps(res), flush=True)
    if args.timing:
        ctx = torch.randn(300, 1024, generator=torch.Generator().manual_seed(0)).to(dev, torch.bfloat16) * 0.1
        ref = torch.randn(1, 16, 1, 1216 // 8, 832 // 8, generator=torch.Generator().manual_seed(1)).to(dev, torch.bfloat16)
        shape = (1, 16, 1, 1216 // 8, 832 // 8)
        times = {}
        with torch.no_grad():
            path.gate.fill_(1.0)
        for name, r, on in (("noref", None, False), ("v1a", ref, False), ("v1a_c2", ref, True), ("noref", None, False)):
            path.enabled = on
            asym.sample(dm, ctx, ctx, ctx, shape, 0, 2, 4.5, 3.0, dev, r, mode_emb=E[:1] if r is not None else None)
            ts = []
            for _ in range(args.reps):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                asym.sample(dm, ctx, ctx, ctx, shape, 0, 28, 4.5, 3.0, dev, r, mode_emb=E[:1] if r is not None else None)
                torch.cuda.synchronize()
                ts.append(time.perf_counter() - t0)
            times.setdefault(name, []).extend(ts)
        res["seconds"] = {k: round(min(v), 3) for k, v in times.items()}
        res["ratio_vs_noref"] = {k: round(v / res["seconds"]["noref"], 3) for k, v in res["seconds"].items()}
        print(json.dumps(res), flush=True)
    if args.out:
        args.out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
