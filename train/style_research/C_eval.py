"""Run an existing eval script (path relative to train/, without .py) on a C2 adapter.

    CUDA_VISIBLE_DEVICES=1 python train/style_research/C_eval.py eval_style_v2 --adapter RUN/adapter_000500.safetensors --out OUT
    CUDA_VISIBLE_DEVICES=1 python train/style_research/C_eval.py style_research/A_eval_identity --adapter ... --out ...

The eval scripts call asym.inject and asym.load; here those are wrapped so that, when the adapter file holds
anima_ref_kv.c2_* tensors, a C_style_path.StylePath is built, installed on the model and filled from the file. Without
c2_* tensors (v1a, C1 runs) nothing changes. Everything else (cases, seeds, scoring) is the eval script's own code.
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

from safetensors import safe_open

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import asym  # noqa: E402
import C_style_path  # noqa: E402

_inject, _load = asym.inject, asym.load
_dm = {}


def inject(dm, cfg):
    _dm["dm"] = dm
    return _inject(dm, cfg)


def load(modules, path, extra=None):
    with safe_open(str(path), "pt") as h:
        keys = set(h.keys())
        if "anima_ref_kv.c2_gate" in keys:
            down = h.get_tensor("anima_ref_kv.c2_down")
            dm = _dm["dm"]
            p = C_style_path.StylePath(down.shape[0], dm.model_channels, down.shape[1], next(dm.parameters()).device)
            p.requires_grad_(False)
            extra = {**(extra or {}), **p.tensors()}
            C_style_path.install(dm, p)
            print(f"C_eval: style path rank {down.shape[1]}, gates {h.get_tensor('anima_ref_kv.c2_gate').tolist()}", flush=True)
    return _load(modules, path, extra)


asym.inject, asym.load = inject, load

if __name__ == "__main__":
    script = sys.argv.pop(1)
    sys.argv[0] = str(HERE.parent / f"{script}.py")
    runpy.run_path(sys.argv[0], run_name="__main__")
