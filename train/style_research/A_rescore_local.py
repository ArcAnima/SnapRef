"""rescore_style.py with the local_linear probe (style_probe_ms.py: DINOv2 statistics over 4 crops of 448 px from the
896 px centre) instead of style_dl. The secondary style metric of PARALLEL.md §3; same gallery, rows and CI.

    python train/style_research/A_rescore_local.py --renders DIR[,DIR...] --out OUT.json \
        --probe ~/style_agent/probe_ms/probe_local_linear.pt
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rescore_style  # noqa: E402
from style_metric import StyleDL  # noqa: E402
from style_probe_ms import local_views  # noqa: E402


class LocalDL(StyleDL):
    def tensor(self, img):
        return local_views(img.convert("RGB"), self.spec["size"])        # [4, 3, 448, 448]

    @torch.no_grad()
    def feats(self, x: torch.Tensor) -> torch.Tensor:
        """x [B, 4, 3, S, S] -> statistics over the tokens of the 4 crops together (extract_local)."""
        b = x.shape[0]
        hs = self.model(pixel_values=x.flatten(0, 1).to(self.device, torch.float16), output_hidden_states=True).hidden_states
        parts = []
        for layer in self.spec["layers"]:
            t = hs[layer][:, self.spec["skip_tokens"]:].float()
            t = t.reshape(b, -1, t.shape[-1])
            parts += [t.mean(1), t.std(1)]
        return torch.cat(parts, 1)


if __name__ == "__main__":
    rescore_style.StyleDL = LocalDL
    raise SystemExit(rescore_style.main())
