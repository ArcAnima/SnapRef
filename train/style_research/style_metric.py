"""Learned style metric (style_dl): style_probe.py's DINOv2 layer statistics + its linear head.

Validated on artists the probe never saw (A10, style_probe_test.py / style_probe_whole.py): 81.1% triplet accuracy
(1,885 sets, CI 79.3-82.9), 84.3% when the anchor is a dark scene and the positive daylight, 56.8% top-1 artist
retrieval among the 33 eval artists (chance 2.3%).

    from style_metric import StyleDL, artist_scores
    m = StyleDL(probe_path, device); e = m.emb(pil_image)      # [256], L2-normalised
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from style_probe import preprocess  # noqa: E402

DEFAULT_PROBE = Path.home() / "style_agent" / "probe" / "probe_dinov2_linear.pt"


class StyleDL:
    def __init__(self, probe_path: Path = DEFAULT_PROBE, device="cuda"):
        from transformers import AutoModel
        p = torch.load(probe_path, map_location="cpu", weights_only=True)
        self.spec, self.device = p["spec"], device
        self.model = AutoModel.from_pretrained(self.spec["repo"]).to(device).eval().half()
        d = p["dims"]
        self.head = (torch.nn.Linear(d, 256) if p["head"] == "linear" else torch.nn.Sequential(
            torch.nn.Linear(d, 1024), torch.nn.GELU(), torch.nn.Dropout(0.1), torch.nn.Linear(1024, 256)))
        self.head.load_state_dict(p["state"])
        self.head = self.head.to(device).eval()
        self.mu, self.sd = p["mu"].to(device), p["sd"].to(device)

    @torch.no_grad()
    def feats(self, x: torch.Tensor) -> torch.Tensor:
        """x [B, 3, S, S] normalised pixels -> raw statistics [B, dims]."""
        hs = self.model(pixel_values=x.to(self.device, torch.float16), output_hidden_states=True).hidden_states
        parts = []
        for layer in self.spec["layers"]:
            t = hs[layer][:, self.spec["skip_tokens"]:].float()
            parts += [t.mean(1), t.std(1)]
        return torch.cat(parts, 1)

    @torch.no_grad()
    def project(self, f: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.head((f.float() - self.mu) / self.sd), dim=-1)

    def tensor(self, img: Image.Image) -> torch.Tensor:
        return preprocess(img.convert("RGB"), self.spec["size"], self.spec["mean"], self.spec["std"])

    @torch.no_grad()
    def emb(self, img: Image.Image) -> torch.Tensor:
        return self.project(self.feats(self.tensor(img)[None]))[0]

    @torch.no_grad()
    def emb_many(self, imgs: list[Image.Image], batch: int = 32) -> torch.Tensor:
        out = []
        for i in range(0, len(imgs), batch):
            out.append(self.project(self.feats(torch.stack([self.tensor(im) for im in imgs[i:i + batch]]))))
        return torch.cat(out)


def artist_scores(e: torch.Tensor, artist: str, gallery: dict[str, torch.Tensor]) -> dict:
    """e [256]; gallery artist -> [n, 256] of that artist's posts (never the reference itself).
    style = mean cos to the artist's posts - mean cos to all other artists' posts; rank = position of the artist
    when the 33 artists are sorted by mean cos (1 = best)."""
    means = {a: float((g @ e).mean()) for a, g in gallery.items()}
    own = means[artist]
    other = float(torch.cat([g @ e for a, g in gallery.items() if a != artist]).mean())
    rank = 1 + sum(v > own for a, v in means.items() if a != artist)
    return {"style": own - other, "own": own, "rank": rank}


def paired_ci(diff: np.ndarray, groups: list, n_boot: int = 2000, seed: int = 0) -> list[float]:
    """95% bootstrap CI of the mean of diff, resampling groups (e.g. artists) with replacement."""
    rng = np.random.default_rng(seed)
    keys = sorted(set(groups))
    idx = {k: [i for i, g in enumerate(groups) if g == k] for k in keys}
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(len(keys), len(keys))
        boots.append(np.mean([diff[i] for k in pick for i in idx[keys[k]]]))
    return [round(float(x), 4) for x in np.percentile(boots, [2.5, 97.5])]
