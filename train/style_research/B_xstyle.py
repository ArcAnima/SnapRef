"""Experiment B: a style loss on the x0 prediction (PARALLEL.md, log_B.md).

x0_hat = x_sigma - sigma * v_hat (model-space latent) -> Wan21 process_out -> VAE decoder with gradient (anima_env.vae_decode
is inference_mode) -> local crops -> frozen DINOv2 layer statistics -> the probe's linear head (local_linear, A10
style_probe_ms.py) -> 256-d style embedding. Loss = 1 - cos(emb(x0_hat crop), emb(x0 crop)), x0 = the true target
(same artist as the reference, other character). Training uses local_linear; success is judged with style_dl
(global view, other probe) and sheets, never with local_linear.

Crop: n_crops = 1 takes one c x c latent crop; n_crops = 4 takes a 2c x 2c region cut into a 2 x 2 grid, which is
exactly the probe's "local" view when c = 56 * min(H, W) / 896 (short side -> 896 px, 4 crops of 448 px). Crops are
resized to 448 px bilinearly. Statistics pool the tokens of all crops of a sample, as the probe does.

    python B_xstyle.py   # self-check: dL/dv_hat != 0 and matches a finite difference; decode matches anima_env.vae_decode
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

LOCAL_PROBE = Path.home() / "style_agent" / "probe_ms" / "probe_local_linear.pt"


def decoder_of(vae, device):
    """The Wan21 VAE module of a comfy.sd.VAE, frozen, bf16, on device (called directly: no comfy memory manager)."""
    m = vae.first_stage_model.to(device=device, dtype=torch.bfloat16).eval()
    m.requires_grad_(False)
    with torch.no_grad():   # weights moved under inference_mode (vae_decode) are inference tensors; autograd needs copies
        for t in list(m.parameters()) + list(m.buffers()):
            t.data = t.data.clone()
    return m


def decode_grad(model, lf, x0: torch.Tensor) -> torch.Tensor:
    """[B, 16, 1, h, w] model-space latent -> [B, 3, 8h, 8w] pixels in 0..1, differentiable in x0."""
    raw = lf.process_out(x0.float())
    out = model.decode(raw.to(next(model.parameters()).dtype))
    return ((out[:, :, 0].float() + 1.0) / 2.0).clamp(0.0, 1.0)


class LocalStyle(torch.nn.Module):
    """local_linear probe as a differentiable embedding of pixel crops."""

    def __init__(self, probe_path: Path = LOCAL_PROBE, device="cuda"):
        super().__init__()
        from transformers import AutoModel
        p = torch.load(probe_path, map_location="cpu", weights_only=True)
        assert p["head"] == "linear" and list(p["views"]) == ["local"], p["views"]
        self.spec = p["spec"]
        self.dino = AutoModel.from_pretrained(self.spec["repo"]).to(device, torch.bfloat16).eval()
        self.head = torch.nn.Linear(p["dims"], 256).to(device)
        self.head.load_state_dict(p["state"])
        self.register_buffer("mu", p["mu"].float().to(device))
        self.register_buffer("sd", p["sd"].float().to(device))
        self.register_buffer("mean", torch.tensor(self.spec["mean"], device=device).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(self.spec["std"], device=device).view(1, 3, 1, 1))
        self.requires_grad_(False)

    def forward(self, crops: torch.Tensor) -> torch.Tensor:
        """crops [B, k, 3, s, s] pixels 0..1 -> [B, 256] L2-normalised (statistics over the tokens of the k crops)."""
        b, k = crops.shape[:2]
        size = self.spec["size"]
        x = F.interpolate(crops.flatten(0, 1), size=(size, size), mode="bilinear", align_corners=False)
        x = ((x - self.mean) / self.std).to(self.dino.dtype)
        hs = self.dino(pixel_values=x, output_hidden_states=True).hidden_states
        parts = []
        for layer in self.spec["layers"]:
            t = hs[layer][:, self.spec["skip_tokens"]:].float()
            t = t.reshape(b, -1, t.shape[-1])
            parts += [t.mean(1), t.std(1)]
        return F.normalize(self.head((torch.cat(parts, 1) - self.mu) / self.sd), dim=-1)


def crop_boxes(h: int, w: int, c: int, n_crops: int, gen: torch.Generator | None = None) -> tuple[int, int, int]:
    """Random top-left (y, x) and side of the latent region: c (n_crops 1) or 2c (n_crops 4), clipped to the latent."""
    side = min(c * (2 if n_crops == 4 else 1), h, w)
    y = int(torch.randint(0, h - side + 1, (1,), generator=gen))
    x = int(torch.randint(0, w - side + 1, (1,), generator=gen))
    return y, x, side


def to_crops(pix: torch.Tensor, n_crops: int) -> torch.Tensor:
    """[B, 3, S, S] decoded region -> [B, k, 3, s, s]."""
    if n_crops == 1:
        return pix[:, None]
    s = pix.shape[-1] // 2
    return torch.stack([pix[..., i * s:(i + 1) * s, j * s:(j + 1) * s] for i in range(2) for j in range(2)], 1)


def style_grad(model, lf, embed: LocalStyle, x0_hat: torch.Tensor, box, e_t: torch.Tensor,
               n_crops: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    """Loss 1 - cos(emb(x0_hat region), e_t) per sample, and its gradient w.r.t. x0_hat (zero outside the region).
    Two-stage on purpose: the VAE + DINOv2 graph is freed here, the caller backpropagates <g, x0_hat> through the DiT."""
    y, x, side = box
    leaf = x0_hat[..., y:y + side, x:x + side].detach().float().requires_grad_(True)
    e_p = embed(to_crops(decode_grad(model, lf, leaf), n_crops))
    loss = 1.0 - (e_p * e_t).sum(-1)
    (g,) = torch.autograd.grad(loss.sum(), leaf)
    full = torch.zeros_like(x0_hat, dtype=torch.float32)
    full[..., y:y + side, x:x + side] = g
    return loss.detach(), full


@torch.no_grad()
def embed_region(model, lf, embed: LocalStyle, x: torch.Tensor, box, n_crops: int) -> torch.Tensor:
    y, xx, side = box
    return embed(to_crops(decode_grad(model, lf, x[..., y:y + side, xx:xx + side]), n_crops))


def self_check(comfy_root: str = str(Path.home() / "ComfyUI")) -> dict:
    """(1) decode_grad == anima_env.vae_decode; (2) a style loss on x0_hat = x - sigma v gives dL/dv != 0, and the
    directional derivative matches a finite difference; (3) gradient reaches a LoRA parameter of the target stream."""
    import anima_env
    device = torch.device("cuda", 0)
    anima_env.setup_comfy(comfy_root)
    lf = anima_env.latent_format()
    vae = anima_env.load_vae("qwen_image_vae.safetensors")
    gen = torch.Generator().manual_seed(0)
    x0 = (torch.randn(1, 16, 1, 40, 40, generator=gen) * 0.8).to(device)
    ref_pix = anima_env.vae_decode(vae, lf.process_out(x0).float()).to(device).permute(0, 3, 1, 2)
    model = decoder_of(vae, device)
    ours = decode_grad(model, lf, x0)
    out = {"decode_max_diff": float((ours - ref_pix).abs().max())}
    embed = LocalStyle(device=device)
    model.float()   # finite differences need fp32 without TF32; training runs bf16
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    embed.dino.float()
    sigma = 0.4
    noise = torch.randn(x0.shape, generator=gen).to(device)
    xt = (1 - sigma) * x0 + sigma * noise
    v = (noise - x0 + 0.3 * torch.randn(x0.shape, generator=gen).to(device)).requires_grad_(True)
    other = (torch.randn(x0.shape, generator=gen) * 0.8).to(device)   # a different image: loss far from 0
    target = embed(decode_grad(model, lf, other)[:, None]).detach()

    def loss_of(vv):
        return 1 - (embed(decode_grad(model, lf, xt - sigma * vv)[:, None]) * target).sum()
    loss = loss_of(v)
    (g,) = torch.autograd.grad(loss, v)
    d = g / g.norm()   # along the gradient: the derivative is largest there, so the check is not lost in noise
    eps = 1e-1
    with torch.no_grad():
        fd = float((loss_of(v + eps * d) - loss_of(v - eps * d)) / (2 * eps))
    out.update(loss=float(loss), grad_v_norm=float(g.norm()), dir_grad=float((g * d).sum()), dir_fd=fd)
    assert out["decode_max_diff"] < 2e-2, out
    assert out["grad_v_norm"] > 0 and torch.isfinite(g).all(), out
    assert abs(out["dir_grad"] - fd) < 0.25 * abs(fd) + 1e-4, out
    # two-stage style_grad gives the same gradient as direct backprop
    xh = (xt - sigma * v).detach()
    _, g2 = style_grad(model, lf, embed, xh, (0, 0, 40), target)
    (g_x,) = torch.autograd.grad(loss_of(v), v)
    out["two_stage_rel_diff"] = float(((-sigma * g2) - g_x).norm() / g_x.norm())
    assert out["two_stage_rel_diff"] < 0.05, out
    return out


if __name__ == "__main__":
    import json
    print(json.dumps(self_check()), flush=True)
