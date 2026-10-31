"""Experiment C2: a dedicated style path next to the joint reference softmax (asym.py stays untouched).

While the reference stream runs (sigma 0, once per reference), every block also records the channel mean and std over
the reference tokens of its self-attention input (the adaLN-modulated LayerNorm output), next to K/V: cache entries
become (k, v, mean, std). In the target stream each block b turns those statistics into a scale/shift of its own
self-attention input, the same form as the block's adaLN:

    x' = x * (1 + g_b * scale_b) + g_b * shift_b,   (scale_b, shift_b) = up_b(SiLU(down_b(LN([mean_b; std_b]))))

g_b is one learnable scalar gate per block, starting at 0: with every g_b = 0, x' == x bit for bit and the forward is
exactly asym.py's (gate_zero_check). The statistics carry no positions (RoPE does not reach them), so the path can only
pass global statistics, which is what D1 says the style gain is (low-frequency colour and tone, read uniformly by
blocks 0-1). Cost per target step: one [B, 4096] -> rank -> [B, 4096] MLP per block, nothing per token beyond one
multiply-add.

install(dm, path) replaces each block's self_attn.forward (marks it asym-installed so asym.encode_reference /
asym.target_forward / asym.sample keep it). path=None or path.enabled False: the hook behaves exactly like asym.py's
and cache entries may be plain (k, v).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import asym


class StylePath(nn.Module):
    def __init__(self, n_blocks: int, dim: int, rank: int, device=None):
        super().__init__()
        self.down = nn.Parameter(torch.empty(n_blocks, rank, 2 * dim, device=device))
        self.up = nn.Parameter(torch.empty(n_blocks, 2 * dim, rank, device=device))
        nn.init.normal_(self.down, std=(2 * dim) ** -0.5)
        nn.init.normal_(self.up, std=rank ** -0.5 * 0.1)   # |scale|, |shift| ~ 0.1 per unit gate
        self.gate = nn.Parameter(torch.zeros(n_blocks, device=device))
        self.dim = dim
        self.enabled = True

    def modulation(self, i: int, mean: torch.Tensor, std: torch.Tensor):
        """[B, D] statistics of block i -> (scale, shift) [B, 1, D] already multiplied by the gate."""
        s = torch.cat([mean, std], -1).float()
        h = F.silu(F.linear(F.layer_norm(s, s.shape[-1:]), self.down[i]))
        out = F.linear(h, self.up[i]) * self.gate[i]
        return out[:, None, :self.dim], out[:, None, self.dim:]

    def tensors(self) -> dict[str, torch.Tensor]:
        """For asym.save / asym.load extra= (keys anima_ref_kv.c2_*)."""
        return {"c2_down": self.down, "c2_up": self.up, "c2_gate": self.gate}


def install(dm, path: StylePath | None) -> None:
    for index, block in enumerate(dm.blocks):
        attn = block.self_attn
        attn._asym_installed = True    # asym._install_hooks leaves these blocks alone
        attn._asym_index = index
        attn._asym_state = getattr(attn, "_asym_state", None)
        attn._c2_path = path

        def forward(x, context=None, rope_emb=None, transformer_options={}, _attn=attn):
            state = _attn._asym_state
            i = _attn._asym_index
            p = _attn._c2_path
            on = p is not None and p.enabled
            entry = None
            if state is not None and "record" not in state and state.get("cache") is not None:
                entry = state["cache"][i]   # None: this block reads no reference (block probes)
            if on and entry is not None and len(entry) == 4:
                mean, std = entry[2], entry[3]
                if mean.shape[0] != x.shape[0]:
                    mean, std = mean.expand(x.shape[0], -1), std.expand(x.shape[0], -1)
                scale, shift = p.modulation(i, mean, std)
                x = (x.float() * (1 + scale) + shift).to(x.dtype)
            q, k, v = _attn.compute_qkv(x, context, rope_emb=rope_emb, transformer_options=transformer_options)
            if state is not None and "record" in state:
                if on:
                    xf = x.float()
                    state["record"][i] = (k, v, xf.mean(1), xf.std(1))
                else:
                    state["record"][i] = (k, v)
            elif entry is not None:
                k_ref, v_ref = entry[0], entry[1]
                if k_ref.shape[0] != k.shape[0]:
                    k_ref = k_ref.expand(k.shape[0], *k_ref.shape[1:])
                    v_ref = v_ref.expand(v.shape[0], *v_ref.shape[1:])
                k = torch.cat([k_ref.to(k.dtype), k], dim=1)
                v = torch.cat([v_ref.to(v.dtype), v], dim=1)
            return _attn.compute_attention(q, k, v, transformer_options=transformer_options)

        attn.forward = forward


@torch.no_grad()
def gate_zero_check(dm, path: StylePath, device) -> dict:
    """With all gates 0 the output must equal the plain asym path bit for bit; with gates != 0 it must change."""
    gen = torch.Generator(device="cpu").manual_seed(3)
    x = torch.randn(2, 16, 1, 16, 24, generator=gen).to(device, torch.bfloat16)
    ref = torch.randn(2, 16, 1, 12, 12, generator=gen).to(device, torch.bfloat16)
    ctx = torch.randn(2, 512, 1024, generator=gen).to(device, torch.bfloat16) * 0.1
    sigma = torch.tensor([0.9, 0.3], device=device)
    saved_gate = path.gate.detach().clone()
    path.enabled = False
    plain = asym.target_forward(dm, x, sigma, ctx, asym.encode_reference(dm, ref, ctx, (16, 24))).float()
    path.enabled = True
    path.gate.zero_()
    cache = asym.encode_reference(dm, ref, ctx, (16, 24))
    assert all(len(e) == 4 for e in cache)
    zero = asym.target_forward(dm, x, sigma, ctx, cache).float()
    path.gate.fill_(1.0)
    one = asym.target_forward(dm, x, sigma, ctx, cache).float()
    path.gate.copy_(saved_gate)
    d0 = (zero - plain).abs().max().item()
    d1 = (one - plain).abs().max().item()
    if d0 != 0.0:
        raise RuntimeError(f"C2 with gate 0 differs from asym: {d0}")
    if not d1 > 0:
        raise RuntimeError("C2 with gate 1 does not change the output")
    return {"gate0_vs_asym": d0, "gate1_vs_asym": d1}


def grad_check(dm, path: StylePath, device) -> dict:
    """C2 parameter gradients with and without gradient checkpointing must match."""
    gen = torch.Generator(device="cpu").manual_seed(4)
    x = torch.randn(1, 16, 1, 16, 16, generator=gen).to(device, torch.bfloat16)
    ref = torch.randn(1, 16, 1, 16, 12, generator=gen).to(device, torch.bfloat16)
    ctx = torch.randn(1, 512, 1024, generator=gen).to(device, torch.bfloat16) * 0.1
    sigma = torch.tensor([0.5], device=device)
    saved_gate = path.gate.detach().clone()
    with torch.no_grad():
        path.gate.fill_(0.5)   # gate 0 leaves down/up without gradient
    grads = []
    for ckpt in (0, len(dm.blocks)):
        path.zero_grad(set_to_none=True)
        cache = asym.encode_reference(dm, ref, ctx, (16, 16), checkpoint_blocks=ckpt)
        asym.target_forward(dm, x, sigma, ctx, cache, checkpoint_blocks=ckpt).float().pow(2).mean().backward()
        grads.append(torch.cat([p.grad.flatten().float() for p in path.parameters()]))
    path.zero_grad(set_to_none=True)
    with torch.no_grad():
        path.gate.copy_(saved_gate)
    rel = ((grads[0] - grads[1]).norm() / grads[0].norm().clamp_min(1e-12)).item()
    if not rel < 2e-2:
        raise RuntimeError(f"C2 checkpointed gradients differ: rel {rel}")
    return {"c2_ckpt_grad_rel_diff": rel, "c2_grad_norm": grads[0].norm().item()}
