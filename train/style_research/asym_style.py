"""Research copy of asym.py's attention hook with extra state keys (asym.py itself stays untouched).

install(dm) replaces every block's self_attn.forward, like asym._install_hooks, and marks the blocks as installed so
asym.encode_reference / asym.target_forward keep this forward. State keys (attn._asym_state, set per block by
asym._block_fn):
  {"record": list}            store (k, v) after norm + RoPE (asym.py)
  {"cache": list}             prepend the cached (k, v); a None entry = this block reads no reference (asym.py)
  {"cache": list, "stats": S} also measure, for S["qidx"] (query token indices), per head:
      mass   softmax mass on the reference keys,
      ent    entropy of the attention restricted to the reference keys / log(N_ref),
      local  share of the reference mass on keys within S["radius"] patches of the query's aligned position
             (S["ref_pos"] [N_ref, 2] and S["tgt_pos"] [N_tgt, 2] in target patch coordinates);
      appended to S["rows"] as (block, mass[H], ent[H], local[H]) averaged over the sampled queries.
"""
from __future__ import annotations

import math

import torch

import asym


def _stats(q, k, n_ref: int, stats: dict, index: int) -> None:
    idx = stats["qidx"].to(q.device)
    qs = q[:, idx].float()                                   # [B, n, H, D]
    logits = torch.einsum("bnhd,bmhd->bhnm", qs, k.float()) / math.sqrt(q.shape[-1])
    p = logits.softmax(-1)                                   # [B, H, n, M]
    pr = p[..., :n_ref]
    mass = pr.sum(-1)                                        # [B, H, n]
    prn = pr / mass.unsqueeze(-1).clamp_min(1e-12)
    ent = -(prn * prn.clamp_min(1e-12).log()).sum(-1) / math.log(n_ref)
    rp, tp = stats["ref_pos"].to(q.device).float(), stats["tgt_pos"].to(q.device).float()[idx]
    near = ((tp[:, None, :] - rp[None, :, :]).abs().amax(-1) <= stats["radius"]).float()   # [n, N_ref]
    local = (prn * near[None, None]).sum(-1)
    stats["rows"].append((index, mass.mean((0, 2)).cpu(), ent.mean((0, 2)).cpu(), local.mean((0, 2)).cpu()))


def install(dm) -> None:
    for index, block in enumerate(dm.blocks):
        attn = block.self_attn
        if getattr(attn, "_style_installed", False):
            continue
        attn._asym_installed = True    # asym._install_hooks leaves these blocks alone
        attn._style_installed = True
        attn._asym_index = index
        attn._asym_state = None

        def forward(x, context=None, rope_emb=None, transformer_options={}, _attn=attn):
            q, k, v = _attn.compute_qkv(x, context, rope_emb=rope_emb, transformer_options=transformer_options)
            state = _attn._asym_state
            i = _attn._asym_index
            if state is not None and "record" in state:
                state["record"][i] = (k, v)
            elif state is not None and state.get("cache") is not None and state["cache"][i] is not None:
                k_ref, v_ref = state["cache"][i]
                if k_ref.shape[0] != k.shape[0]:
                    k_ref = k_ref.expand(k.shape[0], *k_ref.shape[1:])
                    v_ref = v_ref.expand(v.shape[0], *v_ref.shape[1:])
                k = torch.cat([k_ref.to(k.dtype), k], dim=1)
                v = torch.cat([v_ref.to(v.dtype), v], dim=1)
                if state.get("stats") is not None:
                    _stats(q, k, k_ref.shape[1], state["stats"], i)
            return _attn.compute_attention(q, k, v, transformer_options=transformer_options)

        attn.forward = forward


def target_forward(dm, x: torch.Tensor, sigma: torch.Tensor, context: torch.Tensor, cache: list | None,
                   checkpoint_blocks: int = 0, stats: dict | None = None) -> torch.Tensor:
    """asym.target_forward with the extra state keys of this module (stats)."""
    install(dm)
    x_emb, rope, extra = dm.prepare_embedded_sequence(x, fps=None, padding_mask=None)
    if extra is not None:
        raise NotImplementedError("extra per-block position embeddings are not supported")
    sigma = sigma.to(device=x.device, dtype=torch.float32).reshape(x.shape[0], 1)
    t_emb, adaln = dm.t_embedder[1](dm.t_embedder[0](sigma).to(x_emb.dtype))
    t_emb = dm.t_embedding_norm(t_emb)
    state = {"cache": cache, "stats": stats}
    x_emb = asym._run_blocks(dm, x_emb, t_emb, adaln, context, rope, checkpoint_blocks, "tgt", state)
    out = dm.final_layer(x_emb.to(context.dtype), t_emb, adaln_lora_B_T_3D=adaln)
    return dm.unpatchify(out)[:, :, :1]


def grid_positions(target_hw: tuple[int, int], ref_hw: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
    """Patch coordinates of the target tokens and of the reference tokens as encode_reference places them
    (centre-aligned, offset 0). target_hw / ref_hw are latent sizes (8x downsampled), patches are 2x2."""
    ht, wt = target_hw[0] // 2, target_hw[1] // 2
    hr, wr = ref_hw[0] // 2, ref_hw[1] // 2
    h0 = round((target_hw[0] // 2) / 2 - hr / 2)
    w0 = round((target_hw[1] // 2) / 2 - wr / 2)
    tg = torch.stack(torch.meshgrid(torch.arange(ht), torch.arange(wt), indexing="ij"), -1).reshape(-1, 2)
    rg = torch.stack(torch.meshgrid(torch.arange(hr) + h0, torch.arange(wr) + w0, indexing="ij"), -1).reshape(-1, 2)
    return tg, rg


