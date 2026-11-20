"""CPU regressions for the masked attention fallback (the path MPS / XPU / CPU / fp32 take).

MPS turned the -inf entries of the additive mask (key padding to a multiple of 16, rows that skip the reference)
into black images. Off CUDA the mask must stay finite, and the result must not change: every case is compared
with the pre-fix implementation (kept below verbatim) and with a float64 softmax written out by hand.
"""
import math
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from test_model_reload import load_snapref

snapref = load_snapref()


def reference_masked_attention(q, k_t, v_t, kv_ref, sizes, biases, rows=None, row_biases=None):
    """_masked_attention as it was before the fix (padding and -inf on every device)."""
    k_r, v_r = kv_ref
    B, S, H, D = q.shape
    k = torch.cat([k_r.expand(B, *k_r.shape[1:]).to(q.dtype), k_t], dim=1).transpose(1, 2)
    v = torch.cat([v_r.expand(B, *v_r.shape[1:]).to(q.dtype), v_t], dim=1).transpose(1, 2)
    n = k.shape[2]
    pad = (-n) % 16
    if pad:
        k, v = F.pad(k, (0, 0, 0, pad)), F.pad(v, (0, 0, 0, pad))
    mask = torch.zeros(B if rows is not None else 1, 1, 1, n + pad, device=q.device, dtype=q.dtype)
    start = 0
    for size, bias in zip(sizes, biases):
        mask[..., start:start + size] = bias
        start += size
    if rows is not None:
        selected = torch.zeros(B, device=q.device, dtype=torch.bool)
        selected[rows] = True
        mask[~selected, ..., :start] = float("-inf")
        if row_biases is not None:
            mask[rows, ..., :start] += torch.as_tensor(row_biases, device=q.device, dtype=q.dtype).view(-1, 1, 1, 1)
    if pad:
        mask[..., n:] = float("-inf")
    out = F.scaled_dot_product_attention(q.transpose(1, 2), k, v, attn_mask=mask.expand(B, H, S, n + pad))
    return out.transpose(1, 2).reshape(B, S, H * D)


def exact(q, k_t, v_t, kv_ref, sizes, biases, rows=None, row_biases=None):
    """softmax over [K_ref_i + b_i (+ row bias); K_t] per batch row, float64, no SDPA."""
    q, k_t, v_t = q.double(), k_t.double(), v_t.double()
    k_r, v_r = (t.double()[0] for t in kv_ref)
    B, S, H, D = q.shape
    ref_bias = torch.cat([torch.full((s,), float(b), dtype=torch.float64) for s, b in zip(sizes, biases)])
    out = []
    for b in range(B):
        reads = rows is None or b in rows
        if reads:
            extra = 0.0 if row_biases is None else row_biases[list(rows).index(b)]
            keys, vals = torch.cat([k_r, k_t[b]]), torch.cat([v_r, v_t[b]])
            bias = torch.cat([ref_bias + extra, torch.zeros(k_t.shape[1], dtype=torch.float64)])
        else:
            keys, vals, bias = k_t[b], v_t[b], torch.zeros(k_t.shape[1], dtype=torch.float64)
        logits = torch.einsum("shd,khd->hsk", q[b], keys) / math.sqrt(D) + bias
        out.append(torch.einsum("hsk,khd->shd", logits.softmax(-1), vals).reshape(S, H * D))
    return torch.stack(out)


def inputs(B, S, ref_sizes, H=2, D=16, dtype=torch.float32, seed=0):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g).to(dtype)   # noqa: E731
    return r(B, S, H, D), r(B, S, H, D), r(B, S, H, D), (r(1, sum(ref_sizes), H, D), r(1, sum(ref_sizes), H, D))


# (name, batch, target tokens, reference token counts, biases, rows, row biases); 16-multiples and not
CASES = [
    ("strength 0.7 (Advanced), n % 16 == 0", 1, 48, [32], [0.7], None, None),
    ("strength 0.7 (Advanced), n % 16 != 0", 1, 44, [30], [0.7], None, None),
    ("simple strength 1.5, n % 16 != 0", 2, 45, [36], [math.log(1.5)], None, None),
    ("three references, weights 1 / 0.4 / 1.6, n % 16 != 0", 2, 45, [20, 18, 14],
     [0.0, math.log(0.4), math.log(1.6)], None, None),
    ("three references + strength -0.5, n % 16 == 0", 1, 46, [20, 18, 12],
     [-0.5, -0.5 + math.log(0.4), -0.5 + math.log(1.6)], None, None),
    ("cond reads, uncond does not (Mod Guidance), n % 16 == 0", 2, 48, [32], [0.0], [0], None),
    ("cond reads, uncond does not, two refs weighted, n % 16 != 0", 4, 45, [20, 13], [0.3, math.log(0.4)], [0, 1], None),
    ("Fidelity 0.5: every row, uncond at log(1 - f), n % 16 != 0", 2, 45, [36], [0.0], [0, 1], [0.0, math.log(0.5)]),
    ("Fidelity 0.25 with rows subset, n % 16 != 0", 3, 45, [20, 13], [0.0, math.log(1.6)], [0, 2], [0.0, math.log(0.75)]),
]


class MaskedAttentionTests(unittest.TestCase):
    def test_same_result_as_before_and_as_exact_softmax(self):
        for name, B, S, sizes, biases, rows, row_biases in CASES:
            q, k, v, ref = inputs(B, S, sizes)
            args = (q, k, v, ref, sizes, biases, rows, row_biases)
            with self.subTest(name):
                new = snapref._masked_attention(*args)
                torch.testing.assert_close(new, reference_masked_attention(*args), rtol=1e-5, atol=1e-6)
                torch.testing.assert_close(new.double(), exact(*args), rtol=1e-5, atol=1e-6)

    def test_half_precision_unchanged(self):
        for dtype in (torch.bfloat16, torch.float16):
            for name, B, S, sizes, biases, rows, row_biases in CASES:
                q, k, v, ref = inputs(B, S, sizes, dtype=dtype)
                args = (q, k, v, ref, sizes, biases, rows, row_biases)
                with self.subTest(name, dtype=dtype):
                    new = snapref._masked_attention(*args)
                    self.assertEqual(new.dtype, dtype)
                    self.assertTrue(torch.isfinite(new).all())
                    torch.testing.assert_close(new.float(), reference_masked_attention(*args).float(), rtol=2e-2, atol=2e-2)
                    torch.testing.assert_close(new.double(), exact(*args), rtol=3e-2, atol=3e-2)

    def test_no_infinite_mask_entries_off_cuda(self):
        # stands in for MPS: an SDPA that breaks on -inf in the mask, the way the black images showed it
        real = F.scaled_dot_product_attention
        masks = []

        def strict_sdpa(q, k, v, attn_mask=None, **kw):
            masks.append(attn_mask)
            if attn_mask is not None and not torch.isfinite(attn_mask).all():
                return torch.full_like(real(q, k, v, attn_mask=attn_mask, **kw), float("nan"))
            return real(q, k, v, attn_mask=attn_mask, **kw)

        with patch.object(snapref.F, "scaled_dot_product_attention", strict_sdpa):
            for name, B, S, sizes, biases, rows, row_biases in CASES:
                q, k, v, ref = inputs(B, S, sizes)
                with self.subTest(name):
                    for out in (snapref._masked_attention(q, k, v, ref, sizes, biases, rows, row_biases),
                                snapref._weighted_attention(q, k, v, ref, sizes, biases) if rows is None else
                                snapref._rows_attention(q, k, v, ref, sizes, biases, rows, row_biases)):
                        self.assertTrue(torch.isfinite(out).all())
                        torch.testing.assert_close(out.double(), exact(q, k, v, ref, sizes, biases, rows, row_biases),
                                                   rtol=1e-5, atol=1e-6)
        self.assertTrue(masks)
        for mask in masks:
            if mask is not None:
                self.assertTrue(torch.isfinite(mask).all())

    def test_keys_are_not_padded_off_cuda(self):
        q, k, v, ref = inputs(1, 45, [30])
        seen = []
        real = F.scaled_dot_product_attention
        with patch.object(snapref.F, "scaled_dot_product_attention",
                          lambda q_, k_, v_, **kw: seen.append(k_.shape[2]) or real(q_, k_, v_, **kw)):
            snapref._masked_attention(q, k, v, ref, [30], [0.5])
        self.assertEqual(seen, [75])   # 30 reference + 45 target keys, no padding to 80


if __name__ == "__main__":
    unittest.main()
