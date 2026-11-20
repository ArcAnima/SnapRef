"""Fidelity off CUDA (Mac MPS, CPU, fp32): 0 < f < 1 used to give exactly the ref_in_uncond result, because
the one-call path with the per-row bias was limited to CUDA fp16/bf16 and every other device let the negative
branch read the reference fully. The masked fallback carries the row biases, so Fidelity now works everywhere.
A one-block model runs the real patched self-attention on CPU."""
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from test_model_reload import AnimaBase, load_snapref
from test_masked_attention import exact

snapref = load_snapref()
B_HALF, S, H, D, N_REF = 1, 24, 2, 8, 20


class Attention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_target = torch.nn.Linear(2, 2, bias=False)   # only gives the adapter a module to hook

    def compute_qkv(self, x, context, rope_emb=None, transformer_options=None):
        return x, x.flip(1) * 0.7, x * 0.5 + 1

    def compute_attention(self, q, k, v, transformer_options=None):
        out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
        return out.transpose(1, 2).reshape(q.shape[0], q.shape[1], -1)

    def output_proj(self, x):
        return x

    def output_dropout(self, x):
        return x

    def forward(self, x, context=None, rope_emb=None, transformer_options={}):
        return self.compute_attention(*self.compute_qkv(x, context))


class OneBlock(AnimaBase):
    def __init__(self):
        super().__init__()
        block = torch.nn.Module()
        block.self_attn = Attention()
        self.blocks = torch.nn.ModuleList([block])

    def preprocess_text_embeds(self, text_embeds, text_ids, t5xxl_weights=None):
        return text_embeds


def make_adapter():
    tensors = {}
    for stream in ("ref", "tgt"):
        root = f"anima_ref_kv.{stream}.blocks.0.self_attn.lora_target."
        tensors.update({root + "lora_down": torch.zeros(1, 2), root + "lora_up": torch.zeros(2, 1),
                        root + "alpha": torch.tensor(1.0)})
    with tempfile.TemporaryDirectory() as directory:
        path = str(Path(directory) / "adapter.safetensors")
        save_file(tensors, path, metadata={"snapref": '{"format":"anima_ref_kv/1"}'})
        return snapref._Adapter(path)


class Executor:
    def __init__(self, dm):
        self.class_obj = dm

    def __call__(self, x, timesteps, context, fps=None, padding_mask=None, **kwargs):
        return self.class_obj.blocks[0].self_attn(x, transformer_options=kwargs.get("transformer_options", {}))


class FidelityFallbackTests(unittest.TestCase):
    def setUp(self):
        g = torch.Generator().manual_seed(3)
        self.x = torch.randn(2 * B_HALF, S, H, D, generator=g)
        self.kv = (torch.randn(1, N_REF, H, D, generator=g), torch.randn(1, N_REF, H, D, generator=g))
        self.ad = make_adapter()
        self.dm = OneBlock()
        self.addCleanup(snapref._STATE.update, adapter=None, stream=None)

    def run_node(self, in_uncond=False, fidelity=1.0, strength=0.0, flags=(0, 1)):
        uncond_bias = None if fidelity >= 1.0 or in_uncond else math.log(max(1.0 - fidelity, 1e-6))
        wrapper = snapref._make_wrapper(self.ad, [torch.ones(1, 2)], (torch.ones(1, 1, 2), {}), "cs", in_uncond,
                                        strength, None, uncond_bias)
        self.ad.cache.clear()
        with patch.object(snapref, "_encode_reference", return_value=[self.kv]):
            return wrapper(Executor(self.dm), self.x, torch.zeros(2), torch.zeros(2, 1, 2),
                           transformer_options={"cond_or_uncond": list(flags)})

    def expected(self, row_bias, strength=0.0):
        q, k, v = self.dm.blocks[0].self_attn.compute_qkv(self.x, None)
        return exact(q, k, v, self.kv, [N_REF], [strength], [0, 1], [0.0, row_bias]).float()

    def test_fidelity_is_graded_on_cpu(self):
        full = self.run_node(in_uncond=True)
        none = self.run_node()
        for f in (0.25, 0.5, 0.75):
            with self.subTest(fidelity=f):
                out = self.run_node(fidelity=f)
                torch.testing.assert_close(out, self.expected(math.log(1 - f)), rtol=1e-5, atol=1e-5)
                torch.testing.assert_close(out[0], full[0])                 # the positive branch is unchanged
                self.assertGreater((out[1] - full[1]).abs().max().item(), 1e-3)   # not ref_in_uncond any more
                self.assertGreater((out[1] - none[1]).abs().max().item(), 1e-3)   # nor fidelity 1

    def test_end_points_unchanged(self):
        torch.testing.assert_close(self.run_node(in_uncond=True), self.expected(0.0), rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(self.run_node(fidelity=1.0)[0], self.expected(0.0)[0], rtol=1e-5, atol=1e-5)

    def test_fidelity_with_strength_and_swapped_order(self):
        out = self.run_node(fidelity=0.5, strength=0.8, flags=(1, 0))
        q, k, v = self.dm.blocks[0].self_attn.compute_qkv(self.x, None)
        want = exact(q, k, v, self.kv, [N_REF], [0.8], [0, 1], [math.log(0.5), 0.0]).float()
        torch.testing.assert_close(out, want, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
