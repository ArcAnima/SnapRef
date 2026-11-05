"""CPU regression tests for real adapter hooks; no ComfyUI server or checkpoints required."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from test_model_reload import AnimaBase, TinyAttention, load_snapref

snapref = load_snapref()
# Explicit expected destinations, independent of the production map construction.
EXPECTED = (0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16, 18, 19,
            20, 22, 23, 25, 26, 28, 29, 31, 32, 34, 35, 37, 38, 39)


class Attention(TinyAttention):
    def compute_qkv(self, x, context, **kwargs):
        return x, x, x + 100

    def compute_attention(self, q, k, v, **kwargs):
        # Expose the exact keys and values passed by the production attention wrapper.
        return k, v


class TinyAnima(AnimaBase):
    def __init__(self, count):
        super().__init__()
        self.blocks = torch.nn.ModuleList()
        for _ in range(count):
            block = torch.nn.Module()
            block.self_attn = Attention()
            self.blocks.append(block)


def adapter(count=28):
    tensors = {}
    for i in range(count):
        for stream in ("ref", "tgt"):
            root = f"anima_ref_kv.{stream}.blocks.{i}.self_attn.q_proj."
            tensors.update({root + "lora_down": torch.ones(1, 2),
                            root + "lora_up": torch.full((2, 1), float(i + 1)),
                            root + "alpha": torch.tensor(0.5)})
    with tempfile.TemporaryDirectory() as directory:
        path = str(Path(directory) / "adapter.safetensors")
        save_file(tensors, path, metadata={"snapref": '{"format":"anima_ref_kv/1"}'})
        return snapref._Adapter(path)


def active(obj, enabled=True, stream="tgt"):
    snapref._STATE.update(adapter=obj if enabled else None, stream=stream if enabled else None)


def result(model, i):
    return model.blocks[i].self_attn.q_proj(torch.ones(1, 2))


class BlockLayoutTests(unittest.TestCase):
    def test_native_layouts(self):
        for count in (28, 40):
            self.assertEqual(snapref._adapter_block_map(range(count), TinyAnima(count)), tuple(range(count)))

    def test_expansion_map(self):
        self.assertEqual(snapref._adapter_block_map(range(28), TinyAnima(40)), EXPECTED)

    def test_unknown_and_incomplete_layouts(self):
        for indices, count in ((range(28), 36), (range(40), 28), (range(27), 40), ([], 40)):
            with self.assertRaises(ValueError):
                snapref._adapter_block_map(indices, TinyAnima(count))

    def test_other_model_type_not_guessed(self):
        with self.assertRaises(ValueError):
            snapref._adapter_block_map(range(28), SimpleNamespace(blocks=[None] * 40))


class HookTests(unittest.TestCase):
    def tearDown(self):
        snapref._STATE.update(adapter=None, stream=None)

    def test_native_hooks_preserve_layer_values(self):
        model, ad = TinyAnima(28), adapter()
        ad.install(model)
        active(ad)
        for i in range(28):
            torch.testing.assert_close(result(model, i), torch.full((1, 2), float(i + 1)))

    def test_expanded_hooks_land_on_original_layers(self):
        model, ad = TinyAnima(40), adapter()
        ad.install(model)
        active(ad)
        for src, dst in enumerate(EXPECTED):
            torch.testing.assert_close(result(model, dst), torch.full((1, 2), float(src + 1)))
        for i in set(range(40)) - set(EXPECTED):
            torch.testing.assert_close(result(model, i), torch.zeros(1, 2))

    def test_repeated_install_does_not_double_delta(self):
        model, ad = TinyAnima(40), adapter()
        ad.install(model)
        ad.install(model)
        active(ad)
        torch.testing.assert_close(result(model, 39), torch.full((1, 2), 28.0))

    def test_one_adapter_can_serve_both_models_without_mutating_keys(self):
        ad = adapter()
        before = tuple(ad.lora)
        base, expanded = TinyAnima(28), TinyAnima(40)
        ad.install(expanded)
        ad.install(base)
        active(ad)
        torch.testing.assert_close(result(base, 2), torch.full((1, 2), 3.0))
        torch.testing.assert_close(result(expanded, 3), torch.full((1, 2), 3.0))
        self.assertEqual(before, tuple(ad.lora))

    def test_inactive_adapter_does_not_change_model(self):
        model, ad = TinyAnima(40), adapter()
        ad.install(model)
        active(ad, False)
        for i in range(40):
            torch.testing.assert_close(result(model, i), torch.zeros(1, 2))

    def test_native_40_block_weights_are_not_remapped(self):
        model, ad = TinyAnima(40), adapter(40)
        ad.install(model)
        active(ad)
        torch.testing.assert_close(result(model, 2), torch.full((1, 2), 3.0))
        torch.testing.assert_close(result(model, 39), torch.full((1, 2), 40.0))

    def test_incompatible_layout_does_not_install_hooks(self):
        model, ad = TinyAnima(36), adapter()
        with self.assertRaises(ValueError):
            ad.install(model)
        self.assertFalse(any(b.self_attn.q_proj._forward_hooks for b in model.blocks))

    def test_missing_target_does_not_leave_partial_hooks(self):
        model, ad = TinyAnima(40), adapter()
        ad.lora["blocks.27.missing"] = next(iter(ad.lora.values()))
        with self.assertRaises(AttributeError):
            ad.install(model)
        self.assertNotIn(model, ad.hooked)
        self.assertFalse(any(b.self_attn.q_proj._forward_hooks for b in model.blocks))
        del ad.lora["blocks.27.missing"]
        ad.install(model)
        active(ad)
        torch.testing.assert_close(result(model, 39), torch.full((1, 2), 28.0))


class ReferenceCacheTests(unittest.TestCase):
    def tearDown(self):
        snapref._STATE.update(adapter=None, stream=None)

    def test_reference_cache_uses_all_actual_model_layers(self):
        model, ad = TinyAnima(40), adapter()
        ad.install(model)
        record = [None] * 40
        snapref._set_attn_state(model, {"record": record})
        for i, block in enumerate(model.blocks):
            block.self_attn(torch.full((1, 1, 1, 2), float(i)))
        for i in range(40):
            torch.testing.assert_close(record[i][0], torch.full((1, 1, 1, 2), float(i)))
            torch.testing.assert_close(record[i][1], torch.full((1, 1, 1, 2), float(i + 100)))
        snapref._set_attn_state(model, {"cache": record, "sizes": [1], "biases": [0.0]})
        for i, block in enumerate(model.blocks):
            keys, values = block.self_attn(torch.full((1, 1, 1, 2), -1.0))
            torch.testing.assert_close(keys, torch.tensor([[[[float(i), float(i)]], [[-1.0, -1.0]]]]))
            torch.testing.assert_close(values, torch.tensor([[[[float(i + 100), float(i + 100)]], [[99.0, 99.0]]]]))
        snapref._set_attn_state(model, None)

    def test_both_reference_and_target_streams_use_the_remapped_layer(self):
        model, ad = TinyAnima(40), adapter()
        ad.install(model)
        for stream in ("ref", "tgt"):
            active(ad, stream=stream)
            torch.testing.assert_close(result(model, 39), torch.full((1, 2), 28.0))


class NodeValidationTests(unittest.TestCase):
    def test_both_nodes_reject_incompatible_layout_before_encoding(self):
        model = SimpleNamespace(get_model_object=lambda name: TinyAnima(36))
        with patch.dict(snapref._adapters, {"test": adapter()}), \
             patch.object(snapref.folder_paths, "get_full_path_or_raise", return_value="test", create=True), \
             patch.object(snapref._simple, "load_input_image", return_value=torch.zeros(1, 1, 1, 3)), \
             patch.object(snapref.comfy.utils, "common_upscale", create=True,
                          side_effect=AssertionError("encoding started before validation")):
            with self.subTest(node="simple"), self.assertRaises(ValueError):
                snapref.SnapRefSimple().apply(model, None, None, "test", "image.png", "Character & Style", 1.0, 1.0)
            with self.subTest(node="advanced"), self.assertRaises(ValueError):
                snapref.SnapRef().apply(model, None, None, torch.zeros(1, 1, 1, 3), "test", "cs", False, 1.0)


if __name__ == "__main__":
    unittest.main()
