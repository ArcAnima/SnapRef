"""Old ComfyUI builds fail deep inside KSampler:
  < 0.14.0  Anima.preprocess_text_embeds(text_embeds, text_ids): "takes 3 positional arguments but 4 were given"
  < 0.29.0  Attention.compute_qkv(x, context, rope_emb):        "unexpected keyword argument 'transformer_options'"
Both nodes now stop at the node with an "update ComfyUI" message, before any reference is encoded."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from test_block_layout import Attention, TinyAnima, adapter, snapref


class OldQkv(Attention):
    def compute_qkv(self, x, context=None, rope_emb=None):   # ComfyUI 0.14 - 0.28
        return x, x, x + 100


class NewAnima(TinyAnima):
    def preprocess_text_embeds(self, text_embeds, text_ids, t5xxl_weights=None):   # ComfyUI >= 0.14
        return text_embeds


class OldAnima(TinyAnima):
    def preprocess_text_embeds(self, text_embeds, text_ids):   # ComfyUI 0.11 - 0.13
        return text_embeds


class MiddleAnima(NewAnima):   # ComfyUI 0.14 - 0.28: new text embeds, old compute_qkv
    def __init__(self, count):
        super().__init__(count)
        for block in self.blocks:
            block.self_attn = OldQkv()


class ComfyVersionTests(unittest.TestCase):
    def test_the_reported_errors(self):
        with self.assertRaisesRegex(TypeError, "takes 3 positional arguments but 4 were given"):
            OldAnima(28).preprocess_text_embeds(torch.zeros(1), None, None)
        with self.assertRaisesRegex(TypeError, "unexpected keyword argument 'transformer_options'"):
            OldQkv().compute_qkv(torch.zeros(1), None, rope_emb=None, transformer_options={})

    def test_old_builds_ask_for_an_update(self):
        for model, missing in ((OldAnima(28), "preprocess_text_embeds"), (MiddleAnima(28), "compute_qkv")):
            with self.subTest(missing), self.assertRaisesRegex(RuntimeError, rf"ComfyUI 0\.29\.0 or newer.*{missing}.*Update ComfyUI"):
                snapref._require_comfyui(model)

    def test_supported_and_unknown_models_pass(self):
        snapref._require_comfyui(NewAnima(28))
        snapref._require_comfyui(TinyAnima(28))   # no preprocess_text_embeds: nothing to check
        snapref._require_comfyui(SimpleNamespace(blocks=[None]))

        class Wrapped(TinyAnima):
            def preprocess_text_embeds(self, *args, **kwargs):
                return args[0]
        snapref._require_comfyui(Wrapped(28))

    def test_both_nodes_stop_before_encoding(self):
        for old in (OldAnima, MiddleAnima):
            model = SimpleNamespace(get_model_object=lambda name, old=old: old(28))
            with patch.dict(snapref._adapters, {"test": adapter()}), \
                 patch.object(snapref.folder_paths, "get_full_path_or_raise", return_value="test", create=True), \
                 patch.object(snapref._simple, "load_input_image", return_value=torch.zeros(1, 1, 1, 3)), \
                 patch.object(snapref.comfy.utils, "common_upscale", create=True,
                              side_effect=AssertionError("encoding started before the version check")):
                with self.subTest(node="simple", model=old.__name__), self.assertRaisesRegex(RuntimeError, "Update ComfyUI"):
                    snapref.SnapRefSimple().apply(model, None, None, "test", "image.png", "Character & Style", 1.0, 1.0)
                with self.subTest(node="advanced", model=old.__name__), self.assertRaisesRegex(RuntimeError, "Update ComfyUI"):
                    snapref.SnapRef().apply(model, None, None, torch.zeros(1, 1, 1, 3), "test", "cs", False, 1.0)


if __name__ == "__main__":
    unittest.main()
