"""ComfyUI before 0.14.0: Anima.preprocess_text_embeds(text_embeds, text_ids) has no t5xxl_weights.
SnapRef used to fail inside KSampler with "takes 3 positional arguments but 4 were given"; both nodes now stop
at the node with an "update ComfyUI" message, before any reference is encoded."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from test_block_layout import TinyAnima, adapter, snapref


class OldAnima(TinyAnima):
    def preprocess_text_embeds(self, text_embeds, text_ids):   # ComfyUI 0.11 - 0.13
        return text_embeds


class NewAnima(TinyAnima):
    def preprocess_text_embeds(self, text_embeds, text_ids, t5xxl_weights=None):   # ComfyUI >= 0.14
        return text_embeds


class ComfyVersionTests(unittest.TestCase):
    def test_old_signature_is_the_reported_error(self):
        with self.assertRaisesRegex(TypeError, "takes 3 positional arguments but 4 were given"):
            OldAnima(28).preprocess_text_embeds(torch.zeros(1), None, None)

    def test_old_signature_asks_for_an_update(self):
        with self.assertRaisesRegex(RuntimeError, r"ComfyUI 0\.14\.0 or newer.*Update ComfyUI"):
            snapref._require_comfyui(OldAnima(28))

    def test_supported_and_unknown_models_pass(self):
        snapref._require_comfyui(NewAnima(28))
        snapref._require_comfyui(TinyAnima(28))   # no preprocess_text_embeds: nothing to check

        class Wrapped(TinyAnima):
            def preprocess_text_embeds(self, *args, **kwargs):
                return args[0]
        snapref._require_comfyui(Wrapped(28))

    def test_both_nodes_stop_before_encoding(self):
        model = SimpleNamespace(get_model_object=lambda name: OldAnima(28))
        with patch.dict(snapref._adapters, {"test": adapter()}), \
             patch.object(snapref.folder_paths, "get_full_path_or_raise", return_value="test", create=True), \
             patch.object(snapref._simple, "load_input_image", return_value=torch.zeros(1, 1, 1, 3)), \
             patch.object(snapref.comfy.utils, "common_upscale", create=True,
                          side_effect=AssertionError("encoding started before the version check")):
            with self.subTest(node="simple"), self.assertRaisesRegex(RuntimeError, "Update ComfyUI"):
                snapref.SnapRefSimple().apply(model, None, None, "test", "image.png", "Character & Style", 1.0, 1.0)
            with self.subTest(node="advanced"), self.assertRaisesRegex(RuntimeError, "Update ComfyUI"):
                snapref.SnapRef().apply(model, None, None, torch.zeros(1, 1, 1, 3), "test", "cs", False, 1.0)


if __name__ == "__main__":
    unittest.main()
