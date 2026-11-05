"""CPU regressions for model reloads; run with python -m unittest discover -s tests."""
import gc
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch
import weakref

import torch
from safetensors.torch import save_file


class AnimaBase(torch.nn.Module):
    pass


def load_snapref():
    # Only ComfyUI imports are stubbed; execute the complete node and its real constructor.
    comfy = ModuleType("comfy")
    modules = {"comfy": comfy, "folder_paths": ModuleType("folder_paths")}
    for name in ("patcher_extension", "samplers", "utils", "model_management"):
        module = ModuleType(f"comfy.{name}")
        setattr(comfy, name, module)
        modules[module.__name__] = module
    for name in ("comfy.ldm", "comfy.ldm.anima", "comfy.ldm.anima.model"):
        modules[name] = ModuleType(name)
    modules["comfy.ldm.anima.model"].Anima = AnimaBase
    source = Path(__file__).resolve().parents[1] / "comfyui_nodes/snapref/__init__.py"
    spec = importlib.util.spec_from_file_location("snapref_reload_test", source)
    module = importlib.util.module_from_spec(spec)
    modules[spec.name] = module
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
    return module


snapref = load_snapref()


class TinyAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = torch.nn.Linear(2, 2, bias=False)
        torch.nn.init.zeros_(self.q_proj.weight)

    def forward(self, x, context=None, **kwargs):
        return self.q_proj(x)


class TinyModel(torch.nn.Module):
    def __init__(self, value=1.0):
        super().__init__()
        block = torch.nn.Module()
        block.self_attn = TinyAttention()
        self.blocks = torch.nn.ModuleList([block])
        self.value = value

    def preprocess_text_embeds(self, cross, ids, weights):
        return cross


class CacheReader:
    def __init__(self, model):
        self.class_obj = model

    def __call__(self, *args, **kwargs):
        return self.class_obj.blocks[0].self_attn._ref_kv_state["cache"][0][1]


class ModelReloadTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = str(Path(directory.name) / "adapter.safetensors")
        tensors = {}
        for stream, value in (("ref", 2.0), ("tgt", 3.0)):
            root = f"anima_ref_kv.{stream}.blocks.0.self_attn.q_proj."
            tensors.update({root + "lora_down": torch.ones(1, 2),
                            root + "lora_up": torch.full((2, 1), value),
                            root + "alpha": torch.tensor(0.5)})
        save_file(tensors, path, metadata={"snapref": '{"format":"anima_ref_kv/1"}'})
        self.adapter = snapref._Adapter(path)
        # Force ID reuse only in SnapRef, without depending on allocator timing.
        id_patch = patch.object(snapref, "id", lambda model: 42, create=True)
        id_patch.start()
        self.addCleanup(id_patch.stop)
        self.addCleanup(snapref._STATE.update, adapter=None, stream=None)

    def test_reloaded_model_gets_hooks_without_duplicate_deltas(self):
        for _ in range(2):
            model = TinyModel()
            self.adapter.install(model)
            self.adapter.install(model)
            for stream, expected in (("ref", 2.0), ("tgt", 3.0)):
                snapref._STATE.update(adapter=self.adapter, stream=stream)
                actual = model.blocks[0].self_attn.q_proj(torch.ones(1, 2))
                torch.testing.assert_close(actual, torch.full((1, 2), expected))
            snapref._STATE.update(adapter=None, stream=None)
            torch.testing.assert_close(model.blocks[0].self_attn.q_proj(torch.ones(1, 2)), torch.zeros(1, 2))
            ref = weakref.ref(model)
            del model
            gc.collect()
            self.assertIsNone(ref())

    def test_unloaded_models_leave_no_installation_history(self):
        model = TinyModel()
        self.adapter.install(model)
        ref = weakref.ref(model)
        del model
        gc.collect()
        self.assertIsNone(ref())
        self.assertEqual(len(self.adapter.hooked), 0)

    def test_reference_cache_is_reused_until_model_is_replaced(self):
        encodes = []

        def encode_reference(adapter, model, *args):
            encodes.append(model.value)
            value = torch.full((1, 1, 1, 2), model.value)
            return [(value, value)]

        x = torch.zeros(1, 1, 2, 2)
        context = torch.zeros(1, 1, 2)
        wrapper = snapref._make_wrapper(self.adapter, [x], (context, {}), "cs", False)
        with patch.object(snapref, "_encode_reference", encode_reference):
            for value in (1.0, 2.0):
                executor = CacheReader(TinyModel(value))
                for _ in range(2):
                    actual = wrapper(executor, x, torch.zeros(1), context)
                    torch.testing.assert_close(actual, torch.full((1, 1, 1, 2), value))
                ref = weakref.ref(executor.class_obj)
                del executor
                gc.collect()
                self.assertIsNone(ref())
        self.assertEqual(encodes, [1.0, 2.0])
        self.assertEqual(len(self.adapter.cache), 1)


if __name__ == "__main__":
    unittest.main()
