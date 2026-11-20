"""The repository root works as a ComfyUI custom node (git clone into custom_nodes), next to the old way
(copy comfyui_nodes/snapref to custom_nodes/snapref) without registering anything twice.
Modules are loaded the way ComfyUI's load_custom_node does: spec_from_file_location(<path>, <dir>/__init__.py),
registered in sys.modules under a name built from the path."""
import importlib.util
import os
from pathlib import Path
import shutil
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch

from test_model_reload import AnimaBase

ROOT = Path(__file__).resolve().parents[1]


def comfy_stubs():
    comfy = ModuleType("comfy")
    modules = {"comfy": comfy, "folder_paths": ModuleType("folder_paths")}
    for name in ("patcher_extension", "samplers", "utils", "model_management"):
        module = ModuleType(f"comfy.{name}")
        setattr(comfy, name, module)
        modules[module.__name__] = module
    for name in ("comfy.ldm", "comfy.ldm.anima", "comfy.ldm.anima.model"):
        modules[name] = ModuleType(name)
    modules["comfy.ldm.anima.model"].Anima = AnimaBase
    return modules


def load_custom_node(path: Path):
    """nodes.load_custom_node: module name from the path, module dir = path, WEB_DIRECTORY relative to it."""
    name = str(path).replace(".", "_x_")
    spec = importlib.util.spec_from_file_location(name, str(path / "__init__.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    web = getattr(module, "WEB_DIRECTORY", None)
    return module, (os.path.abspath(os.path.join(path, web)) if web is not None else None)


class PackageLayoutTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        # the old installation: comfyui_nodes/snapref copied to custom_nodes/snapref
        self.copy = self.tmp / "snapref"
        shutil.copytree(ROOT / "comfyui_nodes" / "snapref", self.copy, ignore=shutil.ignore_patterns("__pycache__"))
        modules = patch.dict(sys.modules, comfy_stubs())
        modules.start()
        self.addCleanup(modules.stop)

    def test_repository_root_is_a_custom_node(self):
        module, web = load_custom_node(ROOT)
        self.assertEqual(set(module.NODE_CLASS_MAPPINGS), {"SnapRefSimple", "SnapRef"})
        self.assertEqual(module.NODE_DISPLAY_NAME_MAPPINGS["SnapRefSimple"], "SnapRef")
        self.assertTrue(os.path.isfile(os.path.join(web, "snapref.js")))
        self.assertNotIn("AnimaRefDatagen", " ".join(module.NODE_CLASS_MAPPINGS))   # training nodes stay internal

    def test_copied_subfolder_still_works(self):
        module, web = load_custom_node(self.copy)
        self.assertEqual(set(module.NODE_CLASS_MAPPINGS), {"SnapRefSimple", "SnapRef"})
        self.assertTrue(os.path.isfile(os.path.join(web, "snapref.js")))

    def test_both_installed_register_once_in_either_order(self):
        for first, second in ((self.copy, ROOT), (ROOT, self.copy)):
            with self.subTest(first=first.name), patch.dict(sys.modules, {}), \
                    self.assertLogs(level="WARNING") as logs:
                a, web_a = load_custom_node(first)
                b, web_b = load_custom_node(second)
                self.assertEqual(set(a.NODE_CLASS_MAPPINGS), {"SnapRefSimple", "SnapRef"})
                self.assertIsNotNone(web_a)
                self.assertEqual(b.NODE_CLASS_MAPPINGS, {})
                self.assertEqual(b.NODE_DISPLAY_NAME_MAPPINGS, {})
                self.assertIsNone(web_b)
                self.assertIn("Keep only one SnapRef", "\n".join(logs.output))

    def test_loading_the_same_copy_again_is_not_a_duplicate(self):
        with patch.dict(sys.modules, {}):
            load_custom_node(ROOT)
            again, web = load_custom_node(ROOT)
        self.assertEqual(set(again.NODE_CLASS_MAPPINGS), {"SnapRefSimple", "SnapRef"})
        self.assertIsNotNone(web)

    def test_root_imports_outside_comfyui(self):
        with patch.dict(sys.modules, {k: None for k in comfy_stubs()}):   # None: "import comfy" fails
            module, web = load_custom_node(ROOT)
        self.assertEqual(module.NODE_CLASS_MAPPINGS, {})
        self.assertIsNone(web)


if __name__ == "__main__":
    unittest.main()
