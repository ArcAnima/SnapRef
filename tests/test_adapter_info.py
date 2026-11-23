"""GET /snapref/adapters: each SnapRef file's modes from its safetensors header only, for the simple node's Style
button. A file that cannot be read (still downloading, damaged) is reported, never raised."""
import asyncio
import importlib.util
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from test_model_reload import load_snapref

SOURCE = Path(__file__).resolve().parents[1] / "comfyui_nodes/snapref/adapter_info.py"


def load_info():
    spec = importlib.util.spec_from_file_location("snapref_adapter_info_test", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


info = load_info()

STYLE_META = {"format": "anima_ref_kv/1", "modes": ["cs", "char", "style"],
              "branches": {"style": {"stream": "ref", "modes": ["style"], "combine": "add"},
                           "style_tgt": {"stream": "tgt", "modes": ["style"], "combine": "add"}}}
V2A_META = {"format": "anima_ref_kv/1", "modes": ["cs", "char"]}


class AdapterInfoTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.dir = Path(directory.name)
        info._cache.clear()

    def write(self, name, meta, key="anima_ref_kv"):
        path = self.dir / name
        save_file({"anima_ref_kv.mode_embed": torch.zeros(2, 2)}, str(path),
                  metadata=None if meta is None else {key: json.dumps(meta)})
        return path

    def resolve(self, name):
        path = self.dir / name
        return str(path) if path.exists() else None

    def test_reads_modes_and_style_flag(self):
        self.write("snapref_v2s_pre.safetensors", STYLE_META)
        self.write("snapref_v2a.safetensors", V2A_META, key="snapref")
        self.write("snapref_modes_only.safetensors", {**V2A_META, "modes": ["cs", "char", "style"]})
        names = ["snapref_v2s_pre.safetensors", "snapref_v2a.safetensors", "snapref_modes_only.safetensors"]
        self.assertEqual(info.describe(names, self.resolve), {
            "snapref_v2s_pre.safetensors": {"modes": ["cs", "char", "style"], "style": True},
            "snapref_v2a.safetensors": {"modes": ["cs", "char"], "style": False},
            "snapref_modes_only.safetensors": {"modes": ["cs", "char", "style"], "style": False},
        })
        self.assertEqual(info.read_meta(str(self.dir / "snapref_v2s_pre.safetensors")), STYLE_META)

    def test_missing_metadata_defaults_to_cs(self):
        self.write("snapref_v1a.safetensors", {"format": "anima_ref_kv/1"})
        self.write("snapref_bare.safetensors", None)
        out = info.describe(["snapref_v1a.safetensors", "snapref_bare.safetensors"], self.resolve)
        self.assertEqual(out["snapref_v1a.safetensors"], {"modes": ["cs"], "style": False})
        self.assertEqual(out["snapref_bare.safetensors"], {"modes": ["cs"], "style": False})

    def test_unreadable_file_is_reported_not_raised(self):
        good = self.write("snapref_v2s_pre.safetensors", STYLE_META)
        (self.dir / "snapref_partial.safetensors").write_bytes(good.read_bytes()[:20])   # still downloading
        (self.dir / "snapref_empty.safetensors").write_bytes(b"")
        (self.dir / "snapref_huge.safetensors").write_bytes(struct.pack("<Q", 200 * 1024 * 1024) + b"{}")
        (self.dir / "snapref_junk.safetensors").write_bytes(struct.pack("<Q", 4) + b"\xff\xfe{x")
        names = ["snapref_partial.safetensors", "snapref_empty.safetensors", "snapref_huge.safetensors",
                 "snapref_junk.safetensors", "snapref_gone.safetensors", "snapref_v2s_pre.safetensors"]
        out = info.describe(names, self.resolve)
        self.assertEqual(list(out), names)
        for name in names[:-1]:
            with self.subTest(name=name):
                self.assertEqual(list(out[name]), ["error"])
                self.assertIsInstance(out[name]["error"], str)
                self.assertTrue(out[name]["error"])
        self.assertTrue(out["snapref_v2s_pre.safetensors"]["style"])
        json.dumps(out)   # the route can always serialize it

    def test_cache_invalidates_on_mtime(self):
        path = self.write("snapref_v2s_pre.safetensors", STYLE_META)
        reads = []
        real = info.read_meta
        with patch.object(info, "read_meta", side_effect=lambda p: reads.append(p) or real(p)):
            for _ in range(3):
                info.describe(["snapref_v2s_pre.safetensors"], self.resolve)
            self.assertEqual(len(reads), 1)
            self.write("snapref_v2s_pre.safetensors", V2A_META)
            stat = path.stat()
            os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 5_000_000_000))
            out = info.describe(["snapref_v2s_pre.safetensors"], self.resolve)
        self.assertEqual(len(reads), 2)
        self.assertEqual(out["snapref_v2s_pre.safetensors"], {"modes": ["cs", "char"], "style": False})

    def test_errors_are_not_cached(self):
        (self.dir / "snapref_v2s_pre.safetensors").write_bytes(b"")
        self.assertIn("error", info.describe(["snapref_v2s_pre.safetensors"], self.resolve)["snapref_v2s_pre.safetensors"])
        self.write("snapref_v2s_pre.safetensors", STYLE_META)
        self.assertTrue(info.describe(["snapref_v2s_pre.safetensors"], self.resolve)["snapref_v2s_pre.safetensors"]["style"])


class RouteTests(unittest.TestCase):
    def test_route_lists_the_simple_node_adapters(self):
        routes = {}

        class Routes:
            def get(self, path):
                def register(handler):
                    routes[path] = handler
                    return handler
                return register

        server = ModuleType("server")
        server.PromptServer = SimpleNamespace(instance=SimpleNamespace(routes=Routes()))
        with patch.dict(sys.modules, {"server": server}):
            snapref = load_snapref()
        self.assertIn("/snapref/adapters", routes)
        seen = []
        with patch.object(snapref._simple, "snapref_files", return_value=["snapref_v2a.safetensors"]), \
             patch.object(snapref.folder_paths, "get_full_path", side_effect=lambda kind, n: seen.append((kind, n)),
                          create=True):
            resp = asyncio.run(routes["/snapref/adapters"](None))
        self.assertEqual(seen, [("loras", "snapref_v2a.safetensors")])
        self.assertEqual(resp.status, 200)
        body = json.loads(resp.text)
        self.assertEqual(list(body["adapters"]), ["snapref_v2a.safetensors"])
        self.assertIn("error", body["adapters"]["snapref_v2a.safetensors"])   # not found: reported, not raised

    def test_two_installed_copies_register_the_route_once(self):
        import shutil
        from test_package_layout import ROOT, comfy_stubs, load_custom_node
        registered = []

        class Routes:
            def get(self, path):
                return lambda handler: registered.append(path) or handler

        server = ModuleType("server")
        server.PromptServer = SimpleNamespace(instance=SimpleNamespace(routes=Routes()))
        with tempfile.TemporaryDirectory() as tmp:
            copy = Path(tmp) / "snapref"
            shutil.copytree(ROOT / "comfyui_nodes" / "snapref", copy, ignore=shutil.ignore_patterns("__pycache__"))
            with patch.dict(sys.modules, {**comfy_stubs(), "server": server}), self.assertLogs(level="WARNING"):
                load_custom_node(ROOT)
                load_custom_node(copy)
        self.assertEqual(registered, ["/snapref/adapters"])

    def test_no_server_no_route(self):
        for server in (None, SimpleNamespace(PromptServer=SimpleNamespace(instance=None))):
            with self.subTest(server=server), patch.dict(sys.modules, {"server": server}):
                load_snapref()   # nothing to register with: loads quietly


if __name__ == "__main__":
    unittest.main()
