"""Chained SnapRef nodes: SnapRef -> SnapRef -> KSampler reads the references of both nodes, each encoded in its
own node's mode (e.g. char then style), through one wrapper. Two nested wrappers would read only the inner one's
reference. A single node keeps its wrapper, cache key and settings."""
import math
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from test_branches import BranchModel, write_adapter
from test_model_reload import load_snapref

snapref = load_snapref()
KINDS = SimpleNamespace(DIFFUSION_MODEL="diffusion_model", OUTER_SAMPLE="outer_sample")
KEY = snapref.WRAPPER_KEY
IDLE = {"adapter": None, "stream": None, "branches": (), "reads_ref": False, "ref_rows": None}

CHAR = torch.full((1, 3), 1.0)    # three reference tokens of value 1
STYLE = torch.full((1, 2), 2.0)   # two of value 2
OTHER = torch.full((1, 1), 3.0)


def cond(value):
    return (torch.full((1, 1, 2), float(value)), {})


class Patcher:
    """What the nodes use of ComfyUI's ModelPatcher: clone() copies the wrapper lists (model_patcher.py),
    add_wrapper_with_key appends to the list under its key."""
    def __init__(self, wrappers=None, post_cfg=()):
        self.wrappers = wrappers or {}
        self.post_cfg = list(post_cfg)

    def clone(self):
        return Patcher({kind: {key: list(ws) for key, ws in keyed.items()} for kind, keyed in self.wrappers.items()},
                       self.post_cfg)

    def add_wrapper_with_key(self, kind, key, wrapper):
        self.wrappers.setdefault(kind, {}).setdefault(key, []).append(wrapper)

    def get_model_object(self, name):
        return object()

    def set_model_sampler_post_cfg_function(self, fn):
        self.post_cfg.append(fn)

    def model_wrappers(self):
        return [w for ws in self.wrappers.get(KINDS.DIFFUSION_MODEL, {}).values() for w in ws]


class Chain:
    """comfy.patcher_extension.WrapperExecutor: the wrappers in order, then the diffusion model, which here only
    records what the first block's self-attention would read."""
    def __init__(self, dm, wrappers, seen, idx=0):
        self.class_obj, self.wrappers, self.seen, self.idx = dm, wrappers, seen, idx

    def __call__(self, *args, **kwargs):
        return Chain(self.class_obj, self.wrappers, self.seen, self.idx + 1).execute(*args, **kwargs)

    def execute(self, *args, **kwargs):
        if self.idx < len(self.wrappers):
            return self.wrappers[self.idx](self, *args, **kwargs)
        x = args[0]
        state = self.class_obj.blocks[0].self_attn._ref_kv_state
        seen = {"batch": x.shape[0], "branches": snapref._STATE["branches"], "reads_ref": snapref._STATE["reads_ref"]}
        if state is not None:
            seen.update(keys=state["cache"][0][0][0, :, 0, 0].tolist(), sizes=list(state["sizes"]),
                        biases=list(state["biases"]), rows=state.get("rows"), row_biases=state.get("row_biases"))
        self.seen.append(seen)
        return x


def encoder(log):
    def encode(adapter, dm, lat, ctx, target_hw, mode):
        log.append((mode, float(lat.flatten()[0]), float(ctx.flatten()[0])))
        value = torch.full((1, lat.shape[-1], 1, 2), float(lat.flatten()[0]))
        return [(value, value)]
    return encode


class ChainTestCase(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.dir = directory.name
        self.ad = snapref._Adapter(write_adapter(self.dir, name="snapref_v2s.safetensors"))
        kinds = patch.object(snapref.comfy.patcher_extension, "WrappersMP", KINDS, create=True)
        kinds.start()
        self.addCleanup(kinds.stop)
        self.addCleanup(snapref._STATE.update, IDLE)

    def node(self, patcher, wrapper):
        """What a node's apply does with its wrapper: clone, chain, add, SnapRef first, its own outer wrapper."""
        m = patcher.clone()
        wrapper = snapref._chain(m, wrapper)
        m.add_wrapper_with_key(KINDS.DIFFUSION_MODEL, KEY, wrapper)
        keyed = m.wrappers[KINDS.DIFFUSION_MODEL]
        m.wrappers[KINDS.DIFFUSION_MODEL] = {KEY: keyed.pop(KEY), **keyed}
        m.add_wrapper_with_key(KINDS.OUTER_SAMPLE, KEY, snapref._make_outer(self.ad))
        return m, wrapper

    def run_model(self, patcher, flags=(0, 1), log=None, dm=None, **options):
        dm, seen, B = dm or BranchModel(), [], len(flags)
        with patch.object(snapref, "_encode_reference", encoder([] if log is None else log)):
            out = Chain(dm, patcher.model_wrappers(), seen).execute(
                torch.ones(B, 2), torch.zeros(B), torch.zeros(B, 1, 2),
                transformer_options={"cond_or_uncond": list(flags), **options})
        self.assertEqual(out.shape[0], B)
        self.assertEqual(snapref._STATE, IDLE)
        self.assertIsNone(dm.blocks[0].self_attn._ref_kv_state)
        return seen

    def char_then_style(self, **style):
        first, w1 = self.node(Patcher(), snapref._make_wrapper(self.ad, [CHAR], cond(10), "char", False))
        second, merged = self.node(first, snapref._make_wrapper(self.ad, [STYLE], cond(20), "style", False, **style))
        return first, w1, second, merged


class ChainedReferenceTests(ChainTestCase):
    def test_single_node_keeps_its_wrapper_and_cache_key(self):
        wrapper = snapref._make_wrapper(self.ad, [CHAR], cond(10), "char", False, 0.3, [0.5], -0.7)
        m, kept = self.node(Patcher(), wrapper)
        self.assertIs(kept, wrapper)
        spec = wrapper._snapref
        self.assertEqual([g["mode"] for g in spec["groups"]], ["char"])
        self.assertEqual(spec["groups"][0]["biases"], [0.3 + math.log(0.5)])
        self.assertEqual((spec["in_uncond"], spec["uncond_bias"]), (False, -0.7))
        self.run_model(m, flags=(0,))
        self.assertEqual([key[0] for key in self.ad.cache], [snapref._ref_digest([CHAR], cond(10), "char")])

    def test_chained_nodes_read_both_references_each_in_its_own_mode(self):
        first, w1, second, merged = self.char_then_style(weights=[0.5])
        self.assertIsNot(merged, w1)
        self.assertEqual(second.wrappers[KINDS.DIFFUSION_MODEL][KEY], [merged])
        self.assertEqual(len(second.wrappers[KINDS.OUTER_SAMPLE][KEY]), 1)
        log = []
        dm = BranchModel()
        seen = self.run_model(second, log=log, dm=dm)
        self.assertEqual(log, [("char", 1.0, 10.0), ("style", 2.0, 20.0)])   # own mode and own reference text
        self.assertEqual(seen, [
            {"batch": 1, "branches": ("style", "style_tgt"), "reads_ref": True, "keys": [1.0, 1.0, 1.0, 2.0, 2.0],
             "sizes": [3, 2], "biases": [0.0, math.log(0.5)], "rows": None, "row_biases": None},
            {"batch": 1, "branches": (), "reads_ref": False}])   # the negative branch: no reference, no branch
        self.run_model(second, log=log, dm=dm)
        self.assertEqual(len(log), 2)   # the joined set is cached as one
        self.assertEqual(len(self.ad.cache), 1)

    def test_earlier_node_output_still_reads_only_its_own_reference(self):
        first, w1, second, merged = self.char_then_style()
        self.assertEqual(first.wrappers[KINDS.DIFFUSION_MODEL][KEY], [w1])
        self.assertEqual([g["mode"] for g in w1._snapref["groups"]], ["char"])
        seen = self.run_model(first, flags=(0,))
        self.assertEqual((seen[0]["keys"], seen[0]["sizes"], seen[0]["branches"]), ([1.0, 1.0, 1.0], [3], ()))

    def test_nested_wrappers_read_only_the_inner_reference(self):
        # why _chain exists: each wrapper sets the attention state of every block, so the inner one wins
        nested = Patcher()
        nested.add_wrapper_with_key(KINDS.DIFFUSION_MODEL, KEY, snapref._make_wrapper(self.ad, [CHAR], cond(10), "char", False))
        nested.add_wrapper_with_key(KINDS.DIFFUSION_MODEL, KEY, snapref._make_wrapper(self.ad, [STYLE], cond(20), "style", False))
        seen = self.run_model(nested, flags=(0,))
        self.assertEqual((seen[0]["keys"], seen[0]["sizes"]), ([2.0, 2.0], [2]))

    def test_three_nodes_keep_their_order(self):
        first, _, second, _ = self.char_then_style()
        third, merged = self.node(second, snapref._make_wrapper(self.ad, [OTHER, OTHER], cond(30), "cs", False, 0.2))
        self.assertEqual([g["mode"] for g in merged._snapref["groups"]], ["char", "style", "cs"])
        self.assertEqual(len(third.wrappers[KINDS.DIFFUSION_MODEL][KEY]), 1)
        log = []
        seen = self.run_model(third, flags=(0,), log=log)
        self.assertEqual([entry[0] for entry in log], ["char", "style", "cs", "cs"])
        self.assertEqual((seen[0]["keys"], seen[0]["sizes"], seen[0]["biases"]),
                         ([1.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0], [3, 2, 1, 1], [0.0, 0.0, 0.2, 0.2]))

    def test_modes_without_a_branch_run_no_branch(self):
        first, _ = self.node(Patcher(), snapref._make_wrapper(self.ad, [CHAR], cond(10), "cs", False))
        second, _ = self.node(first, snapref._make_wrapper(self.ad, [OTHER], cond(10), "char", False))
        seen = self.run_model(second, flags=(0,))
        self.assertEqual((seen[0]["branches"], seen[0]["sizes"]), ((), [3, 1]))

    def test_other_wrappers_stay_and_snapref_runs_first(self):
        def other(executor, *args, **kwargs):
            return executor(*args, **kwargs)
        base = Patcher()
        base.add_wrapper_with_key(KINDS.DIFFUSION_MODEL, "other", other)
        first, _ = self.node(base, snapref._make_wrapper(self.ad, [CHAR], cond(10), "char", False))
        second, merged = self.node(first, snapref._make_wrapper(self.ad, [STYLE], cond(20), "style", False))
        self.assertEqual(second.model_wrappers(), [merged, other])
        self.assertEqual(self.run_model(second, flags=(0,))[0]["sizes"], [3, 2])


class ChainedSettingsTests(ChainTestCase):
    def merged(self, old, new):
        first, _ = self.node(Patcher(), snapref._make_wrapper(self.ad, [CHAR], cond(10), "char", **old))
        return self.node(first, snapref._make_wrapper(self.ad, [STYLE], cond(20), "style", **new))

    def test_fidelity_is_the_last_nodes(self):
        with self.assertLogs(level="INFO") as logs:
            second, w = self.merged({"in_uncond": False, "uncond_bias": -0.7}, {"in_uncond": False, "uncond_bias": math.log(0.5)})
        self.assertEqual((w._snapref["in_uncond"], w._snapref["uncond_bias"]), (False, math.log(0.5)))
        self.assertTrue(any("an earlier node's setting was overridden" in line for line in logs.output))
        # Fidelity < 1: one call, every row reads both references, the negative row weaker
        self.assertEqual(self.run_model(second), [
            {"batch": 2, "branches": ("style", "style_tgt"), "reads_ref": True, "keys": [1.0, 1.0, 1.0, 2.0, 2.0],
             "sizes": [3, 2], "biases": [0.0, 0.0], "rows": [0, 1], "row_biases": [0.0, math.log(0.5)]}])
        _, w = self.merged({"in_uncond": False, "uncond_bias": -0.7}, {"in_uncond": False})   # last node: Fidelity 1
        self.assertEqual((w._snapref["in_uncond"], w._snapref["uncond_bias"]), (False, None))

    def test_any_node_puts_the_references_in_the_negative_branch(self):
        # ref_in_uncond or ref_guidance_on (its post-CFG step stays on the model): every reference, Fidelity off
        for old, new in (({"in_uncond": True}, {"in_uncond": False, "uncond_bias": -0.7}),
                         ({"in_uncond": False, "uncond_bias": -0.7}, {"in_uncond": True})):
            with self.subTest(old=old):
                second, w = self.merged(old, new)
                self.assertEqual((w._snapref["in_uncond"], w._snapref["uncond_bias"]), (True, None))
                seen = self.run_model(second)
                self.assertEqual([(s["batch"], s["sizes"], s["rows"]) for s in seen], [(2, [3, 2], None)])
                # the reference-free call of ref_guidance reads none of them
                self.assertEqual(self.run_model(second, flags=(1,), **{snapref.NO_REF_KEY: True}),
                                 [{"batch": 1, "branches": (), "reads_ref": False}])

    def test_same_settings_log_no_override(self):
        with self.assertLogs(level="INFO") as logs:
            self.merged({"in_uncond": False, "uncond_bias": -0.7}, {"in_uncond": False, "uncond_bias": -0.7})
        self.assertFalse(any("overridden" in line for line in logs.output))
        self.assertTrue(any("modes char + style" in line for line in logs.output))

    def test_different_adapters_raise(self):
        other = snapref._Adapter(write_adapter(self.dir, name="snapref_v2a.safetensors", branches=False))
        first, _ = self.node(Patcher(), snapref._make_wrapper(other, [CHAR], cond(10), "char", False))
        with self.assertRaises(ValueError) as caught:
            self.node(first, snapref._make_wrapper(self.ad, [STYLE], cond(20), "style", False))
        message = str(caught.exception)
        self.assertIn("must use the same adapter file", message)
        self.assertIn("snapref_v2a.safetensors", message)
        self.assertIn("snapref_v2s.safetensors", message)
        self.assertEqual(len(first.wrappers[KINDS.DIFFUSION_MODEL][KEY]), 1)   # the earlier node's model is untouched


class ChainedNodeTests(ChainTestCase):
    """The two node classes, one after another."""
    def setUp(self):
        super().setUp()
        self.path = "loras/snapref_v2s.safetensors"
        self.clip = SimpleNamespace(tokenize=lambda text: text,
                                    encode_from_tokens_scheduled=lambda text: [cond(len(text))])

    def test_simple_nodes_chain(self):
        core = SimpleNamespace(_adapters={self.path: self.ad}, _Adapter=snapref._Adapter, WRAPPER_KEY=KEY,
                               _encode_image=lambda model, vae, px, megapixels: (px, None),
                               _make_wrapper=snapref._make_wrapper, _make_outer=snapref._make_outer,
                               _chain=snapref._chain, _adapter_block_map=lambda *a: None)
        cls = snapref._simple.make_simple_node(core)
        pictures = {"char.png": CHAR, "face.png": CHAR, "style.png": STYLE}
        with patch.object(snapref._simple.folder_paths, "get_full_path_or_raise", return_value=self.path, create=True), \
             patch.object(snapref._simple, "load_input_image", side_effect=pictures.__getitem__):
            (first,) = cls().apply(Patcher(), self.clip, None, "snapref_v2s.safetensors", "char.png", "Character", 1.0,
                                   1.0, "face.png", 0.5)
            (second,) = cls().apply(first, self.clip, None, "snapref_v2s.safetensors", "style.png", "Style", 0.5, 0.75)
        (wrapper,) = second.wrappers[KINDS.DIFFUSION_MODEL][KEY]
        spec = wrapper._snapref
        self.assertEqual([(g["mode"], len(g["latents"]), g["biases"]) for g in spec["groups"]],
                         [("char", 2, [0.0, math.log(0.5)]), ("style", 1, [math.log(0.5)])])
        self.assertEqual((spec["in_uncond"], spec["uncond_bias"]), (False, math.log(0.25)))   # the Style node's Fidelity
        self.assertEqual(list(second.wrappers[KINDS.DIFFUSION_MODEL])[0], KEY)
        self.assertEqual(len(second.wrappers[KINDS.OUTER_SAMPLE][KEY]), 1)
        self.assertEqual([len(w._snapref["groups"]) for w in first.wrappers[KINDS.DIFFUSION_MODEL][KEY]], [1])
        log = []
        seen = self.run_model(second, flags=(0,), log=log)
        self.assertEqual([entry[0] for entry in log], ["char", "char", "style"])
        self.assertEqual(seen[0]["sizes"], [3, 3, 2])

    def test_advanced_nodes_chain(self):
        image = lambda lat: lat.view(1, 1, -1, 1)   # noqa: E731  IMAGE [B, H, W, C]
        with patch.object(snapref.folder_paths, "get_full_path_or_raise", return_value=self.path, create=True), \
             patch.dict(snapref._adapters, {self.path: self.ad}, clear=True), \
             patch.object(snapref, "_adapter_block_map", lambda *a: None), \
             patch.object(snapref, "_require_comfyui", lambda dm: None), \
             patch.object(snapref, "_encode_image", lambda model, vae, px, megapixels: (px.flatten()[None], (8, 8))):
            (first,) = snapref.SnapRef().apply(Patcher(), self.clip, None, image(CHAR), "snapref_v2s.safetensors", "char",
                                               False, 1.0, strength=0.3, ref_caption="1girl", fidelity=0.5)
            (second,) = snapref.SnapRef().apply(first, self.clip, None, image(STYLE), "snapref_v2s.safetensors", "style",
                                                False, 1.0, ref_guidance_on=True, image_2=image(STYLE), weight_2=0.5)
        (wrapper,) = second.wrappers[KINDS.DIFFUSION_MODEL][KEY]
        spec = wrapper._snapref
        self.assertEqual([(g["mode"], g["biases"], float(g["cond"][0].flatten()[0])) for g in spec["groups"]],
                         [("char", [0.3], 5.0), ("style", [0.0, math.log(0.5)], 0.0)])
        self.assertEqual((spec["in_uncond"], spec["uncond_bias"]), (True, None))
        self.assertEqual((len(first.post_cfg), len(second.post_cfg)), (0, 1))
        self.assertEqual(len(second.wrappers[KINDS.OUTER_SAMPLE][KEY]), 1)
        log = []
        seen = self.run_model(second, log=log)
        self.assertEqual(log, [("char", 1.0, 5.0), ("style", 2.0, 0.0), ("style", 2.0, 0.0)])
        self.assertEqual([(s["batch"], s["sizes"], s["biases"]) for s in seen], [(2, [3, 2, 2], [0.3, 0.0, math.log(0.5)])])


if __name__ == "__main__":
    unittest.main()
