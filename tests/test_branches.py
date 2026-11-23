"""Mode-gated branch LoRAs (anima_ref_kv.branch.<branch>.<module>.down / up / alpha), as train/style_branch.py:
a reference-stream branch runs while the reference is encoded in one of its modes, a target-stream branch only in
target forwards that read that mode's reference. With no branch active the hooks compute exactly what they did
before, so cs / char outputs stay bit-identical."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from test_model_reload import TinyModel, load_snapref

snapref = load_snapref()

REF_BRANCH = "blocks.0.self_attn.q_proj"   # has ref and tgt stream LoRAs
TGT_BRANCH = "blocks.0.mlp"                # has no stream LoRA of its own


def branch_spec(stream, rank=2, alpha=1.0, targets=("self_attn.q_proj",)):
    return {"stream": stream, "modes": ["style"], "rank": rank, "alpha": alpha, "targets": list(targets),
            "combine": "add"}


def write_adapter(directory, name="adapter.safetensors", branches=True, meta_branches=None, modes=None, fp8=False,
                  ref_up=5.0, tgt_up=7.0):
    """Stream LoRAs on q_proj (ref up = 2, tgt up = 3, scale 0.5), a ref branch "style" on q_proj (up = ref_up,
    scale 0.5) and a tgt branch "style_tgt" on mlp (up = tgt_up, scale 0.5)."""
    tensors = {}
    for stream, value in (("ref", 2.0), ("tgt", 3.0)):
        root = f"anima_ref_kv.{stream}.{REF_BRANCH}."
        tensors.update({root + "lora_down": torch.ones(1, 2), root + "lora_up": torch.full((2, 1), value),
                        root + "alpha": torch.tensor(0.5)})
    meta = {"format": "anima_ref_kv/1", "modes": modes or ["cs", "char"]}
    if branches:
        for branch, module, up in (("style", REF_BRANCH, ref_up), ("style_tgt", TGT_BRANCH, tgt_up)):
            root = f"anima_ref_kv.branch.{branch}.{module}."
            down = torch.ones(2, 2)
            if fp8:
                tensors[root + "down"] = down * 0.25
                tensors[root + "down_scale"] = torch.full((2, 1), 4.0)
            else:
                tensors[root + "down"] = down
            tensors[root + "up"] = torch.full((2, 2), up)
            tensors[root + "alpha"] = torch.tensor(1.0)
        meta["modes"] = modes or ["cs", "char", "style"]
        tensors["anima_ref_kv.mode_embed"] = torch.zeros(len(meta["modes"]), 2)
        if meta_branches is None:
            meta_branches = {"style": branch_spec("ref"), "style_tgt": branch_spec("tgt", targets=("mlp",))}
    if meta_branches:
        meta["branches"] = meta_branches
    path = str(Path(directory) / name)
    tensors = {k: v.to(torch.bfloat16) if k.endswith(("down", "up")) else v for k, v in tensors.items()}
    save_file(tensors, path, metadata={"anima_ref_kv": json.dumps(meta)})
    return path


class BranchModel(TinyModel):
    def __init__(self):
        super().__init__()
        self.blocks[0].mlp = torch.nn.Linear(2, 2, bias=False)
        torch.nn.init.zeros_(self.blocks[0].mlp.weight)


def outputs(model, x=None):
    x = torch.ones(1, 2) if x is None else x
    return model.blocks[0].self_attn.q_proj(x), model.blocks[0].mlp(x)


class BranchTestCase(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.dir = directory.name
        self.addCleanup(snapref._STATE.update, adapter=None, stream=None, branches=(), reads_ref=False, ref_rows=None)

    def installed(self, **kwargs):
        ad = snapref._Adapter(write_adapter(self.dir, name=f"a{len(list(Path(self.dir).iterdir()))}.safetensors", **kwargs))
        model = BranchModel()
        ad.install(model)
        return ad, model

    def run_state(self, ad, model, **state):
        snapref._STATE.update(adapter=ad, **{"branches": (), "reads_ref": False, "ref_rows": None, **state})
        try:
            return outputs(model)
        finally:
            snapref._STATE.update(adapter=None, stream=None, branches=(), reads_ref=False, ref_rows=None)


class BranchLoadTests(BranchTestCase):
    def test_branch_keys_parse_with_scale(self):
        ad = snapref._Adapter(write_adapter(self.dir))
        self.assertEqual(list(ad.branches), ["style", "style_tgt"])
        self.assertEqual(ad.branches["style"]["stream"], "ref")
        self.assertEqual(ad.branches["style_tgt"]["stream"], "tgt")
        self.assertEqual(ad.branches["style"]["modes"], ["style"])
        self.assertEqual(ad.branches["style"]["modules"][REF_BRANCH][2], 1.0 / 2)
        self.assertEqual(list(ad.branches["style_tgt"]["modules"]), [TGT_BRANCH])
        self.assertEqual(ad.branches_for("style"), ("style", "style_tgt"))
        self.assertEqual(ad.branches_for("cs"), ())
        self.assertEqual(ad.branches_for("char"), ())
        self.assertEqual(snapref._Adapter(write_adapter(self.dir, name="plain.safetensors", branches=False)).branches, {})

    def test_branches_follow_metadata_order(self):
        specs = {"style_tgt": branch_spec("tgt", targets=("mlp",)), "style": branch_spec("ref")}
        ad = snapref._Adapter(write_adapter(self.dir, meta_branches=specs))
        self.assertEqual(ad.branches_for("style"), ("style_tgt", "style"))

    def test_branch_without_metadata_raises(self):
        path = write_adapter(self.dir, meta_branches={})
        with self.assertRaisesRegex(ValueError, "branches.style"):
            snapref._Adapter(path)

    def test_unsupported_branch_spec_raises(self):
        for bad in ({"stream": "both"}, {"combine": "replace"}):
            spec = {**branch_spec("ref"), **bad}
            path = write_adapter(self.dir, name=f"bad_{list(bad)[0]}.safetensors",
                                 meta_branches={"style": spec, "style_tgt": branch_spec("tgt", targets=("mlp",))})
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                snapref._Adapter(path)

    def test_branch_weights_dequantize_fp8_scale(self):
        ad = snapref._Adapter(write_adapter(self.dir, fp8=True))
        down = ad.branches["style"]["modules"][REF_BRANCH][0]
        self.assertEqual(down.dtype, torch.bfloat16)
        torch.testing.assert_close(down.float(), torch.ones(2, 2))   # 0.25 * scale 4

    def test_check_mode_message(self):
        plain = snapref._Adapter(write_adapter(self.dir, name="v2a.safetensors", branches=False))
        with self.assertRaises(ValueError) as caught:
            plain.check_mode("style", "snapref_v2a.safetensors")
        message = str(caught.exception)
        self.assertIn("SnapRef: snapref_v2a.safetensors has no 'style' mode (its modes: cs, char).", message)
        self.assertIn("snapref_v2s_pre", message)
        self.assertIn("use mode cs or char with this adapter", message)
        for mode in ("cs", "char"):
            plain.check_mode(mode, "snapref_v2a.safetensors")   # never an error
        snapref._Adapter(write_adapter(self.dir)).check_mode("style", "snapref_v2s_pre.safetensors")

    def test_check_mode_needs_branch_weights_for_a_listed_style_mode(self):
        path = write_adapter(self.dir, branches=False, modes=["cs", "char", "style"])
        with self.assertRaisesRegex(ValueError, "style"):
            snapref._Adapter(path).check_mode("style", "odd.safetensors")


class RefBranchTests(BranchTestCase):
    def test_ref_branch_adds_only_while_active(self):
        ad, model = self.installed()
        plain, plain_model = self.installed(branches=False)
        lora_only = self.run_state(plain, plain_model, stream="ref")
        # stream LoRA: 0.5 * 2 * (1 + 1) = 2; ref branch: 0.5 * 5 * (2 + 2) = 10 per output
        q, mlp = self.run_state(ad, model, stream="ref", branches=("style",))
        torch.testing.assert_close(q, torch.full((1, 2), 2.0 + 10.0))
        self.assertTrue(torch.equal(mlp, torch.zeros(1, 2)))
        q, mlp = self.run_state(ad, model, stream="ref")
        self.assertTrue(torch.equal(q, lora_only[0]))
        self.assertTrue(torch.equal(mlp, lora_only[1]))
        # the ref branch never runs in the target stream
        tgt_plain = self.run_state(plain, plain_model, stream="tgt", reads_ref=True)
        q, mlp = self.run_state(ad, model, stream="tgt", branches=("style",), reads_ref=True)
        self.assertTrue(torch.equal(q, tgt_plain[0]))
        self.assertTrue(torch.equal(mlp, tgt_plain[1]))

    def test_tgt_branch_needs_reads_ref(self):
        ad, model = self.installed()
        plain, plain_model = self.installed(branches=False)
        tgt_plain = self.run_state(plain, plain_model, stream="tgt")
        q, mlp = self.run_state(ad, model, stream="tgt", branches=("style_tgt",), reads_ref=False)
        self.assertTrue(torch.equal(q, tgt_plain[0]))
        self.assertTrue(torch.equal(mlp, tgt_plain[1]))
        q, mlp = self.run_state(ad, model, stream="tgt", branches=("style_tgt",), reads_ref=True)
        self.assertTrue(torch.equal(q, tgt_plain[0]))
        torch.testing.assert_close(mlp, torch.full((1, 2), 0.5 * 7.0 * 4))
        # nor in the reference stream
        q, mlp = self.run_state(ad, model, stream="ref", branches=("style_tgt",), reads_ref=True)
        self.assertTrue(torch.equal(mlp, torch.zeros(1, 2)))

    def test_hook_on_branch_only_module(self):
        ad, model = self.installed()
        self.assertNotIn("blocks.0.mlp", ad.lora)
        self.assertEqual(len(model.blocks[0].mlp._forward_hooks), 1)
        self.assertEqual(len(model.blocks[0].self_attn.q_proj._forward_hooks), 1)
        ad.install(model)   # installing again adds nothing
        self.assertEqual(len(model.blocks[0].mlp._forward_hooks), 1)
        q, mlp = self.run_state(ad, model, stream="tgt", branches=("style", "style_tgt"), reads_ref=True)
        torch.testing.assert_close(mlp, torch.full((1, 2), 14.0))
        snapref._STATE.update(adapter=None, stream=None)
        self.assertTrue(torch.equal(outputs(model)[1], torch.zeros(1, 2)))   # inactive adapter: base output only

    def test_encode_reference_runs_mode_branches_then_resets(self):
        ad, model = self.installed()
        seen = []

        class Block(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.self_attn = torch.nn.Module()

            def forward(self, x, *args, **kwargs):
                seen.append((snapref._STATE["stream"], snapref._STATE["branches"], snapref._STATE["reads_ref"]))
                return x

        dm = fake_encoder_model(Block())
        for mode, want in (("style", ("style", "style_tgt")), ("cs", ())):
            seen.clear()
            snapref._encode_reference(ad, dm, torch.zeros(1, 2, 1, 2, 2), None, (4, 4), mode)
            self.assertEqual(seen, [("ref", want, False)])
            self.assertEqual((snapref._STATE["adapter"], snapref._STATE["branches"], snapref._STATE["reads_ref"],
                              snapref._STATE["ref_rows"]), (None, (), False, None))


def fake_encoder_model(block):
    """Just enough of MiniTrainDIT for _encode_reference: embedding, RoPE tables, time embedding and the blocks."""
    dm = SimpleNamespace()
    dm.prepare_embedded_sequence = lambda ref, fps=None, padding_mask=None: (torch.zeros(1, 1, 1, 1, 2), None, None)
    dm.pos_embedder = SimpleNamespace(t_ntk_factor=1.0, h_ntk_factor=1.0, w_ntk_factor=1.0,
                                      dim_temporal_range=torch.zeros(1), dim_spatial_range=torch.zeros(1))
    dm.t_embedder = [lambda t: torch.zeros(1, 1, 2), lambda t: (t, None)]
    dm.t_embedding_norm = lambda t: t
    dm.blocks = [block]
    return dm


class ModuleExecutor:
    """Stands for the diffusion model: one target forward through the hooked q_proj (tgt stream LoRA) and mlp (target
    branch only); records the branch state of every call."""
    def __init__(self, dm, calls):
        self.class_obj, self.calls = dm, calls

    def __call__(self, x, timesteps, context, fps=None, padding_mask=None, **kwargs):
        self.calls.append((x.shape[0], snapref._STATE["branches"], snapref._STATE["reads_ref"], snapref._STATE["ref_rows"]))
        block = self.class_obj.blocks[0]
        return torch.cat([block.self_attn.q_proj(x), block.mlp(x)], dim=-1)


def encode(*args):
    return [(torch.ones(1, 1, 1, 2), torch.ones(1, 1, 1, 2))]


Q = 0.5 * 3.0 * 2      # tgt stream LoRA on q_proj for x = ones
DELTA = 0.5 * 7.0 * 4  # tgt branch on mlp for x = ones


class TargetBranchTests(BranchTestCase):
    def setUp(self):
        super().setUp()
        self.ad = snapref._Adapter(write_adapter(self.dir, name="v2t.safetensors"))
        self.plain = snapref._Adapter(write_adapter(self.dir, name="v2a.safetensors", branches=False))

    def call(self, ad, mode="style", flags=(0, 1), in_uncond=False, uncond_bias=None, per=1, mod_guidance=False,
             no_ref=False, calls=None):
        dm = BranchModel()
        if mod_guidance:
            dm._anima_mod_block_state = object()
        B = per * len(flags)
        options = {"cond_or_uncond": list(flags)}
        if no_ref:
            options[snapref.NO_REF_KEY] = True
        wrapper = snapref._make_wrapper(ad, [torch.ones(1, 2)], (torch.ones(1, 1, 2), {}), mode, in_uncond,
                                        uncond_bias=uncond_bias)
        with patch.object(snapref, "_encode_reference", side_effect=encode):
            out = wrapper(ModuleExecutor(dm, [] if calls is None else calls), torch.ones(B, 2), torch.zeros(B),
                          torch.zeros(B, 1, 2), transformer_options=options)
        self.assertEqual(snapref._STATE, {"adapter": None, "stream": None, "branches": (), "reads_ref": False,
                                          "ref_rows": None})
        return out

    def assert_rows(self, out, branch_rows):
        want = torch.tensor([[Q, Q, DELTA if r in branch_rows else 0.0, DELTA if r in branch_rows else 0.0]
                             for r in range(out.shape[0])])
        torch.testing.assert_close(out, want)

    def test_tgt_branch_on_cond_only_by_default(self):
        for flags in ((0, 1), (1, 0)):
            with self.subTest(flags=flags):
                calls = []
                out = self.call(self.ad, flags=flags, calls=calls)
                self.assert_rows(out, [flags.index(0)])
                plain = self.call(self.plain, mode="cs", flags=flags)
                uncond = flags.index(1)
                self.assertTrue(torch.equal(out[uncond], plain[uncond]))
                self.assertEqual(calls, [(1, ("style", "style_tgt"), True, None) if f == 0 else (1, (), False, None)
                                         for f in flags])

    def test_tgt_branch_on_both_with_ref_in_uncond(self):
        self.assert_rows(self.call(self.ad, in_uncond=True), [0, 1])

    def test_tgt_branch_whole_call_with_fidelity(self):
        calls = []
        self.assert_rows(self.call(self.ad, uncond_bias=-0.7, calls=calls), [0, 1])
        self.assertEqual(calls, [(2, ("style", "style_tgt"), True, None)])

    def test_tgt_branch_off_in_reference_free_call(self):
        for kwargs in ({}, {"in_uncond": True}, {"uncond_bias": -0.7}):
            with self.subTest(**kwargs):
                out = self.call(self.ad, no_ref=True, **kwargs)
                self.assert_rows(out, [])
                self.assertTrue(torch.equal(out, self.call(self.plain, mode="cs", no_ref=True, **kwargs)))

    def test_tgt_branch_rows_path_only_reference_rows(self):
        for flags, per, rows in (((0, 1), 1, [0]), ((1, 0), 2, [2, 3])):
            with self.subTest(flags=flags, per=per):
                calls = []
                out = self.call(self.ad, flags=flags, per=per, mod_guidance=True, calls=calls)
                self.assert_rows(out, rows)
                self.assertEqual(calls, [(2 * per, ("style", "style_tgt"), True, rows)])
                plain = self.call(self.plain, mode="cs", flags=flags, per=per, mod_guidance=True)
                others = [r for r in range(2 * per) if r not in rows]
                self.assertTrue(torch.equal(out[others], plain[others]))

    def test_cs_after_style_runs_no_branch(self):
        for kwargs in ({}, {"in_uncond": True}, {"uncond_bias": -0.7}, {"mod_guidance": True}):
            with self.subTest(**kwargs):
                self.call(self.ad, mode="style", **kwargs)
                calls = []
                for mode in ("cs", "char"):
                    out = self.call(self.ad, mode=mode, calls=calls, **kwargs)
                    self.assertTrue(torch.equal(out, self.call(self.plain, mode=mode, **kwargs)))
                self.assertTrue(all(c[1] == () for c in calls), calls)


if __name__ == "__main__":
    unittest.main()
