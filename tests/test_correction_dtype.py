"""CPU regressions for the ref_guidance correction dtype; run with python -m unittest discover -s tests.

Intel XPU has no float64 arithmetic ("Required aspect fp64 is not supported on the device",
"Kernel is incompatible with all devices in devs"), so the APG correction has to leave float64
on such a backend. It must not do so on a whitelist: any backend whose arithmetic probe passes
keeps float64, including ones this test suite has never heard of. The rescale branch never used
float64 and must keep running in the model dtype on every backend.
"""
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from test_model_reload import load_snapref

snapref = load_snapref()


def reference_apg(guided, pos):
    """APG written exactly as the pre-fix code, the float64 value the CUDA path has to keep producing."""
    dims = tuple(range(1, pos.ndim))
    update = (guided - pos).double()
    unit = F.normalize(pos.double().flatten(1), dim=1).view_as(pos)
    return pos + (update - (update * unit).sum(dim=dims, keepdim=True) * unit).to(pos.dtype)


class ProbeTests(unittest.TestCase):
    def setUp(self):
        # the probe caches by device type and honours SNAPREF_FP64: isolate every test from both
        self.addCleanup(snapref._FP64_ARITH.update, dict(snapref._FP64_ARITH))
        snapref._FP64_ARITH.clear()
        env = patch.dict(os.environ, {}, clear=False)
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop("SNAPREF_FP64", None)

    def test_cpu_probe_is_true(self):
        # measured on this machine: CPU has float64 arithmetic
        self.assertIs(snapref._fp64_arith_ok(torch.device("cpu")), True)

    def test_probe_result_is_cached_per_device_type(self):
        calls = []
        real_ones = torch.ones

        def counting_ones(*args, **kwargs):
            calls.append(kwargs.get("dtype"))
            return real_ones(*args, **kwargs)

        with patch.object(torch, "ones", counting_ones):
            for _ in range(5):
                self.assertIs(snapref._fp64_arith_ok(torch.device("cpu")), True)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0], torch.float64)

    def test_probe_failure_degrades_to_float32(self):
        for error in (RuntimeError("Required aspect fp64 is not supported on the device"),
                      NotImplementedError("no fp64 scalar type"),
                      TypeError("unsupported dtype")):
            snapref._FP64_ARITH.clear()
            with patch.object(torch, "ones", side_effect=error):
                self.assertIs(snapref._fp64_arith_ok(torch.device("xpu")), False, repr(error))

    def test_failed_probe_is_cached_too(self):
        calls = []

        def failing_ones(*args, **kwargs):
            calls.append(1)
            raise RuntimeError("Kernel is incompatible with all devices in devs")

        with patch.object(torch, "ones", failing_ones):
            for _ in range(3):
                self.assertIs(snapref._fp64_arith_ok(torch.device("xpu")), False)
        self.assertEqual(len(calls), 1)

    def test_reduction_failure_alone_is_enough_to_degrade(self):
        # elementwise math can work while reductions do not: both are probed
        real_ones = torch.ones

        class SumFails:
            def __init__(self, inner):
                self.inner = inner

            def __mul__(self, other):
                return SumFails(self.inner * other)

            def __add__(self, other):
                return SumFails(self.inner + other)

            def sum(self, *args, **kwargs):
                raise RuntimeError("Kernel is incompatible with all devices in devs")

            def item(self):
                return self.inner.item()

        with patch.object(torch, "ones", lambda *a, **kw: SumFails(real_ones(*a, **kw))):
            self.assertIs(snapref._fp64_arith_ok(torch.device("xpu")), False)

    def test_unlisted_backend_keeps_float64_when_the_probe_passes(self):
        # the point of probing instead of whitelisting: an unknown device is not degraded unless
        # arithmetic actually fails there. torch.ones returning a real tensor stands in for a
        # backend that does support float64.
        real_ones = torch.ones
        for kind in ("mlu", "npu", "privateuseone", "xpu", "mps"):
            snapref._FP64_ARITH.clear()
            fake = SimpleNamespace(device=SimpleNamespace(type=kind), is_cuda=(kind == "cuda"))
            with patch.object(torch, "ones", lambda *a, **kw: real_ones(8, dtype=torch.float64)):
                self.assertEqual(snapref._correction_dtype(fake), torch.float64, kind)
            self.assertIs(snapref._FP64_ARITH[kind], True)

    def test_unlisted_backend_degrades_when_the_probe_fails(self):
        for kind in ("mlu", "npu", "privateuseone", "xpu", "mps"):
            snapref._FP64_ARITH.clear()
            fake = SimpleNamespace(device=SimpleNamespace(type=kind), is_cuda=(kind == "cuda"))
            with patch.object(torch, "ones", side_effect=RuntimeError("Required aspect fp64 is not supported on the device")):
                self.assertEqual(snapref._correction_dtype(fake), torch.float32, kind)
            self.assertIs(snapref._FP64_ARITH[kind], False)

    def test_env_override_forces_both_ways(self):
        for raw, expected in (("0", False), ("false", False), ("off", False), ("no", False),
                              ("1", True), ("true", True), ("on", True), ("yes", True)):
            os.environ["SNAPREF_FP64"] = raw
            self.assertIs(snapref._fp64_arith_ok(torch.device("cpu")), expected, raw)
            self.assertIs(snapref._fp64_arith_ok(torch.device("xpu")), expected, raw)

    def test_forced_float32_skips_the_probe(self):
        os.environ["SNAPREF_FP64"] = "0"
        with patch.object(torch, "ones", side_effect=AssertionError("must not probe")):
            self.assertIs(snapref._fp64_arith_ok(torch.device("cpu")), False)
            self.assertEqual(snapref._correction_dtype(torch.zeros(1, 1, 2, 2)), torch.float32)

    def test_unset_or_unknown_env_does_not_override(self):
        for raw in ("", "maybe", "2", " "):
            os.environ["SNAPREF_FP64"] = raw
            self.assertIsNone(snapref._fp64_env_override(), raw)

    def test_correction_dtype_follows_the_probe(self):
        snapref._FP64_ARITH.clear()
        self.assertEqual(snapref._correction_dtype(torch.zeros(1, 1, 2, 2)), torch.float64)
        snapref._FP64_ARITH.clear()
        with patch.object(snapref, "_fp64_arith_ok", lambda device: False):
            self.assertEqual(snapref._correction_dtype(torch.zeros(1, 1, 2, 2)), torch.float32)


class CorrectionTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(snapref._FP64_ARITH.update, dict(snapref._FP64_ARITH))
        snapref._FP64_ARITH.clear()
        env = patch.dict(os.environ, {}, clear=False)
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop("SNAPREF_FP64", None)

    def test_apg_avoids_float64_math_where_the_backend_cannot_run_it(self):
        guided, pos = torch.randn(2, 4, 8, 8), torch.randn(2, 4, 8, 8)
        seen = []
        real_normalize = F.normalize

        def spy_normalize(t, *args, **kwargs):
            seen.append(t.dtype)
            return real_normalize(t, *args, **kwargs)

        with patch.object(snapref, "_fp64_arith_ok", lambda device: False), \
             patch.object(torch.nn.functional, "normalize", spy_normalize):
            snapref._correct(guided, pos, "apg")
        self.assertEqual(seen, [torch.float32])

    def test_apg_keeps_the_float64_result_on_cpu(self):
        torch.manual_seed(0)
        guided = torch.randn(2, 4, 16, 16)
        pos = torch.randn(2, 4, 16, 16) * 0.5
        expected = reference_apg(guided, pos)
        self.assertEqual(expected.dtype, pos.dtype)
        torch.testing.assert_close(snapref._correct(guided, pos, "apg"), expected)

    def test_float32_apg_is_close_to_the_float64_one(self):
        torch.manual_seed(0)
        guided = torch.randn(2, 16, 1, 64, 96)   # Anima x0 prediction scale
        pos = guided + 0.3 * torch.randn_like(guided)
        expected = reference_apg(guided, pos).double()
        with patch.object(snapref, "_fp64_arith_ok", lambda device: False):
            actual = snapref._correct(guided, pos, "apg")
        self.assertEqual(actual.dtype, pos.dtype)
        self.assertLess((actual.double() - expected).abs().max().item(), 1e-5)

    def test_apg_drops_the_component_parallel_to_the_condition(self):
        guided, pos = torch.randn(1, 4, 8, 8), torch.randn(1, 4, 8, 8)
        for forced in (True, False):
            with patch.object(snapref, "_fp64_arith_ok", lambda device, f=forced: f):
                out = snapref._correct(guided, pos, "apg")
            unit = F.normalize(pos.double().flatten(1), dim=1)
            offset = (out - pos).double().flatten(1)
            self.assertLess(offset.mul(unit).sum(dim=1).abs().max().item(), 1e-4)

    def test_apg_keeps_the_input_dtype_on_both_paths(self):
        for forced in (True, False):
            for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
                guided = torch.randn(1, 4, 8, 8).to(dtype)
                pos = torch.randn(1, 4, 8, 8).to(dtype)
                with patch.object(snapref, "_fp64_arith_ok", lambda device, f=forced: f):
                    actual = snapref._correct(guided, pos, "apg")
                self.assertEqual(actual.dtype, dtype, f"float64={forced} {dtype}")
                self.assertTrue(torch.isfinite(actual).all())

    def test_rescale_never_promotes_to_float64(self):
        # rescale always ran in the model dtype; it must not start using float64 on any backend
        dims = (1, 2, 3)
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            guided = torch.randn(2, 4, 8, 8).to(dtype) * 2 + 1
            pos = torch.randn(2, 4, 8, 8).to(dtype)
            actual = snapref._correct(guided, pos, "rescale")
            self.assertEqual(actual.dtype, dtype)
            ratio = pos.float().std(dim=dims, keepdim=True) / guided.float().std(dim=dims, keepdim=True)
            expected = (0.5 * guided.float() * ratio + 0.5 * guided.float()).to(dtype)
            torch.testing.assert_close(actual, expected, rtol=1e-2 if dtype != torch.float32 else 1e-6,
                                       atol=1e-2 if dtype != torch.float32 else 1e-6)

    def test_no_correction_is_identity(self):
        guided = torch.randn(1, 4, 8, 8)
        self.assertIs(snapref._correct(guided, torch.randn(1, 4, 8, 8), "none"), guided)

    def test_correct_does_not_probe_per_call(self):
        guided, pos = torch.randn(1, 4, 8, 8), torch.randn(1, 4, 8, 8)
        calls = []
        real_ones = torch.ones

        def counting_ones(*args, **kwargs):
            calls.append(1)
            return real_ones(*args, **kwargs)

        with patch.object(torch, "ones", counting_ones):
            for _ in range(30):
                snapref._correct(guided, pos, "apg")
        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
