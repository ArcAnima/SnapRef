"""Device checks for the Mac / Intel fixes. Skipped unless the device is present, so they run where a CPU-only
test cannot reach: python -m unittest discover -s tests -v   (on a Mac with MPS, or a PyTorch XPU build)."""
import math
import unittest

import torch

from test_correction_dtype import reference_apg
from test_masked_attention import CASES, exact, inputs, snapref

DEVICES = [d for d, ok in (("mps", getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
                           ("xpu", hasattr(torch, "xpu") and torch.xpu.is_available())) if ok]


@unittest.skipUnless(DEVICES, "no MPS / XPU device")
class DevicePathTests(unittest.TestCase):
    def test_masked_attention_matches_cpu(self):
        for device in DEVICES:
            for dtype in (torch.float32, torch.float16):
                for name, B, S, sizes, biases, rows, row_biases in CASES:
                    q, k, v, (kr, vr) = inputs(B, S, sizes, dtype=dtype)
                    args = [t.to(device) for t in (q, k, v)]
                    with self.subTest(name, device=device, dtype=dtype):
                        out = snapref._masked_attention(*args, (kr.to(device), vr.to(device)), sizes, biases, rows, row_biases)
                        self.assertTrue(torch.isfinite(out).all(), "NaN / inf: the black-image path")
                        torch.testing.assert_close(out.cpu().double(), exact(q, k, v, (kr, vr), sizes, biases, rows, row_biases),
                                                   rtol=3e-2, atol=3e-2)

    def test_apg_runs_without_float64(self):
        for device in DEVICES:
            snapref._FP64_ARITH.pop(device, None)
            guided, pos = torch.randn(1, 16, 1, 64, 64), torch.randn(1, 16, 1, 64, 64)
            with self.subTest(device=device):
                out = snapref._correct(guided.to(device), pos.to(device), "apg")
                self.assertTrue(torch.isfinite(out).all())
                self.assertLess((out.cpu().double() - reference_apg(guided, pos).double()).abs().max().item(), 1e-3)
                print(f"{device}: float64 arithmetic {'yes' if snapref._fp64_arith_ok(torch.device(device)) else 'no'}")

    def test_fidelity_is_graded(self):
        from test_fidelity_fallback import FidelityFallbackTests
        for device in DEVICES:
            case = FidelityFallbackTests("test_fidelity_is_graded_on_cpu")
            case.setUp()
            x, kv = case.x, case.kv
            case.x, case.kv = x.to(device), tuple(t.to(device) for t in kv)
            full = case.run_node(in_uncond=True)
            half = case.run_node(fidelity=0.5)
            case.x, case.kv = x, kv   # float64 reference on the CPU
            with self.subTest(device=device):
                self.assertTrue(torch.isfinite(half).all())
                self.assertGreater((half[1] - full[1]).abs().max().item(), 1e-3)
                torch.testing.assert_close(half.cpu().float(), case.expected(math.log(0.5)).float(), rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    unittest.main()
