"""The cond/uncond split must hand each part its own share of the per-batch extras.

By default only the positive branch reads the reference, so a batched [cond, uncond] call is split in two.
ppm NegPip passes its mask as an extra cond (c_ppm_negpip_mask, one row per batch row) and multiplies the
cross-attention values by it; with the whole-batch mask on a one-row call, older ppm raised in KSampler and
newer ppm skips NegPip. The inner executor below stands for NegPip's diffusion-model wrapper (pre-guard ppm).
"""
import unittest
from unittest.mock import patch

import torch

from test_model_reload import TinyModel, load_snapref

snapref = load_snapref()


class Executor:
    def __init__(self, dm, calls):
        self.class_obj, self.calls = dm, calls

    def __call__(self, x, timesteps, context, fps=None, padding_mask=None, **kwargs):
        mask = kwargs["c_ppm_negpip_mask"]
        v = torch.ones(x.shape[0], 512, 4)
        v = v * mask   # ppm NegPip before its shape guard: a two-row mask turns a one-row v into two rows
        if v.shape[0] != x.shape[0]:
            raise RuntimeError(f"NegPip mask has {mask.shape[0]} rows for a {x.shape[0]}-row call")
        opts = kwargs["transformer_options"]
        self.calls.append((x.shape[0], mask[:, 0, 0].tolist(), list(opts["cond_or_uncond"]), list(opts["uuids"]),
                           opts["sigmas"].tolist(), kwargs["c_other"].shape[0], kwargs["scalar"],
                           dict(snapref._STATE), None if padding_mask is None else padding_mask.shape[0]))
        return x


def encode(*args):
    return [(torch.ones(1, 1, 1, 2), torch.ones(1, 1, 1, 2))]


class SplitExtrasTests(unittest.TestCase):
    def setUp(self):
        self.ad = type("Ad", (), {"cache": {}, "install": lambda self, dm: None})()
        self.addCleanup(snapref._STATE.update, adapter=None, stream=None)

    def call(self, flags, in_uncond=False, uncond_bias=None, batch_per_chunk=1):
        B = batch_per_chunk * len(flags)
        calls = []
        wrapper = snapref._make_wrapper(self.ad, [torch.ones(1, 2)], (torch.ones(1, 1, 2), {}), "cs", in_uncond,
                                        uncond_bias=uncond_bias)
        x = torch.arange(B, dtype=torch.float32).view(B, 1, 1, 1, 1).expand(B, 2, 1, 2, 2).contiguous()
        options = {"cond_or_uncond": list(flags), "uuids": [f"u{f}" for f in flags],
                   "sigmas": torch.arange(B, dtype=torch.float32), "sample_sigmas": torch.ones(5)}
        mask = torch.arange(1, B + 1, dtype=torch.float32).view(B, 1, 1).expand(B, 512, 1)
        with patch.object(snapref, "_encode_reference", side_effect=encode):
            out = wrapper(Executor(TinyModel(), calls), x, torch.arange(B, dtype=torch.float32), torch.ones(B, 1, 2),
                          None, torch.zeros(B, 1, 2, 2), transformer_options=options, c_ppm_negpip_mask=mask,
                          c_other=torch.zeros(B, 3), scalar=7)
        torch.testing.assert_close(out, x)
        self.assertEqual(options["cond_or_uncond"], list(flags))   # the caller's options are not mutated
        return calls

    def test_split_hands_each_part_its_rows(self):
        for flags in ([0, 1], [1, 0]):
            for per in (1, 2):
                with self.subTest(flags=flags, batch_per_chunk=per):
                    calls = self.call(flags, batch_per_chunk=per)
                    self.assertEqual(len(calls), 2)
                    for i, (rows, mask_rows, cou, uuids, sigmas, other, scalar, state, pm) in enumerate(calls):
                        expect = list(range(i * per, (i + 1) * per))
                        self.assertEqual(rows, per)
                        self.assertEqual(mask_rows, [r + 1.0 for r in expect])
                        self.assertEqual(cou, [flags[i]])
                        self.assertEqual(uuids, [f"u{flags[i]}"])
                        self.assertEqual(sigmas, [float(r) for r in expect])
                        self.assertEqual((other, scalar, pm), (per, 7, per))

    def test_unsplit_calls_keep_the_whole_batch(self):
        for kwargs in ({"in_uncond": True}, {"uncond_bias": -0.7}):
            with self.subTest(**kwargs):
                calls = self.call([0, 1], **kwargs)
                self.assertEqual(len(calls), 1)
                rows, mask_rows, cou, uuids, sigmas, other, scalar, state, pm = calls[0]
                self.assertEqual((rows, mask_rows, cou, uuids, sigmas, other, pm), (2, [1.0, 2.0], [0, 1], ["u0", "u1"],
                                                                                    [0.0, 1.0], 2, 2))


if __name__ == "__main__":
    unittest.main()
