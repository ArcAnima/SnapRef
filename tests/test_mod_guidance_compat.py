"""Mod Guidance owns the full CFG batch; reference encoding must not inherit it."""
import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from test_model_reload import load_snapref, TinyModel

snapref = load_snapref()


class ModGuidanceTests(unittest.TestCase):
    def test_mod_cfg_batch_is_preserved_and_reference_is_unmodulated(self):
        dm = TinyModel()
        delta, block_state = object(), object()
        dm.t_embedding_norm = SimpleNamespace(_anima_mod_delta=delta)
        dm._anima_mod_block_state = block_state
        ad = SimpleNamespace(cache={}, install=lambda dm: None)
        calls = []
        case = self
        class Executor:
            class_obj = dm
            def __call__(self, x, *args, **kwargs):
                case.assertEqual(x.shape[0], 2, 'Mod Guidance owns a two-row CFG batch')
                case.assertIs(dm.t_embedding_norm._anima_mod_delta, delta)
                case.assertIs(dm._anima_mod_block_state, block_state)
                calls.append(dm.blocks[0].self_attn._ref_kv_state['rows'])
                return x
        def encode(*args):
            case.assertIsNone(dm.t_embedding_norm._anima_mod_delta)
            case.assertIsNone(dm._anima_mod_block_state)
            return [(torch.ones(1, 1, 1, 2), torch.ones(1, 1, 1, 2))]
        wrapper = snapref._make_wrapper(ad, [torch.ones(1, 2)], (torch.ones(1, 1, 2), {}), 'cs', False)
        with patch.object(snapref, '_encode_reference', side_effect=encode):
            result = wrapper(Executor(), torch.ones(2, 2), torch.ones(2), torch.ones(2, 1, 2),
                             transformer_options={'cond_or_uncond': [0, 1]})
        self.assertEqual(result.shape[0], 2)
        self.assertEqual(calls, [[0]])

    def test_rows_fallback_preserves_negative_branch_and_fidelity(self):
        q = torch.zeros(2, 1, 1, 2)
        ref = (torch.zeros(1, 1, 1, 2), torch.full((1, 1, 1, 2), 2.))
        actual = snapref._rows_attention(q, q, q, ref, [1], [0.], [0], [math.log(.5)])
        self.assertIsNotNone(actual)
        torch.testing.assert_close(actual, torch.tensor([[[2/3, 2/3]], [[0., 0.]]]))

    def test_reference_scope_restores_mod_state_after_error(self):
        dm = TinyModel()
        delta, block_state = object(), object()
        dm.t_embedding_norm = SimpleNamespace(_anima_mod_delta=delta)
        dm._anima_mod_block_state = block_state
        with self.assertRaisesRegex(RuntimeError, 'reference failed'):
            with snapref._reference_context(dm):
                self.assertIsNone(dm.t_embedding_norm._anima_mod_delta)
                self.assertIsNone(dm._anima_mod_block_state)
                raise RuntimeError('reference failed')
        self.assertIs(dm.t_embedding_norm._anima_mod_delta, delta)
        self.assertIs(dm._anima_mod_block_state, block_state)


if __name__ == '__main__':
    unittest.main()
