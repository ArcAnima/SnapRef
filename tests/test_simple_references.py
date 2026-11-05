from types import SimpleNamespace
import unittest
from unittest.mock import patch

from test_model_reload import load_snapref

snapref = load_snapref()
simple = snapref._simple


class SimpleReferenceTests(unittest.TestCase):
    def test_schema_preserves_old_inputs_and_appends_six_pairs(self):
        with patch.object(simple, 'input_images', return_value=['a.png']), patch.object(simple, 'snapref_files', return_value=['a.safetensors']):
            schema = snapref.SnapRefSimple.INPUT_TYPES()
        self.assertEqual(list(schema['optional'])[:2], ['reference_2', 'strength_2'])
        for i in range(3, 9):
            self.assertEqual(schema['optional'][f'reference_{i}'][1]['default'], 'none')
            self.assertEqual(schema['optional'][f'strength_{i}'][1]['default'], 1.0)

    def test_apply_passes_extra_images_and_matching_weights(self):
        captured = []
        def wrapper(ad, latents, cond, mode, uncond, bias, weights, fidelity):
            captured.append((latents, weights))
            return 'wrapper'
        class Model:
            def __init__(self): self.wrappers = {'diffusion': {}}
            def clone(self): return Model()
            def get_model_object(self, name): return object()
            def add_wrapper_with_key(self, kind, key, value): self.wrappers[kind][key] = [value]
        ad = SimpleNamespace(block_indices={0}, meta={})
        core = SimpleNamespace(_adapters={'adapter': ad}, _Adapter=None, _adapter_block_map=lambda *a: None,
                               _encode_image=lambda m, v, px, size: (px, None), _make_wrapper=wrapper, WRAPPER_KEY='snapref')
        cls = simple.make_simple_node(core)
        clip = SimpleNamespace(tokenize=lambda s: s, encode_from_tokens_scheduled=lambda t: [('condition', {})])
        extra = {f'reference_{i}': f'{i}.png' for i in range(3, 9)}
        extra.update(strength_3=0.0, strength_8=0.25)
        with patch.object(simple.folder_paths, 'get_full_path_or_raise', return_value='adapter', create=True), \
             patch.object(simple, 'load_input_image', side_effect=lambda name: name), \
             patch.object(simple.comfy.patcher_extension, 'WrappersMP', SimpleNamespace(DIFFUSION_MODEL='diffusion'), create=True):
            cls().apply(Model(), clip, None, 'adapter', '1.png', 'Character', 1.0, 1.0, '2.png', 0.5)
            cls().apply(Model(), clip, None, 'adapter', '1.png', 'Character', 1.0, 1.0, '2.png', 0.5, **extra)
        self.assertEqual(captured[0], (['1.png', '2.png'], [1.0, 0.5]))
        self.assertEqual(captured[1], (['1.png', '2.png', '4.png', '5.png', '6.png', '7.png', '8.png'], [1.0, 0.5, 1.0, 1.0, 1.0, 1.0, 0.25]))

    def test_validation_checks_last_reference(self):
        with patch.object(simple.folder_paths, 'exists_annotated_filepath', side_effect=lambda n: n != 'missing.png', create=True):
            self.assertNotEqual(snapref.SnapRefSimple.VALIDATE_INPUTS(reference='a.png', reference_8='missing.png'), True)

    def test_cache_fingerprint_tracks_last_reference(self):
        with patch.object(simple.folder_paths, 'get_annotated_filepath', side_effect=lambda n: n, create=True), \
             patch.object(simple.os.path, 'exists', return_value=True), \
             patch.object(simple.os.path, 'getmtime', return_value=1):
            before = snapref.SnapRefSimple.IS_CHANGED(reference='a.png', reference_8='b.png')
            with patch.object(simple.os.path, 'getmtime', side_effect=lambda n: 2 if n == 'b.png' else 1):
                after = snapref.SnapRefSimple.IS_CHANGED(reference='a.png', reference_8='b.png')
        self.assertNotEqual(before, after)
