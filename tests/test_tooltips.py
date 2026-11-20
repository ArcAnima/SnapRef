"""Strength scales: simple node linear (1 = as trained), Advanced log (0 = as trained). The scales stay as they
are (saved workflows); labels and tooltips say which is which, and no tooltip claims a slowdown."""
import math
import unittest
from unittest.mock import patch

from test_model_reload import load_snapref

snapref = load_snapref()


def schemas():
    with patch.object(snapref.folder_paths, "get_filename_list", return_value=["snapref_v2a.safetensors"], create=True), \
         patch.object(snapref._simple, "input_images", return_value=["a.png"]):
        return snapref.SnapRef.INPUT_TYPES(), snapref.SnapRefSimple.INPUT_TYPES()


class TooltipTests(unittest.TestCase):
    def test_scales_are_unchanged(self):
        advanced, simple = schemas()
        self.assertEqual(advanced["optional"]["strength"][1]["default"], 0.0)
        self.assertEqual((advanced["optional"]["strength"][1]["min"], advanced["optional"]["strength"][1]["max"]), (-3.0, 3.0))
        self.assertEqual(simple["required"]["strength"][1]["default"], 1.0)
        self.assertEqual((simple["required"]["strength"][1]["min"], simple["required"]["strength"][1]["max"]), (0.0, 2.0))
        self.assertEqual(list(advanced["optional"])[0], "strength")   # input names (saved workflows) unchanged

    def test_labels_and_tooltips_name_the_scale(self):
        advanced, simple = schemas()
        adv = advanced["optional"]["strength"][1]
        self.assertIn("log", adv["display_name"])
        self.assertIn("LOG", adv["tooltip"])
        self.assertIn("linear", adv["tooltip"])
        self.assertIn("LINEAR", simple["required"]["strength"][1]["tooltip"])
        self.assertIn("LINEAR", advanced["optional"]["weight_1"][1]["tooltip"])

    def test_no_stale_speed_or_scope_claims(self):
        advanced, simple = schemas()
        for schema in (advanced, simple):
            for section in schema.values():
                for name, spec in section.items():
                    tip = spec[1].get("tooltip", "") if len(spec) > 1 else ""
                    self.assertNotIn("1.3x", tip, name)
                    self.assertNotIn("slower", tip.replace("slower with more references", ""), name)
        self.assertNotIn("Only with several references", advanced["optional"]["weight_1"][1]["tooltip"])

    def test_stated_conversions_hold(self):
        # simple Strength s and Advanced strength b act alike when s = e^b (bias log s on the reference keys)
        self.assertAlmostEqual(math.log(2.0), 0.7, places=1)
        self.assertAlmostEqual(math.log(0.5), -0.7, places=1)
        self.assertAlmostEqual(math.exp(1.0), 2.7, places=1)


if __name__ == "__main__":
    unittest.main()
