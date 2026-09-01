import unittest

from app.teeth_context import extract_teeth_context


class TeethContextExtractionTests(unittest.TestCase):
    def test_dreamer_one_painful_front_tooth(self):
        context = extract_teeth_context("One of my front teeth fell out with pain.")
        self.assertTrue(context["has_teeth"])
        self.assertEqual("dreamer", context["owner"])
        self.assertEqual("one", context["count"])
        self.assertEqual("painful", context["pain"])
        self.assertIn("front", context["positions"])

    def test_dreamer_multiple_painless_teeth(self):
        context = extract_teeth_context("Several of my teeth fell out without pain.")
        self.assertEqual("dreamer", context["owner"])
        self.assertEqual("multiple", context["count"])
        self.assertEqual("painless", context["pain"])

    def test_relationship_owner_is_not_dreamer(self):
        context = extract_teeth_context("My sister's tooth came out.")
        self.assertEqual("other", context["owner"])
        self.assertEqual("sister", context["owner_relationship"])
        self.assertEqual("one", context["count"])

    def test_pronoun_owner_is_other_without_inventing_relationship(self):
        context = extract_teeth_context("Her back tooth fell out.")
        self.assertEqual("other", context["owner"])
        self.assertEqual("", context["owner_relationship"])
        self.assertIn("back", context["positions"])

    def test_molar_is_structural_back_position_only(self):
        context = extract_teeth_context("A molar was loose.")
        self.assertTrue(context["has_teeth"] is False)
        # Molar alone is intentionally not promoted into the Teeth family yet.
        # This protects the extractor from silently expanding aliases without
        # an explicit integration decision.

    def test_upper_lower_are_captured_without_kinship_mapping(self):
        context = extract_teeth_context("My upper tooth and lower tooth were loose.")
        self.assertIn("upper", context["positions"])
        self.assertIn("lower", context["positions"])
        self.assertNotIn("mother", context)
        self.assertNotIn("father", context)

    def test_absence_of_pain_language_stays_unknown(self):
        context = extract_teeth_context("My tooth fell out.")
        self.assertEqual("unknown", context["pain"])

    def test_non_teeth_dream_does_not_invent_context(self):
        context = extract_teeth_context("I was walking beside the ocean.")
        self.assertFalse(context["has_teeth"])
        self.assertEqual("unknown", context["owner"])
        self.assertEqual("unknown", context["count"])
        self.assertEqual("unknown", context["pain"])
        self.assertEqual([], context["positions"])


if __name__ == "__main__":
    unittest.main()
