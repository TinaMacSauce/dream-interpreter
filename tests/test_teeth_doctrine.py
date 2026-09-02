import unittest

from app.teeth_doctrine import build_teeth_doctrine_context, build_teeth_narration_facts


class TeethDoctrineContextTests(unittest.TestCase):
    def test_one_painful_dreamer_tooth_maps_approved_rules(self):
        facts = build_teeth_doctrine_context("One of my front teeth fell out with pain.")
        self.assertTrue(facts["active_fallout"])
        self.assertEqual(facts["relationship_scope"], "relative_or_close_friend")
        self.assertEqual(facts["warning_count"], "one_person")
        self.assertEqual(facts["proximity"], "very_close_or_close_relative")
        self.assertIn("front", facts["positions"])

    def test_multiple_painless_dreamer_teeth_maps_approved_rules(self):
        facts = build_teeth_doctrine_context("Several of my teeth fell out without pain.")
        self.assertTrue(facts["active_fallout"])
        self.assertEqual(facts["relationship_scope"], "relative_or_close_friend")
        self.assertEqual(facts["warning_count"], "multiple_people")
        self.assertEqual(facts["proximity"], "friend_acquaintance_or_more_distant")

    def test_other_person_tooth_does_not_inherit_dreamer_relationship_scope(self):
        facts = build_teeth_doctrine_context("My sister's tooth fell out with pain.")
        self.assertTrue(facts["active_fallout"])
        self.assertEqual(facts["owner"], "other")
        self.assertEqual(facts["owner_relationship"], "sister")
        self.assertEqual(facts["relationship_scope"], "")
        self.assertEqual(facts["warning_count"], "one_person")

    def test_negated_fallout_does_not_activate_doctrine(self):
        facts = build_teeth_doctrine_context("My tooth did not fall out.")
        self.assertFalse(facts["active_fallout"])
        self.assertEqual(facts["warning_count"], "")
        self.assertEqual(facts["proximity"], "")

    def test_hypothetical_fallout_does_not_activate_doctrine(self):
        facts = build_teeth_doctrine_context("I was afraid my tooth might fall out with pain.")
        self.assertFalse(facts["active_fallout"])
        self.assertEqual(facts["warning_count"], "")
        self.assertEqual(facts["proximity"], "")

    def test_near_miss_does_not_activate_doctrine(self):
        facts = build_teeth_doctrine_context("My tooth almost fell out without pain.")
        self.assertFalse(facts["active_fallout"])
        self.assertEqual(facts["warning_count"], "")

    def test_position_remains_structural_only(self):
        facts = build_teeth_doctrine_context("One of my lower teeth fell out.")
        self.assertTrue(facts["active_fallout"])
        self.assertIn("lower", facts["positions"])
        self.assertNotIn("child", str(facts).lower())
        self.assertNotIn("younger", str(facts).lower())
        self.assertNotIn("blood", str(facts).lower())

    def test_narration_one_painful_own_tooth_is_doctrine_safe(self):
        facts = build_teeth_narration_facts("One of my front teeth fell out with pain.")
        self.assertTrue(facts["active"])
        self.assertIn("one fallen tooth", facts["lead"].lower())
        self.assertIn("relative or close friend", " ".join(facts["details"]).lower())
        self.assertIn("very close", " ".join(facts["details"]).lower())
        self.assertNotIn("blood", str(facts).lower())
        self.assertNotIn("front", facts["lead"].lower())

    def test_narration_multiple_painless_own_teeth_maps_approved_distinction(self):
        facts = build_teeth_narration_facts("Several of my lower teeth fell out without pain.")
        self.assertTrue(facts["active"])
        self.assertIn("multiple fallen teeth", facts["lead"].lower())
        detail_text = " ".join(facts["details"]).lower()
        self.assertIn("relative or close friend", detail_text)
        self.assertIn("friend, acquaintance", detail_text)
        self.assertNotIn("younger", str(facts).lower())
        self.assertNotIn("child", str(facts).lower())

    def test_narration_stays_inactive_for_negated_fallout(self):
        facts = build_teeth_narration_facts("My tooth did not fall out with pain.")
        self.assertFalse(facts["active"])
        self.assertEqual(facts["lead"], "")
        self.assertEqual(facts["details"], [])


if __name__ == "__main__":
    unittest.main()
