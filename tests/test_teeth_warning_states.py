import unittest

from app.teeth_context import extract_teeth_context
from app.teeth_doctrine import build_teeth_doctrine_context, build_teeth_narration_facts
from app.teeth_integration import attach_teeth_narration_facts


class TeethWarningStateTests(unittest.TestCase):
    def test_physical_tooth_pain_is_distinct_from_emotional_hurt(self):
        physical = extract_teeth_context("My tooth fell out and it hurt badly.")
        emotional = extract_teeth_context("My tooth fell out and I felt hurt and sad afterward.")

        self.assertEqual(physical["pain"], "painful")
        self.assertEqual(emotional["pain"], "unknown")

    def test_painful_fallout_adds_proximity_and_emotional_intensity(self):
        doctrine = build_teeth_doctrine_context("One of my teeth fell out and it was painful.")

        self.assertTrue(doctrine["active_fallout"])
        self.assertEqual(doctrine["warning_count"], "one_person")
        self.assertEqual(doctrine["proximity"], "very_close_or_close_relative")
        self.assertEqual(doctrine["emotional_intensity"], "heightened")

    def test_loose_tooth_is_sickness_warning_without_completed_loss(self):
        doctrine = build_teeth_doctrine_context("My tooth was loose and wobbly but it never fell out.")
        narration = build_teeth_narration_facts("My tooth was loose and wobbly but it never fell out.")

        self.assertTrue(doctrine["active_warning"])
        self.assertFalse(doctrine["active_fallout"])
        self.assertTrue(doctrine["loose_warning"])
        self.assertEqual(doctrine["warning_kind"], "loose_sickness")
        self.assertEqual(doctrine["warning_count"], "")
        self.assertIn("sickness warning", narration["lead"].lower())
        self.assertIn("not a medical diagnosis", narration["lead"].lower())

    def test_loose_tooth_restoration_is_preserved_without_invented_cancellation(self):
        narration = build_teeth_narration_facts("My tooth was loose, but later it became firm again.")

        self.assertTrue(narration["active"])
        self.assertTrue(narration["restorative_state"])
        joined = " ".join(narration["details"]).lower()
        self.assertIn("restorative ending", joined)
        self.assertIn("no cancellation meaning", joined)

    def test_standalone_bleeding_gums_is_approaching_omen_warning(self):
        doctrine = build_teeth_doctrine_context("My gums were bleeding in the dream.")
        narration = build_teeth_narration_facts("My gums were bleeding in the dream.")

        self.assertTrue(doctrine["active_warning"])
        self.assertFalse(doctrine["active_fallout"])
        self.assertTrue(doctrine["bleeding_gums_warning"])
        self.assertEqual(doctrine["warning_kind"], "bleeding_gums")
        self.assertIn("bad omen may be approaching", narration["lead"].lower())
        self.assertNotIn("blood relative", narration["lead"].lower())

    def test_bleeding_from_brushing_does_not_force_standalone_omen_rule(self):
        doctrine = build_teeth_doctrine_context("I was brushing my teeth and my gums started bleeding.")

        self.assertTrue(doctrine["bleeding_physical_cause"])
        self.assertFalse(doctrine["bleeding_gums_warning"])
        self.assertFalse(doctrine["active_warning"])

    def test_blood_on_fallen_tooth_increases_severity_only(self):
        doctrine = build_teeth_doctrine_context("One of my teeth fell out and there was blood on the tooth.")
        narration = build_teeth_narration_facts("One of my teeth fell out and there was blood on the tooth.")

        self.assertTrue(doctrine["active_fallout"])
        self.assertTrue(doctrine["blood_on_fallen_tooth"])
        self.assertEqual(doctrine["severity_modifier"], "increased")
        self.assertEqual(doctrine["warning_count"], "one_person")
        self.assertEqual(doctrine["relationship_scope"], "relative_or_close_friend")

        joined = " ".join(narration["details"]).lower()
        self.assertIn("increases the severity or intensity", joined)
        self.assertIn("does not determine who is involved", joined)

    def test_gum_bleeding_is_not_misread_as_blood_on_fallen_tooth(self):
        doctrine = build_teeth_doctrine_context("My gums were bleeding and then one of my teeth fell out.")

        self.assertTrue(doctrine["active_fallout"])
        self.assertFalse(doctrine["blood_on_fallen_tooth"])
        self.assertEqual(doctrine["severity_modifier"], "")

    def test_other_person_bloody_tooth_does_not_gain_dreamer_relationship_scope(self):
        doctrine = build_teeth_doctrine_context("My sister's tooth fell out and the tooth was covered in blood.")

        self.assertTrue(doctrine["active_fallout"])
        self.assertEqual(doctrine["owner"], "other")
        self.assertEqual(doctrine["owner_relationship"], "sister")
        self.assertEqual(doctrine["relationship_scope"], "")
        self.assertEqual(doctrine["warning_count"], "one_person")
        self.assertEqual(doctrine["severity_modifier"], "increased")

    def test_live_bridge_uses_approved_state_warning_and_suppresses_generic_risk(self):
        facts = {
            "lead_message": "generic",
            "risk": "Low risk",
            "state_meaning": "generic state meaning",
            "event_context": {"primary_state": {"name": "Loose", "meaning": "generic"}},
        }
        enriched = attach_teeth_narration_facts("My tooth was loose and wobbly.", facts)

        self.assertIn("sickness warning", enriched["lead_message"].lower())
        self.assertEqual(enriched["risk"], "")
        self.assertEqual(enriched["state_meaning"], "")
        self.assertEqual(enriched["teeth_narration"]["warning_kind"], "loose_sickness")


if __name__ == "__main__":
    unittest.main()
