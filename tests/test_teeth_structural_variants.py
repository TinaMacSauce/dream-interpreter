import unittest

from app.teeth_context import extract_teeth_context
from app.teeth_doctrine import build_teeth_doctrine_context, build_teeth_narration_facts


class TeethStructuralVariantRegressionTests(unittest.TestCase):
    def test_broken_tooth_uses_approved_sickness_warning(self):
        context = extract_teeth_context("My tooth was broken.")
        doctrine = build_teeth_doctrine_context("My tooth was broken.")

        self.assertTrue(context["broken_or_cracked"])
        self.assertEqual("broken_or_cracked", doctrine["event_status"])
        self.assertTrue(doctrine["active_warning"])
        self.assertEqual("broken_sickness", doctrine["warning_kind"])
        self.assertIn("TEETH-STATE-BROKEN", doctrine["applied_rule_ids"])
        self.assertNotIn("broken_or_cracked_teeth", doctrine["pending_distinctions"])

    def test_negated_broken_tooth_does_not_create_pending_breakage(self):
        context = extract_teeth_context("My tooth was not broken.")
        doctrine = build_teeth_doctrine_context("My tooth was not broken.")

        self.assertFalse(context["broken_or_cracked"])
        self.assertNotIn("broken_or_cracked_teeth", doctrine["pending_distinctions"])
        self.assertFalse(doctrine["active_warning"])

    def test_near_miss_loss_is_not_completed_loss(self):
        doctrine = build_teeth_doctrine_context("My tooth almost fell out but stayed in place.")

        self.assertTrue(doctrine["near_miss_loss"])
        self.assertEqual("near_miss_loss", doctrine["event_status"])
        self.assertFalse(doctrine["active_fallout"])
        self.assertFalse(doctrine["active_warning"])
        self.assertEqual("", doctrine["warning_count"])

    def test_hypothetical_loss_is_not_completed_loss(self):
        doctrine = build_teeth_doctrine_context("I thought my tooth was going to fall out.")

        self.assertTrue(doctrine["hypothetical_loss"])
        self.assertEqual("hypothetical_loss", doctrine["event_status"])
        self.assertFalse(doctrine["active_fallout"])
        self.assertFalse(doctrine["active_warning"])

    def test_gold_teeth_without_loss_uses_limited_favorable_modifier(self):
        doctrine = build_teeth_doctrine_context("I dreamed I had gold teeth and none fell out.")
        narration = build_teeth_narration_facts("I dreamed I had gold teeth and none fell out.")

        self.assertTrue(doctrine["gold_teeth"])
        self.assertEqual("gold_without_loss", doctrine["event_status"])
        self.assertFalse(doctrine["active_warning"])
        self.assertTrue(doctrine["active_doctrine"])
        self.assertEqual("outwardly_favorable", doctrine["favorable_modifier"])
        self.assertIn("TEETH-MOD-GOLD", doctrine["applied_rule_ids"])
        self.assertTrue(narration["active"])
        self.assertIn("favorable in outward appearance", narration["lead"])

    def test_rotten_tooth_loss_with_later_healthy_growth_preserves_both_events(self):
        dream = "My rotten tooth fell out and later a new healthy tooth grew back."
        doctrine = build_teeth_doctrine_context(dream)
        narration = build_teeth_narration_facts(dream)

        self.assertTrue(doctrine["active_fallout"])
        self.assertEqual("completed_loss", doctrine["event_status"])
        self.assertEqual("one_person", doctrine["warning_count"])
        self.assertTrue(doctrine["rotten_or_decayed"])
        self.assertTrue(doctrine["replacement_growth"])
        self.assertIn("tooth_state_with_fallout_precedence", doctrine["pending_distinctions"])
        self.assertIn("replacement_growth_meaning", doctrine["pending_distinctions"])
        self.assertTrue(narration["active"])
        self.assertTrue(narration["replacement_growth"])

    def test_loose_to_firm_is_preserved_as_restored_without_cancellation_claim(self):
        dream = "My tooth was loose, but later it became firm again."
        doctrine = build_teeth_doctrine_context(dream)
        narration = build_teeth_narration_facts(dream)

        self.assertEqual("loose_restored", doctrine["event_status"])
        self.assertTrue(doctrine["loose_warning"])
        self.assertTrue(doctrine["restorative_state"])
        self.assertTrue(narration["active"])
        self.assertIn("no cancellation meaning", narration["details"][0])


if __name__ == "__main__":
    unittest.main()
