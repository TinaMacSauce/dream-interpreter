import unittest

from app.teeth_integration import attach_teeth_narration_facts


class TeethIntegrationTests(unittest.TestCase):
    def test_attaches_approved_active_teeth_facts_without_losing_existing_doctrine(self):
        original = {"event_context": {"primary_subject": "Teeth"}, "top_symbols": ["Teeth"]}

        merged = attach_teeth_narration_facts(
            "One of my front teeth fell out with pain.",
            original,
        )

        self.assertEqual(merged["event_context"], original["event_context"])
        self.assertEqual(merged["top_symbols"], ["Teeth"])

        teeth = merged["teeth_narration"]
        self.assertTrue(teeth["active"])
        self.assertEqual(teeth["warning_count"], "one_person")
        self.assertEqual(teeth["relationship_scope"], "relative_or_close_friend")
        self.assertEqual(teeth["proximity"], "very_close_or_close_relative")
        self.assertNotIn("blood", str(teeth).lower())
        self.assertNotIn("older", str(teeth).lower())
        self.assertNotIn("younger", str(teeth).lower())

    def test_negated_tooth_loss_remains_inactive(self):
        merged = attach_teeth_narration_facts(
            "My tooth did not fall out and it hurt.",
            {"top_symbols": ["Teeth"]},
        )

        teeth = merged["teeth_narration"]
        self.assertFalse(teeth["active"])
        self.assertEqual(teeth["lead"], "")
        self.assertEqual(teeth["details"], [])

    def test_multiple_painless_teeth_preserve_approved_distinction(self):
        merged = attach_teeth_narration_facts(
            "Several of my lower teeth fell out without pain.",
            {},
        )

        teeth = merged["teeth_narration"]
        self.assertTrue(teeth["active"])
        self.assertEqual(teeth["warning_count"], "multiple_people")
        self.assertEqual(teeth["proximity"], "friend_acquaintance_or_more_distant")
        self.assertNotIn("child", str(teeth).lower())
        self.assertNotIn("parent", str(teeth).lower())


if __name__ == "__main__":
    unittest.main()
