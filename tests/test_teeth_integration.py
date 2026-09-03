import unittest

from app.teeth_integration import attach_teeth_narration_facts


class TeethIntegrationTests(unittest.TestCase):
    def test_attaches_approved_active_teeth_facts_and_prioritizes_live_lead(self):
        original = {
            "event_context": {
                "primary_action": {"name": "fell out", "meaning": "generic loss"},
                "primary_subject": "Teeth",
                "primary_ending": {},
            },
            "top_symbols": ["Teeth"],
            "risk": "Low",
        }

        merged = attach_teeth_narration_facts(
            "One of my front teeth fell out with pain.",
            original,
        )

        self.assertEqual(merged["top_symbols"], ["Teeth"])
        self.assertEqual(merged["event_context"]["primary_action"], {})
        self.assertEqual(merged["event_context"]["primary_subject"], "Teeth")
        self.assertEqual(merged["risk"], "")
        self.assertIn("one person", merged["lead_message"].lower())
        self.assertIn("relative or close friend", merged["lead_message"].lower())
        self.assertIn("very close", merged["lead_message"].lower())

        teeth = merged["teeth_narration"]
        self.assertTrue(teeth["active"])
        self.assertEqual(teeth["warning_count"], "one_person")
        self.assertEqual(teeth["relationship_scope"], "relative_or_close_friend")
        self.assertEqual(teeth["proximity"], "very_close_or_close_relative")
        user_text = " ".join([teeth["lead"], *teeth["details"]]).lower()
        self.assertNotIn("blood", user_text)
        self.assertNotIn("older", str(teeth).lower())
        self.assertNotIn("younger", str(teeth).lower())

    def test_negated_tooth_loss_remains_inactive_and_does_not_rewrite_generic_facts(self):
        original = {"top_symbols": ["Teeth"], "risk": "Low"}
        merged = attach_teeth_narration_facts(
            "My tooth did not fall out and it hurt.",
            original,
        )

        teeth = merged["teeth_narration"]
        self.assertFalse(teeth["active"])
        self.assertEqual(teeth["lead"], "")
        self.assertEqual(teeth["details"], [])
        self.assertEqual(merged["risk"], "Low")
        self.assertNotIn("lead_message", merged)

    def test_multiple_painless_teeth_preserve_approved_distinction(self):
        merged = attach_teeth_narration_facts(
            "Several of my lower teeth fell out without pain.",
            {},
        )

        teeth = merged["teeth_narration"]
        self.assertTrue(teeth["active"])
        self.assertEqual(teeth["warning_count"], "multiple_people")
        self.assertEqual(teeth["proximity"], "friend_acquaintance_or_more_distant")
        self.assertIn("multiple people", merged["lead_message"].lower())
        self.assertIn("friend, acquaintance", merged["lead_message"].lower())
        self.assertNotIn("child", str(teeth).lower())
        self.assertNotIn("parent", str(teeth).lower())

    def test_fabricated_teeth_ending_is_removed_but_real_ending_is_preserved(self):
        fake = attach_teeth_narration_facts(
            "Three of my teeth fell out.",
            {
                "event_context": {
                    "primary_action": {"name": "fell out"},
                    "primary_ending": {"name": "teeth", "meaning": "incorrect ending"},
                }
            },
        )
        self.assertEqual(fake["event_context"]["primary_ending"], {})

        real = attach_teeth_narration_facts(
            "Three of my teeth fell out and later new healthy teeth grew in.",
            {
                "event_context": {
                    "primary_action": {"name": "fell out"},
                    "primary_ending": {"name": "healthy growth", "meaning": "restoration"},
                }
            },
        )
        self.assertEqual(real["event_context"]["primary_ending"]["name"], "healthy growth")


if __name__ == "__main__":
    unittest.main()
