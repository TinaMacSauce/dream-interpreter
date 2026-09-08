import unittest

from app.snake_doctrine import (
    build_snake_doctrine_context,
    build_snake_narration_facts,
)


class SnakeFounderClarificationDecision02Tests(unittest.TestCase):
    def test_027_watching_is_target_scoped_without_completed_attack(self):
        doctrine = build_snake_doctrine_context(
            "A snake watched my sister from across the room."
        )

        self.assertTrue(doctrine["active_doctrine"])
        self.assertEqual("watching", doctrine["action"])
        self.assertEqual("sister", doctrine["action_target"])
        self.assertIn("SNAKE-WATCHING-TARGET", doctrine["applied_rule_ids"])
        self.assertNotIn("SNAKE-ATTACK", doctrine["applied_rule_ids"])
        self.assertFalse(doctrine["completed_bite"])

    def test_028_attempted_bite_preserves_cousin_and_failed_contact(self):
        doctrine = build_snake_doctrine_context(
            "The snake tried to bite my cousin but never touched her."
        )

        self.assertEqual("attempted_bite", doctrine["action"])
        self.assertEqual("cousin", doctrine["action_target"])
        self.assertTrue(doctrine["attempted_bite"])
        self.assertFalse(doctrine["completed_bite"])
        self.assertFalse(doctrine["venom"])
        self.assertEqual("not_established", doctrine["outcome"])
        self.assertIn("SNAKE-BITE-ATTEMPT-TARGET", doctrine["applied_rule_ids"])
        self.assertNotIn("SNAKE-BITE", doctrine["applied_rule_ids"])
        self.assertNotIn("SNAKE-END-DEFEAT", doctrine["applied_rule_ids"])

    def test_029_carving_is_bounded_and_not_a_live_snake_event(self):
        doctrine = build_snake_doctrine_context(
            "No snake chased, bit, or attacked me. I only saw a carving of one."
        )
        narration = build_snake_narration_facts(
            "No snake chased, bit, or attacked me. I only saw a carving of one."
        )["narration_text"].lower()

        self.assertTrue(doctrine["active_doctrine"])
        self.assertEqual("carving", doctrine["representation_type"])
        self.assertEqual("representation", doctrine["entity_form"])
        self.assertEqual("lurking_opposition_warning", doctrine["base_meaning"])
        self.assertEqual(["SNAKE-REP-CARVING"], doctrine["applied_rule_ids"])
        self.assertEqual([], doctrine["event_inventory"])
        self.assertIn("not a live-snake event", narration)
        for forbidden in ("hidden person is", "surveillance is", "supernatural cause is"):
            self.assertNotIn(forbidden, narration)

    def test_030_bedroom_maps_to_intimacy_without_partner_accusation(self):
        dream = "A snake was in my bedroom, therefore my partner cursed me."
        doctrine = build_snake_doctrine_context(dream)
        narration = build_snake_narration_facts(dream)["narration_text"].lower()

        self.assertEqual("bedroom", doctrine["location_observed"])
        self.assertEqual("intimate_life_sphere", doctrine["location_scope"])
        self.assertIn("SNAKE-LOC-BEDROOM", doctrine["applied_rule_ids"])
        self.assertNotIn("partner cursed", narration)
        self.assertIn("without identifying a partner or culprit", narration)

    def test_031_house_maps_to_dreamers_life_without_invented_attack(self):
        doctrine = build_snake_doctrine_context(
            "A snake moved through my house but never approached anyone."
        )

        self.assertEqual("house", doctrine["location_observed"])
        self.assertEqual("life_sphere", doctrine["location_scope"])
        self.assertIn("SNAKE-LOC-HOUSE", doctrine["applied_rule_ids"])
        self.assertNotIn("SNAKE-ATTACK", doctrine["applied_rule_ids"])
        self.assertEqual("", doctrine["action_target"])

    def test_032_kitchen_uses_only_approved_scope(self):
        dream = "A snake was hiding in my kitchen."
        doctrine = build_snake_doctrine_context(dream)
        narration = build_snake_narration_facts(dream)["narration_text"].lower()

        self.assertEqual("kitchen", doctrine["location_observed"])
        self.assertEqual(
            "productivity_healing_replenishment_sphere",
            doctrine["location_scope"],
        )
        self.assertIn("SNAKE-LOC-KITCHEN", doctrine["applied_rule_ids"])
        self.assertIn("without implying contamination, illness, or a culprit", narration)

    def test_033_bathroom_meaning_remains_unresolved_and_inactive(self):
        doctrine = build_snake_doctrine_context(
            "A snake was in my bathroom and did not attack."
        )

        self.assertTrue(doctrine["active_doctrine"])
        self.assertEqual("bathroom", doctrine["location_observed"])
        self.assertEqual("", doctrine["location_scope"])
        self.assertNotIn("SNAKE-LOCATION", doctrine["applied_rule_ids"])
        self.assertNotIn("SNAKE-LOC-BATHROOM", doctrine["applied_rule_ids"])
        self.assertNotIn("SNAKE-ATTACK", doctrine["applied_rule_ids"])

    def test_unresolved_registry_rows_can_never_activate(self):
        unresolved = {
            "SNAKE-LOC-BATHROOM",
            "SNAKE-LOC-LIVING-AREA",
            "SNAKE-REP-NONCARVING",
            "SNAKE-OWNERSHIP-CONTROL",
            "SNAKE-FAITH-ELIGIBILITY-EXTENDED",
        }
        dreams = (
            "A snake was in my bathroom and did not attack.",
            "A snake was in my living room.",
            "I saw a painting of a snake.",
            "I held a pet snake under unusual control.",
        )
        for dream in dreams:
            with self.subTest(dream=dream):
                rules = set(build_snake_doctrine_context(dream)["applied_rule_ids"])
                self.assertFalse(rules & unresolved)


if __name__ == "__main__":
    unittest.main()
