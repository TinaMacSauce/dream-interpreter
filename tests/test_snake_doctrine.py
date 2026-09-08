import unittest

from app.snake_doctrine import build_snake_doctrine_context, build_snake_narration_facts
from app.snake_integration import bind_snake_output_contract


class SnakeDoctrineTests(unittest.TestCase):
    def doctrine(self, dream):
        return build_snake_doctrine_context(dream)

    def test_base_presence_and_negation(self):
        active = self.doctrine("I saw a snake in the dream.")
        negated = self.doctrine("I did not see any snake.")
        self.assertTrue(active["active_doctrine"])
        self.assertEqual("enemy_or_opposition", active["base_meaning"])
        self.assertIn("SNAKE-BASE-ENEMY", active["applied_rule_ids"])
        self.assertFalse(negated["active_doctrine"])
        self.assertEqual([], negated["applied_rule_ids"])

    def test_action_target_and_noncompletion_are_preserved(self):
        watching = self.doctrine("A snake watched my sister.")
        attempted = self.doctrine("The snake tried to bite me but did not.")
        attack = self.doctrine("A snake attacked me, but the dream ended before the fight was over.")
        self.assertEqual("watching", watching["action"])
        self.assertEqual("sister", watching["action_target"])
        self.assertIn("SNAKE-WATCHING", watching["applied_rule_ids"])
        self.assertEqual("attempted_bite", attempted["action"])
        self.assertFalse(attempted["completed_bite"])
        self.assertNotIn("SNAKE-BITE", attempted["applied_rule_ids"])
        self.assertEqual("unresolved", attack["outcome"])
        self.assertNotIn("SNAKE-END-DEFEAT", attack["applied_rule_ids"])

    def test_genuine_outcomes_and_found_dead_distinction(self):
        victory = self.doctrine("I fought the snake and killed it.")
        found = self.doctrine("I found a dead snake beside the road.")
        defeat = self.doctrine("The snake defeated me at the end.")
        bitten = self.doctrine("The snake bit me on the hand.")
        self.assertEqual("dreamer_victory", victory["outcome"])
        self.assertIn("SNAKE-END-VICTORY", victory["applied_rule_ids"])
        self.assertEqual("not_established", found["outcome"])
        self.assertNotIn("SNAKE-END-VICTORY", found["applied_rule_ids"])
        self.assertEqual("opposition_victory_in_encounter", defeat["outcome"])
        self.assertEqual("opposition_victory_in_encounter", bitten["outcome"])
        self.assertIn("SNAKE-BITE", bitten["applied_rule_ids"])

    def test_modifiers_scope_without_culprit_or_certainty(self):
        doctrine = self.doctrine(
            "Three huge red snakes appeared in my office at work. My neighbor owned them. "
            "One snake transformed into a person."
        )
        self.assertEqual("multiple", doctrine["quantity"])
        self.assertEqual("stronger_or_more_dangerous", doctrine["strength"])
        self.assertEqual("work_sphere", doctrine["location_scope"])
        self.assertTrue(doctrine["transformed_into_person"])
        self.assertEqual("low", doctrine["ownership_weight"])
        self.assertEqual(["red"], doctrine["colors_ignored"])
        narration = build_snake_narration_facts(
            "A huge red snake transformed into a person in my office at work."
        )["narration_text"].lower()
        self.assertIn("does not prove", narration)
        self.assertIn("without identifying a culprit", narration)
        self.assertIn("color is deliberately excluded", narration)

    def test_venom_is_explicit_and_never_medical_evidence(self):
        venom = self.doctrine("The snake bit me and venom entered my arm.")
        bite = self.doctrine("The snake bit me.")
        self.assertTrue(venom["venom"])
        self.assertIn("SNAKE-VENOM", venom["applied_rule_ids"])
        self.assertFalse(bite["venom"])
        narration = build_snake_narration_facts(
            "The snake bit me and venom entered my arm."
        )["narration_text"].lower()
        self.assertIn("not medical evidence", narration)

    def test_faith_guidance_is_separate_and_non_guaranteeing(self):
        neutral = self.doctrine("A snake watched me.")
        self.assertNotIn("SNAKE-FAITH-RESPONSE", neutral["applied_rule_ids"])
        self.assertNotIn("SNAKE-FAITH-BEST-PRACTICE", neutral["applied_rule_ids"])

        dream = "A snake attacked me and I woke before either of us won."
        doctrine = self.doctrine(dream)
        self.assertIn("SNAKE-FAITH-RESPONSE", doctrine["applied_rule_ids"])
        self.assertIn("SNAKE-FAITH-BEST-PRACTICE", doctrine["applied_rule_ids"])
        _, interpretation, full = bind_snake_output_contract(
            doctrine_facts={"snake_narration": build_snake_narration_facts(dream)},
            seal={"risk": "high"}, interpretation={}, full_interpretation="",
        )
        self.assertIn("Psalm 91", interpretation["what_to_do"])
        self.assertIn("not scientifically proven", interpretation["what_to_do"])
        self.assertIn("not proof of an enemy", full)

    def test_internal_dreamer_target_is_never_exposed_in_narration(self):
        narration = build_snake_narration_facts(
            "A snake attacked me, but the dream ended before either of us won."
        )["narration_text"]
        self.assertIn("directed toward you", narration)
        self.assertNotIn("directed toward dreamer", narration)

    def test_mixed_teeth_snake_output_does_not_invent_precedence(self):
        snake = build_snake_narration_facts("A snake watched me.")
        seal, interpretation, full = bind_snake_output_contract(
            doctrine_facts={"snake_narration": snake, "teeth_narration": {"active": True}},
            seal={"risk": "existing"}, interpretation={"spiritual_meaning": "teeth"},
            full_interpretation="teeth full",
        )
        self.assertEqual("existing", seal["risk"])
        self.assertEqual("teeth", interpretation["spiritual_meaning"])
        self.assertEqual("teeth full", full)


if __name__ == "__main__":
    unittest.main()
