import unittest

from app.routes.qa import SNAKE_QA_CASES
from app.snake_doctrine import build_snake_doctrine_context, build_snake_narration_facts


EXPECTED = {
    "REG-SNAKE-ATTACK-001": {
        "fields": {"action": "attack", "action_target": "dreamer", "outcome": "unresolved", "unfinished_battle": True},
        "include": {"SNAKE-ATTACK", "SNAKE-UNFINISHED-BATTLE"},
        "exclude": {"SNAKE-END-DEFEAT"},
        "events": [("attack", "chain-snake-1", "dreamer")],
        "frontiers": [("chain-snake-1", "unresolved")],
    },
    "REG-SNAKE-BITE-ATTEMPT-001": {
        "fields": {"attempted_bite": True, "completed_bite": False, "outcome": "not_established"},
        "exclude": {"SNAKE-BITE", "SNAKE-END-DEFEAT"},
        "events": [("attempted_bite", "chain-snake-1", "dreamer")],
        "frontiers": [("chain-snake-1", "no_completed_conflict")],
    },
    "REG-SNAKE-BITE-DREAMER-001": {
        "fields": {"action_target": "dreamer", "completed_bite": True, "outcome": "opposition_victory_in_encounter"},
        "include": {"SNAKE-BITE", "SNAKE-END-DEFEAT"},
        "events": [("bite", "chain-snake-1", "dreamer-hand")],
        "frontiers": [("chain-snake-1", "defeat")],
    },
    "REG-SNAKE-CHASE-ESCAPE-001": {
        "fields": {"completed_bite": False, "outcome": "not_established"},
        "exclude": {"SNAKE-BITE", "SNAKE-END-DEFEAT"},
        "events": [("chase", "chain-snake-1", "dreamer"), ("escape", "chain-snake-1", "snake-1")],
        "frontiers": [("chain-snake-1", "escaped_without_decisive_outcome")],
    },
    "REG-SNAKE-CHASE-CAPTURE-001": {
        "fields": {"completed_bite": False, "outcome": "unresolved"},
        "exclude": {"SNAKE-BITE", "SNAKE-END-DEFEAT"},
        "events": [("chase", "chain-snake-1", "dreamer"), ("capture", "chain-snake-1", "dreamer"), ("bite", "chain-snake-1", "dreamer")],
        "frontiers": [("chain-snake-1", "unresolved")],
    },
    "REG-SNAKE-DEFEAT-001": {
        "fields": {"outcome": "opposition_victory_in_encounter"},
        "include": {"SNAKE-END-DEFEAT"},
        "events": [("overpower", "chain-snake-1", "dreamer")],
        "frontiers": [("chain-snake-1", "defeat")],
    },
    "REG-SNAKE-HYPOTHETICAL-001": {
        "fields": {"completed_bite": False, "outcome": "not_established"},
        "exclude": {"SNAKE-BITE", "SNAKE-END-DEFEAT"},
        "events": [("bite", "chain-snake-1", "dreamer")],
        "frontiers": [],
    },
    "REG-SNAKE-MULTI-ACTION-001": {
        "fields": {"quantity": "multiple", "outcome": "unresolved"},
        "include": {"SNAKE-WATCHING", "SNAKE-ATTACK", "SNAKE-QUANTITY"},
        "events": [("watch", "chain-snake-1", "dreamer"), ("attack", "chain-snake-2", None)],
        "frontiers": [("chain-snake-1", "no_completed_conflict"), ("chain-snake-2", "unresolved")],
    },
    "REG-SNAKE-MIXED-ENDINGS-001": {
        "fields": {"quantity": "multiple", "completed_bite": True, "outcome": "mixed"},
        "include": {"SNAKE-BITE", "SNAKE-END-VICTORY", "SNAKE-END-DEFEAT", "SNAKE-RETREAT"},
        "events": [("kill_by_dreamer", "chain-snake-1", "snake-1"), ("bite", "chain-snake-2", "dreamer"), ("retreat", "chain-snake-3", "dreamer")],
        "frontiers": [("chain-snake-1", "victory"), ("chain-snake-2", "defeat"), ("chain-snake-3", "retreat")],
    },
    "REG-SNAKE-NEGATION-001": {
        "fields": {"action": "", "attempted_bite": False, "completed_bite": False, "outcome": "not_established"},
        "exclude": {"SNAKE-BITE", "SNAKE-ATTACK", "SNAKE-END-DEFEAT"},
        "events": [("attack", "chain-snake-1", "dreamer"), ("bite", "chain-snake-1", "dreamer")],
        "frontiers": [],
    },
    "REG-SNAKE-PROTECT-OTHER-001": {
        "fields": {"action_target": "child", "completed_bite": False, "outcome": "dreamer_victory"},
        "include": {"SNAKE-ATTACK", "SNAKE-END-VICTORY"},
        "exclude": {"SNAKE-BITE", "SNAKE-END-DEFEAT"},
        "events": [("attack", "chain-snake-1", "child"), ("kill_by_dreamer", "chain-snake-1", "snake-1")],
        "frontiers": [("chain-snake-1", "victory")],
    },
    "REG-SNAKE-SIZE-SPECIES-001": {
        "fields": {"quantity": "multiple", "strength": "stronger_or_more_dangerous"},
        "include": {"SNAKE-QUANTITY", "SNAKE-SIZE-DANGER"},
        "events": [("presence", "chain-snake-1", None), ("presence", "chain-snake-2", None)],
        "frontiers": [("chain-snake-1", "no_completed_conflict"), ("chain-snake-2", "no_completed_conflict")],
    },
    "REG-SNAKE-TRANSFORM-001": {
        "fields": {"transformed_into_person": True},
        "include": {"SNAKE-TRANSFORM-PERSON"},
        "events": [("transform_to_person", "chain-snake-1", "sister")],
        "frontiers": [("chain-snake-1", "no_completed_conflict")],
    },
    "REG-SNAKE-TRANSFORM-ACCUSATION-001": {
        "fields": {"transformed_into_person": True},
        "include": {"SNAKE-TRANSFORM-PERSON"},
        "events": [("transform_to_person", "chain-snake-1", "coworker")],
        "frontiers": [("chain-snake-1", "no_completed_conflict")],
    },
    "REG-SNAKE-VENOM-ABSENT-001": {
        "fields": {"completed_bite": True, "venom": False},
        "exclude": {"SNAKE-VENOM"},
        "events": [("bite", "chain-snake-1", "dreamer")],
        "frontiers": [("chain-snake-1", "defeat")],
    },
    "REG-SNAKE-QUOTED-001": {
        "fields": {"active_doctrine": False, "action": "", "completed_bite": False},
        "exact": [],
        "events": [],
        "frontiers": [],
    },
}


class SnakeOrdinaryLanguageRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dreams = dict(SNAKE_QA_CASES)

    def test_all_16_qa_regressions_preserve_meaning_and_event_lineage(self):
        self.assertEqual(16, len(EXPECTED))
        for case_id, expected in EXPECTED.items():
            doctrine = build_snake_doctrine_context(self.dreams[case_id])
            rules = set(doctrine["applied_rule_ids"])
            graph = doctrine["event_graph"]
            with self.subTest(case_id=case_id):
                for field, value in expected.get("fields", {}).items():
                    self.assertEqual(value, doctrine[field])
                self.assertTrue(expected.get("include", set()).issubset(rules))
                self.assertTrue(rules.isdisjoint(expected.get("exclude", set())))
                if "exact" in expected:
                    self.assertEqual(expected["exact"], doctrine["applied_rule_ids"])
                self.assertEqual(
                    expected["events"],
                    [(event["action"], event["chain_id"], event["target_id"]) for event in graph["events"]],
                )
                self.assertEqual(
                    expected["frontiers"],
                    [(frontier["chain_id"], frontier["outcome"]) for frontier in graph["terminal_frontiers"]],
                )
                self.assertEqual({"verified": True, "reason_codes": []}, graph["graph_integrity"])

    def test_hypothetical_negated_and_quoted_bites_cannot_leak_into_doctrine(self):
        for case_id in ("REG-SNAKE-HYPOTHETICAL-001", "REG-SNAKE-NEGATION-001", "REG-SNAKE-QUOTED-001"):
            doctrine = build_snake_doctrine_context(self.dreams[case_id])
            with self.subTest(case_id=case_id):
                self.assertFalse(doctrine["completed_bite"])
                self.assertNotIn("SNAKE-BITE", doctrine["applied_rule_ids"])
                self.assertNotIn("SNAKE-END-DEFEAT", doctrine["applied_rule_ids"])

    def test_transformation_never_turns_a_named_person_into_a_culprit(self):
        for case_id in ("REG-SNAKE-TRANSFORM-001", "REG-SNAKE-TRANSFORM-ACCUSATION-001"):
            narration = build_snake_narration_facts(self.dreams[case_id])["narration_text"].lower()
            with self.subTest(case_id=case_id):
                self.assertIn("does not prove", narration)
                self.assertNotIn("is my enemy", narration)

    def test_neutral_observation_does_not_receive_troubling_dream_faith_guidance(self):
        doctrine = build_snake_doctrine_context("A snake watched me from a distance.")
        self.assertNotIn("SNAKE-FAITH-RESPONSE", doctrine["applied_rule_ids"])
        self.assertNotIn("SNAKE-FAITH-BEST-PRACTICE", doctrine["applied_rule_ids"])
        self.assertEqual("", doctrine["response_guidance"])
        self.assertEqual("", doctrine["best_practice_guidance"])


if __name__ == "__main__":
    unittest.main()
