import copy
import unittest

from app.routes.qa import SNAKE_QA_CASES
from app.snake_event_graph import (
    SNAKE_EVENT_CONTRACT_VERSION,
    extract_snake_event_graph,
    validate_snake_event_graph,
)


EXPECTED = {
    "SNAKE-002-EVENT-WATCH-001": (["watch"], ["dreamer"], ["no_completed_conflict"]),
    "SNAKE-002-EVENT-ATTEMPT-BITE-001": (["attack", "bite", "contact"], ["dreamer", "dreamer-hand", "dreamer"], ["unresolved"]),
    "SNAKE-002-EVENT-BITE-DREAMER-001": (["bite"], ["dreamer-ankle"], ["defeat"]),
    "SNAKE-002-TARGET-THIRD-PARTY-001": (["bite"], ["sister-wrist"], ["opposition_prevailed_for_target"]),
    "SNAKE-003-VENOM-SCOPE-001": (["bite", "venom_entry"], ["dreamer-arm", "dreamer-arm"], ["defeat"]),
    "SNAKE-002-PROTECTION-BLOCK-001": (["attack", "protect_block"], ["dreamer", "dreamer"], ["no_completed_conflict"]),
    "SNAKE-002-MULTI-MIXED-001": (["watch", "attack", "kill_by_dreamer", "retreat"], ["dreamer", "dreamer", "snake-2", "dreamer"], ["no_completed_conflict", "victory", "retreat"]),
    "SNAKE-002-NEGATION-001": (["bite", "watch"], ["dreamer", "dreamer"], ["no_completed_conflict"]),
    "SNAKE-002-HYPOTHETICAL-001": (["bite", "retreat"], ["dreamer", "dreamer"], ["retreat"]),
    "SNAKE-002-TARGET-AMBIGUOUS-001": (["attack", "bite"], [None, None], ["unresolved"]),
    "SNAKE-003-TRANSFORM-SAFETY-001": (["transform_to_person"], ["friend"], ["no_completed_conflict"]),
    "SNAKE-002-LOCATION-MULTI-SPHERE-001": (["watch", "attack"], ["dreamer", "dreamer"], ["no_completed_conflict", "unresolved"]),
    "SNAKE-002-COLOR-INVARIANT-BLACK-001": (["attack", "retreat"], ["dreamer", "dreamer"], ["retreat"]),
    "SNAKE-002-COLOR-INVARIANT-GREEN-001": (["attack", "retreat"], ["dreamer", "dreamer"], ["retreat"]),
    "SNAKE-002-ENDING-ALREADY-DEAD-001": (["discover_already_dead"], ["snake-1"], ["no_completed_conflict"]),
    "SNAKE-002-ENDING-ESCAPE-001": (["chase", "escape"], ["dreamer", "snake-1"], ["escaped_without_decisive_outcome"]),
    "SNAKE-002-RECURRENCE-UNFINISHED-001": (["battle"], ["snake-1"], ["unresolved"]),
    "SNAKE-002-OWNERSHIP-LOW-001": (["watch"], ["dreamer"], ["no_completed_conflict"]),
    "SNAKE-003-FAITH-SEPARATION-001": (["attack"], ["dreamer"], ["unresolved"]),
    "SNAKE-002-TARGET-LINEAGE-SELF-HAND-001": (["bite"], ["dreamer-hand"], ["defeat"]),
    "SNAKE-002-TARGET-LINEAGE-THIRD-PARTY-WRIST-001": (["bite"], ["sister-wrist"], ["opposition_prevailed_for_target"]),
    "SNAKE-002-TARGET-LINEAGE-COREFERENCE-IT-001": (["bite"], ["sister-hand"], ["opposition_prevailed_for_target"]),
    "SNAKE-002-TARGET-LINEAGE-AMBIGUOUS-001": (["bite"], [None], ["unresolved"]),
    "SNAKE-002-TARGET-LINEAGE-MULTI-PERSON-001": (["bite", "bite"], ["dreamer-hand", "sister-wrist"], ["defeat", "opposition_prevailed_for_target"]),
    "SNAKE-002-TARGET-LINEAGE-SEQUENCE-001": (["attack", "bite"], ["dreamer", "sister-hand"], ["opposition_prevailed_for_target"]),
    "SNAKE-003-TARGET-LINEAGE-VENOM-THIRD-PARTY-001": (["bite", "venom_entry"], ["brother-arm", "brother-arm"], ["opposition_prevailed_for_target"]),
    "SNAKE-002-TARGET-LINEAGE-ATTEMPT-001": (["attempted_bite"], ["dreamer-ankle"], ["no_completed_conflict"]),
    "SNAKE-002-TARGET-LINEAGE-PROTECTION-001": (["attempted_bite", "block"], ["child", "snake-1"], ["no_completed_conflict"]),
    "SNAKE-002-TARGET-LINEAGE-NONPERSON-001": (["bite"], ["travel-bag"], ["no_completed_conflict"]),
    "SNAKE-002-TARGET-LINEAGE-BITE-THEN-VICTORY-001": (["bite", "kill_by_dreamer"], ["dreamer-hand", "snake-1"], ["victory"]),
    "SNAKE-002-TARGET-LINEAGE-NEGATED-CORRECTION-001": (["bite", "bite"], ["sister", "dreamer-hand"], ["defeat"]),
}

LINEAGE_EXPECTED = {
    "SNAKE-002-TARGET-LINEAGE-SELF-HAND-001": [("dreamer-hand", "dreamer", "possessive_body_part_owner", True)],
    "SNAKE-002-TARGET-LINEAGE-THIRD-PARTY-WRIST-001": [("sister-wrist", "sister", "possessive_body_part_owner", True)],
    "SNAKE-002-TARGET-LINEAGE-COREFERENCE-IT-001": [("sister-hand", "sister", "singular_coreference_to_owned_body_part", True)],
    "SNAKE-002-TARGET-LINEAGE-AMBIGUOUS-001": [(None, None, "plural_coreference_ambiguous", False)],
    "SNAKE-002-TARGET-LINEAGE-MULTI-PERSON-001": [("dreamer-hand", "dreamer", "possessive_body_part_owner", True), ("sister-wrist", "sister", "possessive_body_part_owner", True)],
    "SNAKE-002-TARGET-LINEAGE-SEQUENCE-001": [("dreamer", "dreamer", "direct_person_object", True), ("sister-hand", "sister", "possessive_body_part_owner", True)],
    "SNAKE-003-TARGET-LINEAGE-VENOM-THIRD-PARTY-001": [("brother-arm", "brother", "possessive_body_part_owner", True), ("brother-arm", "brother", "pronoun_to_same_chain_body_part", True)],
    "SNAKE-002-TARGET-LINEAGE-ATTEMPT-001": [("dreamer-ankle", "dreamer", "possessive_body_part_owner", False)],
    "SNAKE-002-TARGET-LINEAGE-PROTECTION-001": [("child", "child", "direct_person_object", False)],
    "SNAKE-002-TARGET-LINEAGE-NONPERSON-001": [("travel-bag", None, "explicit_nonperson_target", False)],
    "SNAKE-002-TARGET-LINEAGE-BITE-THEN-VICTORY-001": [("dreamer-hand", "dreamer", "possessive_body_part_owner", True)],
    "SNAKE-002-TARGET-LINEAGE-NEGATED-CORRECTION-001": [("sister", "sister", "direct_person_object", False), ("dreamer-hand", "dreamer", "possessive_body_part_owner", True)],
}


class SnakeContextEventGraphTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dreams = dict(SNAKE_QA_CASES)

    def test_all_31_context_fixtures_match_event_and_terminal_signatures(self):
        self.assertEqual(31, len(EXPECTED))
        for case_id, (actions, targets, outcomes) in EXPECTED.items():
            with self.subTest(case_id=case_id):
                graph = extract_snake_event_graph(self.dreams[case_id])
                self.assertEqual(SNAKE_EVENT_CONTRACT_VERSION, graph["contract_version"])
                self.assertEqual(actions, [event["action"] for event in graph["events"]])
                self.assertEqual(targets, [event["target_id"] for event in graph["events"]])
                self.assertEqual(outcomes, [frontier["outcome"] for frontier in graph["terminal_frontiers"]])
                self.assertEqual({"verified": True, "reason_codes": []}, graph["graph_integrity"])

    def test_target_lineage_firewall_matches_all_12_oracle_fixtures(self):
        self.assertEqual(12, len(LINEAGE_EXPECTED))
        for case_id, expected in LINEAGE_EXPECTED.items():
            graph = extract_snake_event_graph(self.dreams[case_id])
            actual = [
                (item["surface_target_id"], item["affected_person_id"], item["resolution_basis"], item["outcome_eligible"])
                for item in graph["target_lineage"]
                if item["resolution_basis"] != "direct_snake_object"
            ]
            with self.subTest(case_id=case_id):
                self.assertEqual(expected, actual)

    def test_references_and_source_spans_are_exact_and_resolved(self):
        required = {"event_id", "action", "actor_id", "target_id", "target_status", "scene_id", "chain_id",
                    "polarity", "modality", "actuality", "completion", "terminal", "span_text"}
        for case_id in EXPECTED:
            dream = self.dreams[case_id].lower()
            graph = extract_snake_event_graph(dream)
            events = {event["event_id"]: event for event in graph["events"]}
            with self.subTest(case_id=case_id):
                self.assertTrue(all(required.issubset(event) for event in events.values()))
                self.assertTrue(all(event["span_text"].lower() in dream for event in events.values()))
                self.assertTrue(all(item["event_id"] in events for item in graph["target_lineage"]))
                self.assertTrue(all(item["decisive_event_id"] in events for item in graph["terminal_frontiers"]))

    def test_unsafe_target_and_completion_mutations_fail_closed(self):
        graph = extract_snake_event_graph(self.dreams["SNAKE-002-TARGET-LINEAGE-SELF-HAND-001"])
        broken = copy.deepcopy(graph)
        broken["target_lineage"][0]["resolution_path"] = ["dreamer"]
        self.assertIn("BODY_PART_OWNER_LINEAGE_MISMATCH", validate_snake_event_graph(broken)["reason_codes"])

        graph = extract_snake_event_graph(self.dreams["SNAKE-002-TARGET-LINEAGE-AMBIGUOUS-001"])
        broken = copy.deepcopy(graph)
        broken["target_lineage"][0]["outcome_eligible"] = True
        self.assertIn("AMBIGUOUS_TARGET_NOT_FORCED", validate_snake_event_graph(broken)["reason_codes"])

        graph = extract_snake_event_graph(self.dreams["SNAKE-002-TARGET-LINEAGE-ATTEMPT-001"])
        broken = copy.deepcopy(graph)
        broken["target_lineage"][0]["outcome_eligible"] = True
        self.assertIn("ATTEMPT_NOT_COMPLETED_CONTACT", validate_snake_event_graph(broken)["reason_codes"])

        graph = extract_snake_event_graph(self.dreams["SNAKE-002-TARGET-LINEAGE-NONPERSON-001"])
        broken = copy.deepcopy(graph)
        broken["target_lineage"][0]["outcome_eligible"] = True
        self.assertIn("NONPERSON_TARGET_NOT_PERSON_OUTCOME", validate_snake_event_graph(broken)["reason_codes"])

        graph = extract_snake_event_graph(self.dreams["SNAKE-002-TARGET-LINEAGE-NEGATED-CORRECTION-001"])
        broken = copy.deepcopy(graph)
        broken["target_lineage"][0]["outcome_eligible"] = True
        self.assertIn("NEGATED_TARGET_LINEAGE_NOT_RELEASED", validate_snake_event_graph(broken)["reason_codes"])

    def test_color_exclusion_does_not_change_event_arbitration(self):
        black = extract_snake_event_graph(self.dreams["SNAKE-002-COLOR-INVARIANT-BLACK-001"])
        green = extract_snake_event_graph(self.dreams["SNAKE-002-COLOR-INVARIANT-GREEN-001"])
        signature = lambda graph: (
            [(event["action"], event["target_id"], event["completion"], event["terminal"]) for event in graph["events"]],
            graph["terminal_frontiers"],
        )
        self.assertEqual(signature(black), signature(green))


if __name__ == "__main__":
    unittest.main()
