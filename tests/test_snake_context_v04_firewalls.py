import copy
import unittest

from app.routes.qa import SNAKE_QA_CASES
from app.snake_event_graph import extract_snake_event_graph, validate_snake_event_graph


PARTITION_COUNTS = {
    "SNAKE-002-PARTITION-TWO-DISTINCT-ACTIONS-001": (2, 3, 2, 0),
    "SNAKE-002-PARTITION-THREE-MIXED-ENDINGS-001": (3, 4, 3, 3),
    "SNAKE-002-PARTITION-SAME-SNAKE-SEQUENCE-001": (3, 2, 1, 1),
    "SNAKE-002-PARTITION-GROUP-SHARED-ACTION-001": (3, 1, 3, 0),
    "SNAKE-002-PARTITION-MODIFIER-SCOPE-001": (2, 2, 2, 0),
    "SNAKE-002-PARTITION-LOCATION-SCOPE-001": (2, 2, 2, 0),
    "SNAKE-002-PARTITION-TARGET-SCOPE-001": (2, 2, 2, 1),
    "SNAKE-002-PARTITION-PRONOUN-RESOLVED-001": (2, 3, 2, 1),
    "SNAKE-002-PARTITION-PRONOUN-AMBIGUOUS-001": (1, 2, 3, 0),
    "SNAKE-002-PARTITION-NEGATED-MEMBER-001": (2, 3, 2, 1),
    "SNAKE-002-PARTITION-HYPOTHETICAL-MEMBER-001": (2, 3, 2, 0),
    "SNAKE-002-PARTITION-RECURRENCE-SAME-ENTITY-001": (1, 1, 1, 0),
}

ARBITRATION_COUNTS = {
    "SNAKE-002-ARBITRATION-ATTACK-UNRESOLVED-001": (1, 1, 1, 1),
    "SNAKE-002-ARBITRATION-BITE-THEN-VICTORY-001": (2, 1, 2, 1),
    "SNAKE-002-ARBITRATION-ATTEMPT-THEN-ESCAPE-001": (2, 1, 2, 1),
    "SNAKE-002-ARBITRATION-CHASE-CAPTURE-WAKE-001": (3, 1, 3, 1),
    "SNAKE-002-ARBITRATION-DEFEAT-KNOCKDOWN-001": (1, 1, 1, 1),
    "SNAKE-002-ARBITRATION-RETREAT-VS-DISAPPEAR-001": (2, 2, 2, 2),
    "SNAKE-002-ARBITRATION-KILL-VS-FOUND-DEAD-001": (2, 2, 2, 2),
    "SNAKE-002-ARBITRATION-INTERMEDIATE-REVERSED-001": (3, 1, 3, 1),
    "SNAKE-002-ARBITRATION-SCENE-BREAK-001": (2, 2, 2, 2),
    "SNAKE-002-ARBITRATION-QUOTED-ENDING-001": (2, 1, 2, 1),
    "SNAKE-002-ARBITRATION-HYPOTHETICAL-ENDING-001": (2, 1, 2, 1),
    "SNAKE-002-ARBITRATION-AMBIGUOUS-KILL-001": (1, 0, 1, 1),
}


class SnakeContextV04FirewallTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dreams = dict(SNAKE_QA_CASES)

    def test_all_entity_chain_oracles_have_exact_inventory_counts(self):
        for case_id, expected in PARTITION_COUNTS.items():
            graph = extract_snake_event_graph(self.dreams[case_id])
            actual = tuple(len(graph[key]) for key in (
                "events", "snake_mentions", "entity_chain_partitions", "terminal_frontiers"
            ))
            with self.subTest(case_id=case_id):
                self.assertEqual(expected, actual)
                self.assertEqual({"verified": True, "reason_codes": []}, graph["graph_integrity"])

    def test_all_terminal_arbitration_oracles_have_exact_inventory_counts(self):
        for case_id, expected in ARBITRATION_COUNTS.items():
            graph = extract_snake_event_graph(self.dreams[case_id])
            actual = tuple(len(graph[key]) for key in (
                "events", "terminal_frontiers", "arbitration_candidates", "terminal_decisions"
            ))
            with self.subTest(case_id=case_id):
                self.assertEqual(expected, actual)
                self.assertEqual({"verified": True, "reason_codes": []}, graph["graph_integrity"])

    def test_partition_and_arbitration_records_expose_required_provenance(self):
        partition_required = {
            "partition_id", "resolution_status", "snake_entity_id", "snake_candidate_ids",
            "mention_ids", "event_ids", "scene_ids", "chain_ids", "target_ids",
            "location_scope_ids", "terminal_frontier_ids", "count_contribution", "source_spans",
        }
        candidate_required = {
            "candidate_id", "event_id", "snake_id", "chain_id", "scene_id", "target_id",
            "outcome", "precedence_class", "disposition", "reason_codes", "source_span",
        }
        for case_id in set(PARTITION_COUNTS) | set(ARBITRATION_COUNTS):
            graph = extract_snake_event_graph(self.dreams[case_id])
            with self.subTest(case_id=case_id):
                self.assertTrue(all(partition_required <= set(item) for item in graph["entity_chain_partitions"]))
                self.assertTrue(all(candidate_required <= set(item) for item in graph["arbitration_candidates"]))

    def test_unsafe_partition_and_terminal_mutations_fail_closed(self):
        graph = extract_snake_event_graph(self.dreams["SNAKE-002-PARTITION-TWO-DISTINCT-ACTIONS-001"])
        broken = copy.deepcopy(graph)
        broken["entity_chain_partitions"][1]["event_ids"] = ["missing-event"]
        self.assertIn("PARTITION_EVENT_REFERENCE_MISSING", validate_snake_event_graph(broken)["reason_codes"])

        broken = copy.deepcopy(graph)
        broken["entity_chain_partitions"][1]["snake_entity_id"] = "snake-1"
        self.assertIn("ENTITY_PARTITION_CARDINALITY", validate_snake_event_graph(broken)["reason_codes"])

        graph = extract_snake_event_graph(self.dreams["SNAKE-002-ARBITRATION-HYPOTHETICAL-ENDING-001"])
        broken = copy.deepcopy(graph)
        broken["arbitration_candidates"][0]["disposition"] = "selected"
        self.assertIn("NONACTUAL_EVENT_NOT_RELEASED", validate_snake_event_graph(broken)["reason_codes"])

        graph = extract_snake_event_graph(self.dreams["SNAKE-002-ARBITRATION-AMBIGUOUS-KILL-001"])
        broken = copy.deepcopy(graph)
        broken["arbitration_candidates"][0]["disposition"] = "selected"
        self.assertIn("AMBIGUOUS_TERMINAL_NOT_FORCED", validate_snake_event_graph(broken)["reason_codes"])


if __name__ == "__main__":
    unittest.main()
