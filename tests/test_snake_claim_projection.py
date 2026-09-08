import copy
import unittest

from app.routes.qa import SNAKE_QA_CASES
from app.snake_event_graph import extract_snake_event_graph, validate_snake_event_graph


CLAIM_COUNTS = {
    "SNAKE-002-PARTITION-TWO-DISTINCT-ACTIONS-001": (2, 2),
    "SNAKE-002-PARTITION-THREE-MIXED-ENDINGS-001": (3, 4),
    "SNAKE-002-PARTITION-MODIFIER-SCOPE-001": (4, 2),
    "SNAKE-002-PARTITION-LOCATION-SCOPE-001": (4, 2),
    "SNAKE-002-PARTITION-TARGET-SCOPE-001": (2, 2),
    "SNAKE-003-TARGET-LINEAGE-VENOM-THIRD-PARTY-001": (2, 2),
    "SNAKE-002-PARTITION-PRONOUN-AMBIGUOUS-001": (1, 1),
    "SNAKE-002-PARTITION-NEGATED-MEMBER-001": (2, 2),
    "SNAKE-002-PARTITION-HYPOTHETICAL-MEMBER-001": (2, 2),
    "SNAKE-002-ARBITRATION-BITE-THEN-VICTORY-001": (2, 2),
    "SNAKE-003-TRANSFORM-SAFETY-001": (1, 1),
    "SNAKE-002-RECURRENCE-UNFINISHED-001": (2, 2),
}


class SnakeClaimProjectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dreams = dict(SNAKE_QA_CASES)

    def graph(self, case_id):
        return extract_snake_event_graph(self.dreams[case_id])

    def test_all_context_oracles_emit_lossless_claim_manifests(self):
        required_claim = {
            "claim_id", "source_event_ids", "snake_scope_ids", "chain_ids",
            "target_id", "target_status", "rule_id", "claim_family",
            "release_status", "semantic_value", "source_layer",
            "certainty_profile", "safety_qualifiers", "source_spans",
        }
        required_projection = {
            "projection_id", "field", "claim_ids", "projection_status",
            "omitted_claim_ids", "governs_narration",
        }
        for case_id, expected in CLAIM_COUNTS.items():
            graph = self.graph(case_id)
            with self.subTest(case_id=case_id):
                self.assertEqual(
                    "snake-claim-projection-v1",
                    graph["claim_projection_contract_version"],
                )
                self.assertEqual(
                    expected,
                    (
                        len(graph["atomic_claims"]),
                        len(graph["claim_projection_manifest"]),
                    ),
                )
                self.assertTrue(
                    all(
                        required_claim <= set(item)
                        for item in graph["atomic_claims"]
                    )
                )
                self.assertTrue(
                    all(
                        required_projection <= set(item)
                        for item in graph["claim_projection_manifest"]
                    )
                )
                self.assertEqual(
                    {"verified": True, "reason_codes": []},
                    graph["graph_integrity"],
                )

    def test_released_claims_are_losslessly_projected(self):
        for case_id in CLAIM_COUNTS:
            graph = self.graph(case_id)
            released = {
                item["claim_id"]
                for item in graph["atomic_claims"]
                if item["release_status"]
                in {"released", "released_historical"}
            }
            projected = {
                claim_id
                for item in graph["claim_projection_manifest"]
                if item["projection_status"] == "lossless"
                for claim_id in item["claim_ids"]
            }
            with self.subTest(case_id=case_id):
                self.assertEqual(released, projected)

    def test_unsafe_projection_mutations_fail_closed(self):
        graph = self.graph(
            "SNAKE-002-PARTITION-TWO-DISTINCT-ACTIONS-001"
        )
        broken = copy.deepcopy(graph)
        broken["claim_projection_manifest"] = broken[
            "claim_projection_manifest"
        ][:1]
        self.assertIn(
            "RELEASED_CLAIM_NOT_PROJECTED",
            validate_snake_event_graph(broken)["reason_codes"],
        )

        broken = copy.deepcopy(graph)
        broken["atomic_claims"][1]["chain_ids"] = ["chain-snake-1"]
        self.assertIn(
            "CLAIM_EVENT_SNAKE_MISMATCH",
            validate_snake_event_graph(broken)["reason_codes"],
        )

        graph = self.graph(
            "SNAKE-002-PARTITION-THREE-MIXED-ENDINGS-001"
        )
        broken = copy.deepcopy(graph)
        broken["claim_projection_manifest"][-1]["governs_narration"] = True
        self.assertIn(
            "LOSSY_PROJECTION_NOT_NARRATION_SOURCE",
            validate_snake_event_graph(broken)["reason_codes"],
        )

        graph = self.graph("SNAKE-002-PARTITION-LOCATION-SCOPE-001")
        broken = copy.deepcopy(graph)
        next(
            item
            for item in broken["atomic_claims"]
            if item["claim_family"] == "location_sphere"
        )["safety_qualifiers"] = []
        self.assertIn(
            "LOCATION_NOT_CULPRIT",
            validate_snake_event_graph(broken)["reason_codes"],
        )

        graph = self.graph(
            "SNAKE-003-TARGET-LINEAGE-VENOM-THIRD-PARTY-001"
        )
        broken = copy.deepcopy(graph)
        next(
            item
            for item in broken["atomic_claims"]
            if item["claim_family"] == "venom_modifier"
        )["safety_qualifiers"] = []
        self.assertIn(
            "VENOM_SAFETY_BOUNDARY",
            validate_snake_event_graph(broken)["reason_codes"],
        )

        graph = self.graph(
            "SNAKE-002-PARTITION-PRONOUN-AMBIGUOUS-001"
        )
        broken = copy.deepcopy(graph)
        broken["atomic_claims"][0]["release_status"] = "released"
        self.assertIn(
            "AMBIGUOUS_SNAKE_ACTOR_NOT_FORCED",
            validate_snake_event_graph(broken)["reason_codes"],
        )

        graph = self.graph(
            "SNAKE-002-PARTITION-NEGATED-MEMBER-001"
        )
        broken = copy.deepcopy(graph)
        broken["atomic_claims"][0]["release_status"] = "released"
        self.assertIn(
            "NEGATED_EVENT_NOT_RELEASED",
            validate_snake_event_graph(broken)["reason_codes"],
        )

        graph = self.graph(
            "SNAKE-002-PARTITION-HYPOTHETICAL-MEMBER-001"
        )
        broken = copy.deepcopy(graph)
        broken["atomic_claims"][0]["release_status"] = "released"
        self.assertIn(
            "HYPOTHETICAL_EVENT_NOT_RELEASED",
            validate_snake_event_graph(broken)["reason_codes"],
        )

        graph = self.graph(
            "SNAKE-002-ARBITRATION-BITE-THEN-VICTORY-001"
        )
        broken = copy.deepcopy(graph)
        broken["terminal_decisions"][0]["retained_history_event_ids"] = []
        self.assertIn(
            "CLAIM_EVENT_HISTORY_NOT_CONSERVED",
            validate_snake_event_graph(broken)["reason_codes"],
        )

        graph = self.graph("SNAKE-003-TRANSFORM-SAFETY-001")
        broken = copy.deepcopy(graph)
        broken["atomic_claims"][0]["safety_qualifiers"] = []
        self.assertIn(
            "TRANSFORMED_PERSON_NOT_DEFINITIVE_ENEMY",
            validate_snake_event_graph(broken)["reason_codes"],
        )

        graph = self.graph(
            "SNAKE-002-RECURRENCE-UNFINISHED-001"
        )
        broken = copy.deepcopy(graph)
        next(
            item
            for item in broken["atomic_claims"]
            if item["claim_family"] == "recurrence_context"
        )["semantic_value"] = "guaranteed_recurrence"
        self.assertIn(
            "RECURRENCE_NOT_GUARANTEED",
            validate_snake_event_graph(broken)["reason_codes"],
        )


if __name__ == "__main__":
    unittest.main()
