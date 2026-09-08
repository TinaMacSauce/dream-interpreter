import copy
import hashlib
import json
import unittest

from app.snake_event_graph import (
    extract_snake_event_graph,
    validate_snake_event_graph,
)


TARGET_RULE_ORACLE_DIGESTS = {
    "A snake watched me from the gate.": "d263c8726199d0d92f7946fe1882bf3a07d18fb85879532f053c7ec0c6a77b33",
    "The snake watched my sister while I stood nearby.": "13fc001c5ee0b5f849781db859c9bd3b56180e93b9a639e153cce3e809e172e2",
    "One snake watched my brother and another watched me.": "5c57ad6a25192d3cd373ab3191160314e3e6b2eaa4a698b852808c0eea7f0ed6",
    "My sister and cousin stood together while the snake watched her.": "fcf03052b268f82b42f71ab01b9fbdf5bbf5a163efd7d4b139c2b3643352e567",
    "The snake tried to bite me but never touched me.": "b135448f0aaa0008bf492fb2929df7d2cad2d8b07d8cfd9f98c83b7a6c83dc3b",
    "A snake tried to bite my sister's wrist but missed.": "02c432eedec631715169f43208f8848a9e208b5929b6baeb163c373134d8c917",
    "The snake tried to bite the child, but a shield blocked it.": "654c9c113701c4574d0160c5153e2cc520dbc48a75a35dbfbaaf564be37e3416",
    "The snake tried to bite me, then it bit my brother instead.": "63667cd19103d905b8a228ed76af5e6d72c3d078ab06e3cdd14bed3c3258dcf1",
    "The snake did not try to bite my sister; it watched my brother.": "cb480694b138db0535cb9d0f52eb8277f85cc5479fbbede985ce3f5526dc2d39",
    "If the snake tried to bite my cousin, I would run, but it only watched me.": "dbe0cfff3b1ac6ac2c40f33d3a57dec85f287fa4b7d7a03b2f7f578b4f9dc860",
    "One snake watched my mother while another tried to bite me and missed.": "179e26cac53f52066ab68d6362986fb4b0b125c4cb45527ce06d09392b165f07",
    "A snake watched my brother from across the room.": "5ab70406c12bdfb630c6fb5e6a956a5da8a0225f9c703355a9a9d3697109b74a",
}


class SnakeTargetRuleProvenanceTests(unittest.TestCase):
    def graph(self, dream):
        return extract_snake_event_graph(dream)

    def test_all_context_target_rule_oracles_match_exactly(self):
        for dream, expected_digest in TARGET_RULE_ORACLE_DIGESTS.items():
            graph = self.graph(dream)
            encoded = json.dumps(
                graph["target_intent_records"],
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
            with self.subTest(dream=dream):
                self.assertEqual(
                    "snake-rule-provenance-target-v1",
                    graph["target_rule_contract_version"],
                )
                self.assertEqual(expected_digest, hashlib.sha256(encoded).hexdigest())
                used_rules = {
                    record["rule_id"] for record in graph["target_intent_records"]
                }
                self.assertEqual(
                    used_rules,
                    {record["rule_id"] for record in graph["rule_provenance_records"]},
                )
                self.assertEqual(
                    {"verified": True, "reason_codes": []},
                    graph["graph_integrity"],
                )

    def assert_rejected(self, dream, mutate, reason):
        graph = copy.deepcopy(self.graph(dream))
        mutate(graph)
        self.assertIn(reason, validate_snake_event_graph(graph)["reason_codes"])

    def test_all_unsafe_target_rule_mutations_fail_closed(self):
        watch_dreamer = "A snake watched me from the gate."
        watch_sister = "The snake watched my sister while I stood nearby."
        watch_multi = "One snake watched my brother and another watched me."
        watch_ambiguous = "My sister and cousin stood together while the snake watched her."
        attempt = "The snake tried to bite me but never touched me."
        attempt_sister = "A snake tried to bite my sister's wrist but missed."
        blocked = "The snake tried to bite the child, but a shield blocked it."
        attempt_then_bite = "The snake tried to bite me, then it bit my brother instead."
        negated = "The snake did not try to bite my sister; it watched my brother."
        hypothetical = "If the snake tried to bite my cousin, I would run, but it only watched me."
        mixed = "One snake watched my mother while another tried to bite me and missed."
        interleaved = "A snake watched my brother from across the room."

        self.assert_rejected(
            watch_dreamer,
            lambda graph: graph["target_intent_records"][0].update(
                decision_id="DEC-SNAKE-2026-09-08-01"
            ),
            "RULE_PROVENANCE_DECISION_MISMATCH",
        )
        self.assert_rejected(
            watch_sister,
            lambda graph: graph["target_intent_records"][0].update(
                target_ids=["dreamer"]
            ),
            "EXPLICIT_TARGET_NOT_REPORTER_DEFAULT",
        )
        self.assert_rejected(
            watch_multi,
            lambda graph: graph["target_intent_records"].pop(),
            "TARGET_INTENT_CARDINALITY_LOSS",
        )
        self.assert_rejected(
            watch_ambiguous,
            lambda graph: graph["target_intent_records"][0].update(
                target_ids=["sister"], target_status="resolved"
            ),
            "AMBIGUOUS_TARGET_NOT_FORCED",
        )
        self.assert_rejected(
            attempt,
            lambda graph: graph["target_intent_records"][0].update(
                completion="completed", contact_status="contact"
            ),
            "ATTEMPT_NOT_COMPLETED_CONTACT",
        )
        self.assert_rejected(
            attempt_sister,
            lambda graph: graph["target_intent_records"][0].update(
                outcome_status="defeat"
            ),
            "ATTEMPT_NOT_DEFEAT",
        )
        self.assert_rejected(
            blocked,
            lambda graph: graph["target_intent_records"][0].update(
                contact_status="contact"
            ),
            "BLOCKED_CONTACT_NOT_COMPLETED",
        )
        self.assert_rejected(
            attempt_then_bite,
            lambda graph: graph["target_intent_records"][0].update(
                target_ids=["brother"]
            ),
            "EVENT_TARGET_HISTORY_OVERWRITE",
        )
        self.assert_rejected(
            negated,
            lambda graph: graph["target_intent_records"][0].update(
                release_status="released_attempt_only"
            ),
            "NEGATED_EVENT_NOT_RELEASED",
        )
        self.assert_rejected(
            hypothetical,
            lambda graph: graph["target_intent_records"][0].update(
                release_status="released_attempt_only"
            ),
            "HYPOTHETICAL_EVENT_NOT_RELEASED",
        )
        self.assert_rejected(
            mixed,
            lambda graph: graph["target_intent_records"][1].update(
                snake_scope_ids=["snake-1"]
            ),
            "TARGET_PARTITION_MISMATCH",
        )
        self.assert_rejected(
            interleaved,
            lambda graph: graph["target_intent_records"][0].update(
                rule_id="ANIMAL-HELP-DIVINE"
            ),
            "REGISTRY_CLUSTER_SCOPE_MISMATCH",
        )


if __name__ == "__main__":
    unittest.main()
