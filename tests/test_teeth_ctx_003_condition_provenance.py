import copy
import unittest

from app.condition_provenance import (
    CONDITION_PROVENANCE_CONTRACT_VERSION,
    validate_condition_provenance,
)
from app.teeth_doctrine import build_teeth_doctrine_context


CASES = {
    "gums_negated": "My gums were bleeding, but no tooth was loose and none fell out.",
    "gums_only": "My gums were bleeding.",
    "gums_retained": "My gums were bleeding and every tooth stayed firm.",
    "loose_negated_loss": "My tooth was loose but did not fall out.",
    "wobbly_retained": "My tooth was wobbly and stayed in my mouth.",
    "two_loose": "Two of my teeth were loose, but neither fell out.",
    "other_owner": "My sister's tooth was loose but did not fall out.",
    "negated_loose_then_loss": "My tooth was not loose, but it fell out.",
    "loose_then_loss": "My tooth became loose, then it fell out.",
    "quoted": 'My aunt said, "My tooth is loose." My own tooth stayed firm.',
    "hypothetical": "If my tooth were loose, I would visit a dentist.",
    "multi_owner": "My gums were bleeding. My sister's tooth was loose but did not fall out.",
}


class ConditionStateProvenanceTests(unittest.TestCase):
    def graph(self, name):
        doctrine = build_teeth_doctrine_context(CASES[name])
        graph = doctrine["context_graph"]
        self.assertEqual(
            CONDITION_PROVENANCE_CONTRACT_VERSION,
            graph["condition_provenance_contract_version"],
        )
        self.assertTrue(graph["condition_provenance_integrity"]["verified"])
        self.assertTrue(graph["integrity"]["verified"])
        for event in graph["event_inventory"]:
            span = event["source_span"]
            self.assertEqual(CASES[name][span["start"]:span["end"]], span["text"])
        return doctrine, graph

    @staticmethod
    def events(graph):
        return {event["event_id"]: event for event in graph["event_inventory"]}

    @staticmethod
    def public_rules(graph):
        return {
            record["rule_id"]
            for record in graph["rule_sets"]["public_applied"]
        }

    @staticmethod
    def condition_claims(graph):
        condition_rules = {"TEETH-STATE-LOOSE", "TEETH-OMEN-GUM-BLOOD"}
        return [
            claim for claim in graph["claim_manifest"]
            if set(claim.get("consumed_rule_ids", [])) & condition_rules
        ]

    def test_gums_with_negated_loose_and_loss(self):
        _, graph = self.graph("gums_negated")
        events = self.events(graph)
        self.assertEqual(
            {"gum_bleeding_condition", "loose_tooth_condition", "tooth_loss"},
            {event["event_type"] for event in events.values()},
        )
        self.assertEqual({"gum-bleeding-1"}, {key for key, event in events.items() if event["doctrine_eligible"]})
        self.assertEqual(2, sum(event["polarity"] == "negated" for event in events.values()))
        self.assertEqual({"TEETH-OMEN-GUM-BLOOD"}, self.public_rules(graph))
        self.assertEqual(1, len(self.condition_claims(graph)))

    def test_gums_only(self):
        _, graph = self.graph("gums_only")
        events = self.events(graph)
        self.assertEqual({"gum-bleeding-1"}, set(events))
        event = events["gum-bleeding-1"]
        self.assertEqual("dreamer", event["owner_id_or_ambiguous"])
        self.assertEqual(["gums-1"], event["target_entity_ids_or_ambiguous"])
        self.assertEqual("chain-gums-1", event["event_chain_id_or_null"])
        self.assertEqual({"TEETH-OMEN-GUM-BLOOD"}, self.public_rules(graph))

    def test_gums_with_retained_teeth(self):
        _, graph = self.graph("gums_retained")
        self.assertEqual({"gum-bleeding-1", "retained-state-1"}, set(self.events(graph)))
        self.assertFalse(any(event["event_type"] == "tooth_loss" for event in graph["event_inventory"]))
        self.assertNotIn("TEETH-STATE-LOOSE", self.public_rules(graph))

    def test_loose_with_negated_loss(self):
        _, graph = self.graph("loose_negated_loss")
        events = self.events(graph)
        self.assertEqual({"loose-1", "negated-loss-1"}, set(events))
        self.assertEqual({"loose-1"}, {key for key, event in events.items() if event["doctrine_eligible"]})
        self.assertEqual({"TEETH-STATE-LOOSE"}, self.public_rules(graph))
        self.assertEqual(1, len(self.condition_claims(graph)))

    def test_wobbly_retained(self):
        _, graph = self.graph("wobbly_retained")
        events = self.events(graph)
        self.assertEqual("wobbly", events["loose-1"]["state_alias"])
        self.assertEqual("retained_loose", events["retained-state-1"]["terminal_state"])
        frontier = graph["terminal_frontiers"][0]
        self.assertEqual("retained-state-1", frontier["terminal_event_id"])
        self.assertEqual(["loose-1"], frontier["historical_event_ids"])

    def test_two_loose_does_not_leak_fallout_count_rules(self):
        _, graph = self.graph("two_loose")
        event = self.events(graph)["loose-1"]
        self.assertEqual("multiple", event["quantity"])
        self.assertEqual(2, event["quantity_cardinality"])
        self.assertEqual(2, len(event["target_entity_ids_or_ambiguous"]))
        self.assertFalse(self.public_rules(graph) & {"TEETH-FALLOUT-ONE", "TEETH-FALLOUT-MULTIPLE"})

    def test_other_owner_remains_other_owner(self):
        _, graph = self.graph("other_owner")
        events = self.events(graph)
        self.assertEqual({"sister-loose-1", "sister-negated-loss-1"}, set(events))
        self.assertTrue(all(event["owner_id_or_ambiguous"] == "sister" for event in events.values()))
        claim = self.condition_claims(graph)[0]
        self.assertEqual({"sister"}, {events[event_id]["owner_id_or_ambiguous"] for event_id in claim["consumed_event_ids"]})

    def test_negated_loose_does_not_suppress_later_loss(self):
        doctrine, graph = self.graph("negated_loose_then_loss")
        self.assertEqual({"negated-loose-1", "loss-1"}, set(self.events(graph)))
        self.assertEqual({"loss-1"}, {event["event_id"] for event in graph["event_inventory"] if event["doctrine_eligible"]})
        self.assertEqual({"TEETH-FALLOUT-OWN", "TEETH-FALLOUT-ONE"}, self.public_rules(graph))
        self.assertEqual("tooth_loss", doctrine["warning_kind"])

    def test_loose_then_loss_preserves_transition_and_history(self):
        doctrine, graph = self.graph("loose_then_loss")
        self.assertEqual({"loose-1", "loss-1"}, set(self.events(graph)))
        self.assertEqual("loose-1_before_loss-1", graph["condition_transition_edges"][0]["edge_id"])
        frontier = graph["terminal_frontiers"][0]
        self.assertEqual("loss-1", frontier["terminal_event_id"])
        self.assertEqual(["loose-1"], frontier["historical_event_ids"])
        self.assertEqual("tooth_loss", doctrine["warning_kind"])

    def test_quoted_condition_is_recorded_but_ineligible(self):
        doctrine, graph = self.graph("quoted")
        events = self.events(graph)
        quoted = events["quoted-loose-1"]
        self.assertEqual("quoted", quoted["modality"])
        self.assertFalse(quoted["doctrine_eligible"])
        self.assertFalse(doctrine["active_warning"])
        self.assertEqual(set(), self.public_rules(graph))
        self.assertEqual([], self.condition_claims(graph))

    def test_hypothetical_condition_is_recorded_but_ineligible(self):
        doctrine, graph = self.graph("hypothetical")
        event = self.events(graph)["hypothetical-loose-1"]
        self.assertEqual("nonactual", event["actuality"])
        self.assertEqual("conditional_hypothetical", event["modality"])
        self.assertEqual("If my tooth were loose", event["source_span"]["text"])
        self.assertFalse(doctrine["active_warning"])

    def test_multi_owner_conditions_stay_separate(self):
        doctrine, graph = self.graph("multi_owner")
        conditions = [
            event for event in graph["event_inventory"]
            if event["event_type"] in {"gum_bleeding_condition", "loose_tooth_condition"}
        ]
        self.assertEqual(2, len(conditions))
        self.assertEqual({"dreamer", "sister"}, {event["owner_id_or_ambiguous"] for event in conditions})
        self.assertEqual(2, len({event["event_chain_id_or_null"] for event in conditions}))
        self.assertEqual({"TEETH-OMEN-GUM-BLOOD", "TEETH-STATE-LOOSE"}, self.public_rules(graph))
        self.assertEqual("multiple_condition_warnings", doctrine["warning_kind"])


class ConditionStateProvenanceMutationTests(unittest.TestCase):
    def valid_graph(self, name):
        return copy.deepcopy(build_teeth_doctrine_context(CASES[name])["context_graph"])

    def assert_rejected(self, graph, reason):
        result = validate_condition_provenance(graph)
        self.assertFalse(result["verified"])
        self.assertIn(reason, result["reason_codes"])

    def test_rejects_condition_rule_without_event(self):
        graph = self.valid_graph("gums_only")
        graph["event_inventory"] = []
        self.assert_rejected(graph, "CONDITION_EVENT_MISSING")

    def test_rejects_condition_without_owner(self):
        graph = self.valid_graph("gums_only")
        graph["event_inventory"][0]["owner_id_or_ambiguous"] = None
        self.assert_rejected(graph, "CONDITION_OWNER_MISSING")

    def test_rejects_condition_without_target(self):
        graph = self.valid_graph("gums_only")
        graph["event_inventory"][0]["target_entity_ids_or_ambiguous"] = []
        self.assert_rejected(graph, "CONDITION_TARGET_MISSING")

    def test_rejects_condition_without_chain_membership(self):
        graph = self.valid_graph("gums_only")
        graph["event_chain_inventory"][0]["event_ids"] = []
        self.assert_rejected(graph, "CONDITION_CHAIN_MISSING")

    def test_rejects_condition_without_exact_span(self):
        graph = self.valid_graph("gums_only")
        graph["event_inventory"][0]["source_span"] = {}
        self.assert_rejected(graph, "CONDITION_SPAN_MISSING")

    def test_rejects_negated_condition_rule_leak(self):
        graph = self.valid_graph("gums_negated")
        record = next(record for record in graph["rule_sets"]["public_applied"] if record["rule_id"] == "TEETH-OMEN-GUM-BLOOD")
        record["rule_id"] = "TEETH-STATE-LOOSE"
        record["source_event_ids"] = ["negated-loose-1"]
        record["source_span_ids"] = ["span-negated-loose-1"]
        graph["rule_sets"]["warning_active"] = [record]
        self.assert_rejected(graph, "NEGATED_CONDITION_RULE_LEAK")

    def test_rejects_nonactual_condition_rule_leak(self):
        graph = self.valid_graph("quoted")
        event = next(event for event in graph["event_inventory"] if event["event_id"] == "quoted-loose-1")
        record = {
            "rule_id": "TEETH-STATE-LOOSE",
            "source_event_ids": ["quoted-loose-1"],
            "source_span_ids": [event["source_span"]["span_id"]],
        }
        graph["rule_sets"]["public_applied"] = [record]
        graph["rule_sets"]["warning_active"] = [record]
        self.assert_rejected(graph, "NONACTUAL_CONDITION_RULE_LEAK")

    def test_rejects_condition_loss_transition_collapse(self):
        graph = self.valid_graph("loose_then_loss")
        graph["condition_transition_edges"] = []
        self.assert_rejected(graph, "CONDITION_LOSS_COLLAPSE")

    def test_rejects_condition_quantity_as_fallout_count(self):
        graph = self.valid_graph("two_loose")
        source = next(record for record in graph["rule_sets"]["public_applied"] if record["rule_id"] == "TEETH-STATE-LOOSE")
        graph["rule_sets"]["public_applied"].append({**source, "rule_id": "TEETH-FALLOUT-MULTIPLE"})
        self.assert_rejected(graph, "CONDITION_QUANTITY_RULE_LEAK")

    def test_rejects_cross_owner_condition_merge(self):
        graph = self.valid_graph("multi_owner")
        sister = next(event for event in graph["event_inventory"] if event["event_id"] == "sister-loose-1")
        sister["event_chain_id_or_null"] = "chain-gums-1"
        gum_chain = next(chain for chain in graph["event_chain_inventory"] if chain["event_chain_id"] == "chain-gums-1")
        gum_chain["event_ids"].append("sister-loose-1")
        gum_chain["entity_ids"].extend(sister["target_entity_ids_or_ambiguous"])
        self.assert_rejected(graph, "CROSS_OWNER_CONDITION_COLLAPSE")

    def test_rejects_condition_rule_bound_to_wrong_event_type(self):
        graph = self.valid_graph("multi_owner")
        gum = next(event for event in graph["event_inventory"] if event["event_id"] == "gum-bleeding-1")
        loose_record = next(record for record in graph["rule_sets"]["public_applied"] if record["rule_id"] == "TEETH-STATE-LOOSE")
        loose_record["source_event_ids"] = ["gum-bleeding-1"]
        loose_record["source_span_ids"] = [gum["source_span"]["span_id"]]
        self.assert_rejected(graph, "CONDITION_RULE_EVENT_MISMATCH")

    def test_rejects_condition_claim_without_event_rule_span_path(self):
        graph = self.valid_graph("gums_only")
        claim = graph["claim_manifest"][0]
        claim["consumed_event_ids"] = []
        claim["consumed_span_ids"] = []
        self.assert_rejected(graph, "CONDITION_CLAIM_PATH_MISSING")


if __name__ == "__main__":
    unittest.main()
