import copy
import unittest

from app.context_graph import GRAPH_CONTRACT_VERSION, validate_context_graph
from app.teeth_doctrine import build_teeth_doctrine_context


CASES = {
    "dreamer": "My tooth fell out and I tried to put it back.",
    "negated": "My tooth fell out, but I never tried to put it back.",
    "hypothetical": "If my tooth fell out, I would try to put it back.",
    "quoted": 'My tooth fell out. My aunt said, "I tried to put it back."',
    "waking": "My tooth fell out. After I woke up, I tried to put it back in my imagination.",
    "other_owner": "My sister's tooth fell out and she tried to put it back.",
    "external_actor": "My tooth fell out and my sister tried to put it back.",
    "multi_owner": "My tooth fell out and I tried to put it back. My sister's tooth fell out and she left it there.",
    "ambiguous": "My tooth and my sister's tooth fell out. I tried to put it back.",
    "then_firm": "My tooth fell out. I tried to put it back, and then the same tooth fitted firmly back into the same socket.",
    "second_loss": "My left tooth fell out and I tried to put it back. Then another tooth fell out.",
    "reported": "My tooth fell out. My sister told me that she tried to put her tooth back yesterday.",
}


EXPECTED = {
    "dreamer": {
        "entities": {"dreamer", "tooth-1"},
        "events": {"loss-1", "attempt-1"},
        "chains": {"chain-tooth-1"},
        "consumed": {"attempt-1"},
    },
    "negated": {
        "events": {"loss-1", "attempt-1"},
        "consumed": set(),
    },
    "hypothetical": {
        "events": {"hypothetical-loss-1", "attempt-1"},
        "eligible": set(),
        "claims": set(),
        "consumed": set(),
    },
    "quoted": {
        "entities": {"dreamer", "tooth-1", "aunt"},
        "events": {"loss-1", "speech-1", "attempt-1"},
        "attempt_chain": None,
        "consumed": set(),
    },
    "waking": {
        "events": {"loss-1", "waking-imagined-attempt-1"},
        "chains": {"chain-tooth-1"},
        "attempt_chain": None,
        "consumed": set(),
    },
    "other_owner": {
        "entities": {"sister", "sister-tooth-1"},
        "events": {"loss-1", "attempt-1"},
        "chains": {"chain-sister-tooth-1"},
    },
    "external_actor": {
        "entities": {"dreamer", "sister", "tooth-1"},
        "events": {"loss-1", "attempt-1"},
        "attempt_actor": "sister",
        "attempt_owner": "dreamer",
    },
    "multi_owner": {
        "entities": {"dreamer", "dreamer-tooth-1", "sister", "sister-tooth-1"},
        "events": {"dreamer-loss-1", "attempt-1", "sister-loss-1"},
        "loss_count": 2,
        "chain_count": 2,
        "aggregate_losses": {"dreamer-loss-1", "sister-loss-1"},
    },
    "ambiguous": {
        "loss_count": 2,
        "attempt_target": "ambiguous",
        "attempt_owner": "ambiguous",
        "attempt_chain": None,
        "consumed": set(),
    },
    "then_firm": {
        "entities": {"dreamer", "tooth-1"},
        "events": {"loss-1", "attempt-1", "firm-return-1"},
        "frontier": "firm-return-1",
        "historical": {"loss-1", "attempt-1"},
        "public_rules": {"TEETH-END-TERMINAL"},
    },
    "second_loss": {
        "entities": {"dreamer", "left-tooth-1", "tooth-2"},
        "events": {"left-loss-1", "attempt-1", "second-loss-1"},
        "loss_count": 2,
        "attempt_target": ["left-tooth-1"],
        "warning_count": "multiple_people",
    },
    "reported": {
        "entities": {"dreamer", "tooth-1", "sister", "sister-tooth-1"},
        "events": {"dreamer-loss-1", "speech-1", "attempt-1"},
        "attempt_chain": None,
        "consumed": set(),
    },
}


class TeethContextGraphReferentialIntegrityTests(unittest.TestCase):
    maxDiff = None

    def assert_case(self, name):
        decision = build_teeth_doctrine_context(CASES[name])
        graph = decision["context_graph"]
        expected = EXPECTED[name]

        self.assertEqual(GRAPH_CONTRACT_VERSION, graph["contract_version"])
        for collection in (
            "entity_inventory", "event_inventory", "event_chain_inventory",
            "restoration_attempt_records", "aggregate_derivations", "rule_sets",
            "claim_manifest", "terminal_frontiers",
        ):
            self.assertIn(collection, decision)
        self.assertTrue(graph["integrity"]["verified"], graph["integrity"])
        self.assertEqual([], graph["integrity"]["reason_codes"])

        entities = {item["entity_id"] for item in graph["entity_inventory"]}
        events = {item["event_id"] for item in graph["event_inventory"]}
        chains = {item["event_chain_id"] for item in graph["event_chain_inventory"]}
        eligible = {
            item["event_id"] for item in graph["event_inventory"]
            if item["doctrine_eligible"]
        }
        claims = {item["claim_id"] for item in graph["claim_manifest"]}
        attempts = graph["restoration_attempt_records"]
        attempt = attempts[0]
        consumed = {
            attempt_id
            for claim in graph["claim_manifest"]
            for attempt_id in claim["consumed_attempt_ids"]
        }
        public_rules = {
            item["rule_id"] for item in graph["rule_sets"]["public_applied"]
        }

        if "entities" in expected:
            self.assertEqual(expected["entities"], entities)
        if "events" in expected:
            self.assertEqual(expected["events"], events)
        if "chains" in expected:
            self.assertEqual(expected["chains"], chains)
        if "eligible" in expected:
            self.assertEqual(expected["eligible"], eligible)
        if "claims" in expected:
            self.assertEqual(expected["claims"], claims)
        if "consumed" in expected:
            self.assertEqual(expected["consumed"], consumed)
        if "attempt_chain" in expected:
            self.assertEqual(expected["attempt_chain"], attempt["event_chain_id_or_null"])
        if "attempt_actor" in expected:
            self.assertEqual(expected["attempt_actor"], attempt["actor_id_or_ambiguous"])
        if "attempt_owner" in expected:
            self.assertEqual(expected["attempt_owner"], attempt["owner_id_or_ambiguous"])
        if "attempt_target" in expected:
            self.assertEqual(expected["attempt_target"], attempt["target_tooth_ids_or_ambiguous"])
        if "loss_count" in expected:
            self.assertEqual(
                expected["loss_count"],
                sum(item["event_type"] == "tooth_loss" for item in graph["event_inventory"]),
            )
        if "chain_count" in expected:
            self.assertEqual(expected["chain_count"], len(chains))
        if "aggregate_losses" in expected:
            for aggregate in graph["aggregate_derivations"]:
                self.assertEqual(expected["aggregate_losses"], set(aggregate["contributing_event_ids"]))
        if "warning_count" in expected:
            self.assertEqual(expected["warning_count"], decision["warning_count"])
        if "frontier" in expected:
            self.assertEqual(expected["frontier"], graph["terminal_frontiers"][0]["terminal_event_id"])
        if "historical" in expected:
            self.assertEqual(expected["historical"], set(graph["terminal_frontiers"][0]["historical_event_ids"]))
        if "public_rules" in expected:
            self.assertEqual(expected["public_rules"], public_rules)

    def assert_mutation_reason(self, graph, reason):
        result = validate_context_graph(graph)
        self.assertFalse(result["verified"])
        self.assertIn(reason, result["reason_codes"])


def _case_test(name):
    def test(self):
        self.assert_case(name)
    return test


for _name in CASES:
    setattr(
        TeethContextGraphReferentialIntegrityTests,
        f"test_ref_{_name}",
        _case_test(_name),
    )


def _mutation_graph(name):
    return copy.deepcopy(build_teeth_doctrine_context(CASES[name])["context_graph"])


def test_mut_dangling_tooth(self):
    graph = _mutation_graph("dreamer")
    graph["entity_inventory"] = [item for item in graph["entity_inventory"] if item["entity_id"] != "tooth-1"]
    self.assert_mutation_reason(graph, "DANGLING_ENTITY_REFERENCE")


def test_mut_dangling_event(self):
    graph = _mutation_graph("dreamer")
    graph["aggregate_derivations"][0]["contributing_event_ids"].append("missing-loss")
    self.assert_mutation_reason(graph, "DANGLING_EVENT_REFERENCE")


def test_mut_dangling_chain(self):
    graph = _mutation_graph("dreamer")
    graph["event_chain_inventory"] = []
    self.assert_mutation_reason(graph, "DANGLING_CHAIN_REFERENCE")


def test_mut_duplicate_id(self):
    graph = _mutation_graph("dreamer")
    graph["entity_inventory"].append(copy.deepcopy(graph["entity_inventory"][1]))
    self.assert_mutation_reason(graph, "DUPLICATE_TYPED_ID")


def test_mut_cross_owner(self):
    graph = _mutation_graph("multi_owner")
    attempt = next(item for item in graph["event_inventory"] if item["event_id"] == "attempt-1")
    attempt["event_chain_id_or_null"] = "chain-sister-tooth-1"
    attempt_record = graph["restoration_attempt_records"][0]
    attempt_record["event_chain_id_or_null"] = "chain-sister-tooth-1"
    for chain in graph["event_chain_inventory"]:
        if chain["event_chain_id"] == "chain-dreamer-tooth-1":
            chain["event_ids"].remove("attempt-1")
        elif chain["event_chain_id"] == "chain-sister-tooth-1":
            chain["event_ids"].append("attempt-1")
    self.assert_mutation_reason(graph, "CROSS_BOUNDARY_REFERENCE")


def test_mut_waking_chain(self):
    graph = _mutation_graph("waking")
    attempt_id = "waking-imagined-attempt-1"
    event = next(item for item in graph["event_inventory"] if item["event_id"] == attempt_id)
    event["event_chain_id_or_null"] = "chain-tooth-1"
    graph["restoration_attempt_records"][0]["event_chain_id_or_null"] = "chain-tooth-1"
    graph["event_chain_inventory"][0]["event_ids"].append(attempt_id)
    self.assert_mutation_reason(graph, "CROSS_BOUNDARY_REFERENCE")


def test_mut_ineligible_consumed(self):
    graph = _mutation_graph("negated")
    graph["claim_manifest"].append({
        "claim_id": "claim-bad-attempt", "claim_type": "structural_attempt_narration",
        "released": True, "consumed_event_ids": ["attempt-1"],
        "consumed_attempt_ids": ["attempt-1"],
        "consumed_rule_ids": [graph["rule_sets"]["public_applied"][0]["rule_id"]],
        "consumed_span_ids": ["span-attempt-1"],
    })
    self.assert_mutation_reason(graph, "INELIGIBLE_ATTEMPT_CONSUMED")


def test_mut_omitted_loss(self):
    graph = _mutation_graph("multi_owner")
    for aggregate in graph["aggregate_derivations"]:
        aggregate["contributing_event_ids"] = ["dreamer-loss-1"]
        aggregate["omitted_event_ids"] = []
    self.assert_mutation_reason(graph, "AGGREGATE_EVENT_OMITTED")


def test_mut_cardinality(self):
    graph = _mutation_graph("multi_owner")
    for aggregate in graph["aggregate_derivations"]:
        if aggregate["field"] == "count":
            aggregate["value"] = "one"
        elif aggregate["field"] == "warning_count":
            aggregate["value"] = "one_person"
    self.assert_mutation_reason(graph, "AGGREGATE_CARDINALITY_MISMATCH")


def test_mut_rule_provenance(self):
    graph = _mutation_graph("dreamer")
    graph["rule_sets"]["public_applied"].append({
        "rule_id": "TEETH-FALLOUT-ONE", "source_event_ids": [], "source_span_ids": []
    })
    self.assert_mutation_reason(graph, "RULE_EVENT_PROVENANCE_MISSING")


def test_mut_claim_consumption(self):
    graph = _mutation_graph("dreamer")
    graph["claim_manifest"].append({
        "claim_id": "claim-bad-attempt", "claim_type": "structural_attempt_narration",
        "released": True, "consumed_event_ids": [], "consumed_attempt_ids": [],
        "consumed_rule_ids": [], "consumed_span_ids": [],
    })
    self.assert_mutation_reason(graph, "CLAIM_REFERENCE_MISSING")


def test_mut_terminal_frontier(self):
    graph = _mutation_graph("then_firm")
    graph["terminal_frontiers"][0]["terminal_event_id"] = "loss-1"
    self.assert_mutation_reason(graph, "TERMINAL_FRONTIER_DANGLING")


for _test in (
    test_mut_dangling_tooth, test_mut_dangling_event, test_mut_dangling_chain,
    test_mut_duplicate_id, test_mut_cross_owner, test_mut_waking_chain,
    test_mut_ineligible_consumed, test_mut_omitted_loss, test_mut_cardinality,
    test_mut_rule_provenance, test_mut_claim_consumption, test_mut_terminal_frontier,
):
    setattr(TeethContextGraphReferentialIntegrityTests, _test.__name__, _test)


if __name__ == "__main__":
    unittest.main()
