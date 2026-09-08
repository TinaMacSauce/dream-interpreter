import copy
import unittest

from app.claim_provenance import PROVENANCE_CONTRACT_VERSION, validate_claim_provenance
from app.teeth_doctrine import build_teeth_doctrine_context


CASES = {
    "one": "My tooth fell out.",
    "multiple": "Three of my teeth fell out.",
    "pain": "My tooth fell out and it hurt badly.",
    "blood": "My tooth fell out, then I saw blood on the fallen tooth.",
    "gums": "My gums were bleeding, but no tooth was loose and none fell out.",
    "loose": "My tooth was loose but did not fall out.",
    "other_owner": "My sister's tooth fell out.",
    "external_actor": "My sister pulled my tooth out.",
    "terminal": "My tooth fell out, and later the same tooth fitted firmly back into the same socket.",
    "negated": "My tooth did not fall out.",
    "multi_owner": "My tooth fell out. My sister's tooth fell out.",
    "second_loss": "My left tooth fell out. I tried to put it back. Then another tooth fell out.",
}

EXPECTED = {
    "one": ({"loss-1"}, {"TEETH-FALLOUT-OWN", "TEETH-FALLOUT-ONE"}, 1, 1, 1),
    "multiple": ({"loss-1"}, {"TEETH-FALLOUT-OWN", "TEETH-FALLOUT-MULTIPLE"}, 3, 1, 1),
    "pain": ({"loss-1", "pain-1"}, {"TEETH-FALLOUT-OWN", "TEETH-FALLOUT-ONE", "TEETH-MOD-PAIN"}, 1, 1, 1),
    "blood": ({"loss-1", "blood-1"}, {"TEETH-FALLOUT-OWN", "TEETH-FALLOUT-ONE", "TEETH-MOD-BLOOD"}, 1, 1, 1),
    "gums": ({"gum-bleeding-1", "negated-loose-1", "negated-loss-1"}, {"TEETH-OMEN-GUM-BLOOD"}, 0, 0, 0),
    "loose": ({"loose-1", "negated-loss-1"}, {"TEETH-STATE-LOOSE"}, 0, 0, 0),
    "other_owner": ({"loss-1"}, {"TEETH-FALLOUT-OTHER", "TEETH-FALLOUT-ONE"}, 1, 1, 1),
    "external_actor": ({"loss-1"}, {"TEETH-FALLOUT-OWN", "TEETH-FALLOUT-ONE", "TEETH-PULL-EXTERNAL"}, 1, 1, 1),
    "terminal": ({"loss-1", "firm-return-1"}, {"TEETH-END-TERMINAL"}, 1, 1, 1),
    "negated": ({"negated-loss-1"}, set(), 0, 0, 0),
    "multi_owner": ({"dreamer-loss-1", "sister-loss-1"}, {"TEETH-FALLOUT-OWN", "TEETH-FALLOUT-OTHER", "TEETH-FALLOUT-MULTIPLE"}, 2, 2, 2),
    "second_loss": ({"left-loss-1", "attempt-1", "second-loss-1"}, {"TEETH-FALLOUT-OWN", "TEETH-FALLOUT-MULTIPLE"}, 2, 1, 2),
}


class ClaimProvenanceReachabilityTests(unittest.TestCase):
    maxDiff = None

    def assert_case(self, name):
        decision = build_teeth_doctrine_context(CASES[name])
        graph = decision["context_graph"]
        expected_events, expected_rules, loss_count, owner_count, chain_count = EXPECTED[name]
        self.assertEqual(PROVENANCE_CONTRACT_VERSION, graph["provenance_contract_version"])
        self.assertEqual(expected_events, {event["event_id"] for event in graph["event_inventory"]})
        self.assertEqual(expected_rules, set(decision["applied_rule_ids"]))
        self.assertTrue(graph["integrity"]["verified"], graph["integrity"])
        self.assertTrue(graph["provenance_integrity"]["verified"], graph["provenance_integrity"])
        self.assertEqual([], graph["provenance_integrity"]["reason_codes"])
        self.assertTrue(graph["provenance_digest"].startswith("sha256:"))
        summary = graph["provenance_summary"]
        self.assertEqual(loss_count, summary["loss_path_count"])
        self.assertEqual(owner_count, summary["owner_path_count"])
        self.assertEqual(chain_count, summary["governing_chain_count"])
        self.assertEqual(0, summary["cross_owner_path_count"])
        for path in graph["provenance_paths"]:
            self.assertTrue(path["complete"], path)
            if path["disposition"] == "released":
                event = next(item for item in graph["event_inventory"] if item["event_id"] == path["event_id"])
                self.assertTrue(event["doctrine_eligible"])
                if path["rule_id"]:
                    rule = graph["provenance_rule_registry"][path["rule_id"]]
                    self.assertEqual("APPROVED", rule["status"])
                    self.assertTrue(rule["active"])
        if name == "terminal":
            unresolved = [path for path in graph["provenance_paths"] if path["rule_id"] == "TEETH-END-RETURNED-SAME"]
            self.assertEqual("withheld", unresolved[0]["disposition"])
            self.assertIsNone(unresolved[0]["claim_id_or_null"])
        if name in {"negated", "gums", "loose"}:
            negated = [path for path in graph["provenance_paths"] if path["event_id"] == "negated-loss-1"]
            self.assertEqual("gated", negated[0]["disposition"])

    def assert_reason(self, graph, reason):
        result = validate_claim_provenance(graph)
        self.assertFalse(result["verified"])
        self.assertIn(reason, result["reason_codes"])


def _case_test(name):
    def test(self):
        self.assert_case(name)
    return test


for _name in CASES:
    setattr(ClaimProvenanceReachabilityTests, f"test_path_{_name}", _case_test(_name))


def _graph(name="one"):
    return copy.deepcopy(build_teeth_doctrine_context(CASES[name])["context_graph"])


def test_mut_claim_path_missing(self):
    graph = _graph()
    graph["provenance_paths"] = []
    self.assert_reason(graph, "CLAIM_PATH_MISSING")


def test_mut_warning_incomplete(self):
    graph = _graph()
    graph["provenance_paths"][0]["complete"] = False
    self.assert_reason(graph, "WARNING_PATH_INCOMPLETE")


def test_mut_unverified_rule(self):
    graph = _graph()
    rule_id = graph["provenance_paths"][0]["rule_id"]
    graph["provenance_rule_registry"][rule_id]["active"] = False
    self.assert_reason(graph, "UNVERIFIED_RULE_IN_PATH")


def test_mut_cross_owner(self):
    graph = _graph()
    graph["provenance_paths"][0]["owner_id_or_ambiguous"] = "sister"
    self.assert_reason(graph, "CROSS_OWNER_PATH")


def test_mut_cross_chain(self):
    graph = _graph()
    graph["provenance_paths"][0]["event_chain_id_or_null"] = "chain-other"
    self.assert_reason(graph, "CROSS_CHAIN_PATH")


def test_mut_ineligible_released(self):
    graph = _graph("negated")
    graph["provenance_paths"][0]["disposition"] = "released"
    self.assert_reason(graph, "INELIGIBLE_EVENT_REACHES_CLAIM")


def test_mut_modifier_replaces_base(self):
    graph = _graph("pain")
    for path in graph["provenance_paths"]:
        if not path["modifier"]:
            path["warning_path"] = False
    self.assert_reason(graph, "MODIFIER_PATH_REPLACES_BASE")


def test_mut_terminal_stale(self):
    graph = _graph("terminal")
    path = next(path for path in graph["provenance_paths"] if path["terminal"])
    path["event_id"] = "loss-1"
    self.assert_reason(graph, "TERMINAL_PATH_STALE")


def test_mut_orphan_active_node(self):
    graph = _graph()
    graph["provenance_nodes"].append({
        "node_id": "candidate:orphan", "node_type": "candidate", "active": True,
        "owner_id_or_ambiguous": "dreamer", "event_chain_id_or_null": "chain-tooth-1",
    })
    self.assert_reason(graph, "ORPHAN_ACTIVE_NODE")


def test_mut_cycle(self):
    graph = _graph()
    claim_node = next(
        edge["to_node_id"] for edge in graph["provenance_edges"]
        if str(edge["to_node_id"]).startswith("claim:")
    )
    graph["provenance_edges"].append({
        "edge_id": "edge-cycle", "path_id": "path-cycle",
        "from_node_id": claim_node, "to_node_id": "event:loss-1",
        "owner_id_or_ambiguous": "dreamer", "event_chain_id_or_null": "chain-tooth-1",
    })
    self.assert_reason(graph, "PROVENANCE_CYCLE")


def test_mut_cardinality(self):
    graph = _graph("multiple")
    graph["provenance_summary"]["loss_path_count"] = 2
    self.assert_reason(graph, "PATH_CARDINALITY_MISMATCH")


def test_mut_digest(self):
    graph = _graph()
    graph["provenance_summary"]["released_path_count"] += 1
    self.assert_reason(graph, "GRAPH_DIGEST_MISMATCH")


for _name, _test in list(globals().items()):
    if _name.startswith("test_mut_"):
        setattr(ClaimProvenanceReachabilityTests, _name, _test)


if __name__ == "__main__":
    unittest.main()
