import copy
import unittest

from app.teeth_doctrine import build_teeth_doctrine_context
from app.warning_claim_partition import (
    WARNING_CLAIM_PARTITION_CONTRACT_VERSION,
    validate_warning_claim_partition,
)


CASES = {
    "multi_owner_conditions": "My gums were bleeding. My sister's tooth was loose but did not fall out.",
    "same_owner_distinct_conditions": "My gums were bleeding and my tooth was loose but did not fall out.",
    "two_owner_loose": "My tooth was loose. My sister's tooth was loose.",
    "two_other_owners": "My sister's tooth was loose. My brother's tooth was loose.",
    "two_loose_one_event": "Two of my teeth were loose, but neither fell out.",
    "one_loss_multi_rule": "My tooth fell out.",
    "loss_plus_other_loose": "My tooth fell out. My sister's tooth was loose but did not fall out.",
    "gums_plus_other_loss": "My gums were bleeding. My sister's tooth fell out.",
    "loose_then_loss": "My tooth became loose, then it fell out.",
    "quoted_plus_gums": 'My aunt said, "My tooth is loose." My gums were bleeding.',
    "hypothetical_plus_other": "If my tooth were loose, I would worry. My sister's tooth was loose.",
    "negated_plus_other": "My tooth was not loose. My sister's tooth was loose.",
}


def graph(name):
    return copy.deepcopy(build_teeth_doctrine_context(CASES[name])["context_graph"])


def atomic_claims(value):
    return [
        claim for claim in value["claim_manifest"]
        if claim.get("claim_scope") == "atomic_warning"
    ]


class WarningClaimPartitionConservationTests(unittest.TestCase):
    maxDiff = None

    def assert_valid(self, value):
        self.assertEqual(
            WARNING_CLAIM_PARTITION_CONTRACT_VERSION,
            value["warning_claim_partition_contract_version"],
        )
        self.assertTrue(
            value["warning_claim_partition_integrity"]["verified"],
            value["warning_claim_partition_integrity"],
        )
        self.assertTrue(value["integrity"]["verified"], value["integrity"])

    def test_multi_owner_conditions_partition_by_owner_and_chain(self):
        value = graph("multi_owner_conditions")
        claims = atomic_claims(value)
        self.assert_valid(value)
        self.assertEqual(2, len(claims))
        self.assertEqual({("dreamer",), ("sister",)}, {tuple(c["owner_ids"]) for c in claims})
        self.assertEqual(2, len({tuple(c["event_chain_ids"]) for c in claims}))
        compound = next(c for c in value["claim_manifest"] if c.get("claim_scope") == "compound_presentation")
        self.assertEqual({c["claim_id"] for c in claims}, set(compound["member_claim_ids"]))

    def test_same_owner_distinct_conditions_partition_by_family(self):
        value = graph("same_owner_distinct_conditions")
        claims = atomic_claims(value)
        self.assert_valid(value)
        self.assertEqual({"bleeding_gums", "loose_sickness"}, {c["warning_family"] for c in claims})
        self.assertEqual(2, len({tuple(c["event_chain_ids"]) for c in claims}))

    def test_two_owner_loose_splits_rule_records(self):
        value = graph("two_owner_loose")
        self.assert_valid(value)
        records = [r for r in value["rule_sets"]["public_applied"] if r["rule_id"] == "TEETH-STATE-LOOSE"]
        self.assertEqual(2, len(records))
        self.assertTrue(all(len(r["source_event_ids"]) == 1 for r in records))

    def test_two_other_owners_stay_independent(self):
        value = graph("two_other_owners")
        claims = atomic_claims(value)
        self.assert_valid(value)
        self.assertEqual({("sister",), ("brother",)}, {tuple(c["owner_ids"]) for c in claims})

    def test_two_loose_teeth_preserve_one_cardinality_two_event(self):
        value = graph("two_loose_one_event")
        self.assert_valid(value)
        loose = next(e for e in value["event_inventory"] if e["event_type"] == "loose_tooth_condition")
        self.assertEqual(2, loose["quantity_cardinality"])
        self.assertEqual(1, len(atomic_claims(value)))
        rules = {r["rule_id"] for r in value["rule_sets"]["public_applied"]}
        self.assertEqual({"TEETH-STATE-LOOSE"}, rules)

    def test_one_loss_keeps_compatible_rules_in_one_atomic_claim(self):
        value = graph("one_loss_multi_rule")
        self.assert_valid(value)
        claim = atomic_claims(value)[0]
        self.assertEqual({"TEETH-FALLOUT-OWN", "TEETH-FALLOUT-ONE"}, set(claim["consumed_rule_ids"]))

    def test_loss_does_not_omit_other_owner_loose_warning(self):
        value = graph("loss_plus_other_loose")
        self.assert_valid(value)
        self.assertEqual({"tooth_loss", "loose_sickness"}, {c["warning_family"] for c in atomic_claims(value)})

    def test_other_owner_loss_does_not_omit_gum_warning(self):
        value = graph("gums_plus_other_loss")
        self.assert_valid(value)
        self.assertEqual({"tooth_loss", "bleeding_gums"}, {c["warning_family"] for c in atomic_claims(value)})

    def test_later_same_chain_loss_makes_loose_event_historical(self):
        value = graph("loose_then_loss")
        self.assert_valid(value)
        self.assertEqual(["tooth_loss"], [c["warning_family"] for c in atomic_claims(value)])
        dispositions = {item["event_id"]: item["disposition"] for item in value["warning_claim_dispositions"]}
        self.assertEqual("historical", dispositions["loose-1"])
        self.assertEqual("released", dispositions["loss-1"])

    def test_quoted_loose_event_is_gated(self):
        value = graph("quoted_plus_gums")
        self.assert_valid(value)
        claim = atomic_claims(value)[0]
        self.assertEqual("bleeding_gums", claim["warning_family"])
        self.assertNotIn("quoted-loose-1", claim["consumed_event_ids"])

    def test_hypothetical_event_is_not_consumed(self):
        value = graph("hypothetical_plus_other")
        self.assert_valid(value)
        claim = atomic_claims(value)[0]
        self.assertEqual(["sister"], claim["owner_ids"])
        self.assertNotIn("hypothetical-loose-1", claim["consumed_event_ids"])

    def test_negated_event_is_not_consumed(self):
        value = graph("negated_plus_other")
        self.assert_valid(value)
        claim = atomic_claims(value)[0]
        self.assertEqual(["sister"], claim["owner_ids"])
        self.assertNotIn("negated-loose-1", claim["consumed_event_ids"])


class WarningClaimPartitionMutationTests(unittest.TestCase):
    def assert_rejected(self, value, reason):
        result = validate_warning_claim_partition(value)
        self.assertFalse(result["verified"])
        self.assertIn(reason, result["reason_codes"])

    def test_rejects_atomic_owner_mix(self):
        value = graph("multi_owner_conditions")
        claims = atomic_claims(value)
        claims[0]["consumed_event_ids"].extend(claims[1]["consumed_event_ids"])
        claims[0]["consumed_span_ids"].extend(claims[1]["consumed_span_ids"])
        self.assert_rejected(value, "ATOMIC_CLAIM_OWNER_MIX")

    def test_rejects_atomic_chain_mix(self):
        value = graph("same_owner_distinct_conditions")
        claims = atomic_claims(value)
        claims[0]["consumed_event_ids"].extend(claims[1]["consumed_event_ids"])
        claims[0]["consumed_span_ids"].extend(claims[1]["consumed_span_ids"])
        self.assert_rejected(value, "ATOMIC_CLAIM_CHAIN_MIX")

    def test_rejects_atomic_family_mix(self):
        value = graph("same_owner_distinct_conditions")
        claims = atomic_claims(value)
        claims[0]["consumed_event_ids"].extend(claims[1]["consumed_event_ids"])
        claims[0]["consumed_span_ids"].extend(claims[1]["consumed_span_ids"])
        self.assert_rejected(value, "ATOMIC_CLAIM_FAMILY_MIX")

    def test_rejects_eligible_warning_omission(self):
        value = graph("loss_plus_other_loose")
        removed = atomic_claims(value)[1]["claim_id"]
        value["claim_manifest"] = [c for c in value["claim_manifest"] if c["claim_id"] != removed]
        self.assert_rejected(value, "ELIGIBLE_WARNING_EVENT_OMITTED")

    def test_rejects_ineligible_event_consumption(self):
        value = graph("hypothetical_plus_other")
        claim = atomic_claims(value)[0]
        event = next(e for e in value["event_inventory"] if e["event_id"] == "hypothetical-loose-1")
        claim["consumed_event_ids"].append(event["event_id"])
        claim["consumed_span_ids"].append(event["source_span"]["span_id"])
        self.assert_rejected(value, "INELIGIBLE_EVENT_CONSUMED")

    def test_rejects_unsplit_rule_source(self):
        value = graph("two_owner_loose")
        records = [r for r in value["rule_sets"]["public_applied"] if r["rule_id"] == "TEETH-STATE-LOOSE"]
        records[0]["source_event_ids"].extend(records[1]["source_event_ids"])
        records[0]["source_span_ids"].extend(records[1]["source_span_ids"])
        self.assert_rejected(value, "RULE_SOURCE_PARTITION_MISSING")

    def test_rejects_missing_compound_member(self):
        value = graph("multi_owner_conditions")
        compound = next(c for c in value["claim_manifest"] if c.get("claim_scope") == "compound_presentation")
        compound["member_claim_ids"].pop()
        self.assert_rejected(value, "COMPOUND_MEMBER_MISSING")

    def test_rejects_compound_raw_source_consumption(self):
        value = graph("multi_owner_conditions")
        compound = next(c for c in value["claim_manifest"] if c.get("claim_scope") == "compound_presentation")
        compound["consumed_event_ids"] = ["gum-bleeding-1"]
        self.assert_rejected(value, "COMPOUND_RAW_SOURCE_CONSUMPTION")

    def test_rejects_event_rule_mismatch(self):
        value = graph("one_loss_multi_rule")
        atomic_claims(value)[0]["consumed_rule_ids"] = ["TEETH-STATE-LOOSE"]
        self.assert_rejected(value, "CLAIM_EVENT_RULE_MISMATCH")

    def test_rejects_claim_span_mismatch(self):
        value = graph("multi_owner_conditions")
        atomic_claims(value)[0]["consumed_span_ids"] = ["span-sister-loose-1"]
        self.assert_rejected(value, "CLAIM_SPAN_MISMATCH")

    def test_rejects_historical_event_reactivation(self):
        value = graph("loose_then_loss")
        claim = atomic_claims(value)[0]
        event = next(e for e in value["event_inventory"] if e["event_id"] == "loose-1")
        claim["consumed_event_ids"].append("loose-1")
        claim["consumed_span_ids"].append(event["source_span"]["span_id"])
        self.assert_rejected(value, "HISTORICAL_EVENT_REACTIVATED")

    def test_rejects_claim_cardinality_drift(self):
        value = graph("multi_owner_conditions")
        value["warning_claim_partition_summary"]["atomic_claim_count"] = 1
        self.assert_rejected(value, "CLAIM_CARDINALITY_DRIFT")


if __name__ == "__main__":
    unittest.main()
