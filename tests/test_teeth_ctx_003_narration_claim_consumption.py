import copy
import unittest

from app.narration_claim_consumption import (
    NARRATION_CLAIM_CONTRACT_VERSION,
    validate_narration_claim_consumption,
)
from app.teeth_doctrine import build_teeth_doctrine_context, build_teeth_narration_facts


CASES = (
    ("My gums were bleeding. My sister's tooth was loose but did not fall out.", 2, {"bleeding_gums", "loose_sickness"}),
    ("My gums were bleeding and my tooth was loose but did not fall out.", 2, {"bleeding_gums", "loose_sickness"}),
    ("My tooth was loose. My sister's tooth was loose.", 2, {"loose_sickness"}),
    ("My tooth fell out. My sister's tooth was loose but did not fall out.", 2, {"tooth_loss", "loose_sickness"}),
    ("My gums were bleeding. My sister's tooth fell out.", 2, {"bleeding_gums", "tooth_loss"}),
    ('My aunt said, "My tooth is loose." My gums were bleeding.', 1, {"bleeding_gums"}),
    ("If my tooth were loose, I would worry. My sister's tooth was loose.", 1, {"loose_sickness"}),
    ("My tooth was not loose. My sister's tooth was loose.", 1, {"loose_sickness"}),
    ("My tooth became loose, then it fell out.", 1, {"tooth_loss"}),
    ("My tooth fell out and I tried to put it back. Then another tooth fell out.", 1, {"tooth_loss"}),
    ("My tooth fell out, then the same tooth fitted firmly back into the same socket.", 0, set()),
    ("My tooth fell out.", 1, {"tooth_loss"}),
)


def _value(dream):
    doctrine = build_teeth_doctrine_context(dream)
    narration = build_teeth_narration_facts(dream)
    return doctrine["context_graph"], narration


class NarrationClaimConsumptionRegressionTests(unittest.TestCase):
    def test_all_twelve_fixture_cases_preserve_claim_boundaries(self):
        for dream, expected_count, expected_families in CASES:
            with self.subTest(dream=dream):
                graph, narration = _value(dream)
                self.assertEqual(NARRATION_CLAIM_CONTRACT_VERSION, narration["contract_version"])
                self.assertEqual({"verified": True, "reason_codes": []}, narration["narration_integrity"])
                members = [c for c in narration["narration_clauses"] if c["clause_type"] == "warning_member"]
                self.assertEqual(expected_count, len(members))
                self.assertEqual(expected_families, {c["warning_family_or_null"] for c in members})
                released = [c for c in graph["claim_manifest"] if c.get("claim_scope") == "atomic_warning"]
                self.assertEqual(
                    {c["claim_id"] for c in released},
                    {claim_id for c in members for claim_id in c["member_claim_ids"]},
                )
                for clause in narration["narration_clauses"]:
                    start, end = clause["output_span"]["start"], clause["output_span"]["end"]
                    self.assertEqual(clause["rendered_text"], narration["narration_text"][start:end])

    def test_compound_summary_never_replaces_member_clauses(self):
        _, narration = _value(CASES[0][0])
        compound = next(c for c in narration["narration_clauses"] if c["clause_type"] == "compound_summary")
        self.assertEqual(2, len(compound["member_claim_ids"]))
        self.assertIn("tradition_attribution", compound["safety_qualifier_ids"])

    def test_attempt_and_terminal_clauses_are_structural_only(self):
        _, attempt = _value(CASES[9][0])
        attempt_clause = next(c for c in attempt["narration_clauses"] if c["clause_type"] == "attempt_structure")
        self.assertTrue(all(not claim.startswith("claim-warning") for claim in attempt_clause["member_claim_ids"]))
        _, terminal = _value(CASES[10][0])
        types = [c["clause_type"] for c in terminal["narration_clauses"]]
        self.assertEqual(["terminal_structure"], types)


class NarrationClaimConsumptionMutationTests(unittest.TestCase):
    def setUp(self):
        self.graph, self.payload = _value(CASES[0][0])
        self.payload = {
            "narration_text": self.payload["narration_text"],
            "narration_clauses": copy.deepcopy(self.payload["narration_clauses"]),
        }
        self.graph = copy.deepcopy(self.graph)

    def assertRejected(self, reason):
        result = validate_narration_claim_consumption(self.payload, self.graph)
        self.assertFalse(result["verified"])
        self.assertIn(reason, result["reason_codes"])

    def test_orphan_claim_rejected(self):
        self.payload["narration_clauses"][0]["member_claim_ids"] = ["claim-missing"]
        self.assertRejected("NARRATION_CLAIM_ORPHAN")

    def test_raw_source_consumption_rejected(self):
        self.payload["narration_clauses"][0]["consumed_event_ids"] = ["gum-bleeding-1"]
        self.assertRejected("NARRATION_RAW_SOURCE_CONSUMPTION")

    def test_missing_atomic_clause_rejected(self):
        self.payload["narration_clauses"] = self.payload["narration_clauses"][1:]
        self.assertRejected("RELEASED_CLAIM_NOT_NARRATED")

    def test_duplicate_atomic_clause_rejected(self):
        duplicate = copy.deepcopy(self.payload["narration_clauses"][0])
        duplicate["narration_clause_id"] = "duplicate"
        self.payload["narration_clauses"].append(duplicate)
        self.assertRejected("CLAIM_NARRATED_MULTIPLE_TIMES")

    def test_owner_drift_rejected(self):
        self.payload["narration_clauses"][1]["owner_ids"] = ["dreamer"]
        self.assertRejected("NARRATION_OWNER_DRIFT")

    def test_chain_drift_rejected(self):
        self.payload["narration_clauses"][0]["event_chain_ids"] = ["chain-other"]
        self.assertRejected("NARRATION_CHAIN_DRIFT")

    def test_family_drift_rejected(self):
        self.payload["narration_clauses"][0]["warning_family_or_null"] = "loose_sickness"
        self.assertRejected("NARRATION_WARNING_FAMILY_DRIFT")

    def test_gated_branch_rejected(self):
        self.graph["claim_manifest"].append({
            "claim_id": "claim-gated", "claim_scope": "atomic_warning",
            "released": False, "release_status": "gated",
        })
        self.payload["narration_clauses"][0]["member_claim_ids"] = ["claim-gated"]
        self.assertRejected("GATED_BRANCH_NARRATED")

    def test_compound_hidden_member_rejected(self):
        self.payload["narration_clauses"] = [
            c for c in self.payload["narration_clauses"]
            if c.get("member_claim_ids") != ["claim-warning-2"]
        ]
        self.assertRejected("COMPOUND_MEMBER_CLAUSE_HIDDEN")

    def test_output_span_mismatch_rejected(self):
        self.payload["narration_clauses"][0]["rendered_text"] += " changed"
        self.assertRejected("NARRATION_OUTPUT_SPAN_MISMATCH")

    def test_safety_qualifier_drop_rejected(self):
        self.payload["narration_clauses"][1]["safety_qualifier_ids"] = []
        self.assertRejected("NARRATION_SAFETY_QUALIFIER_DROPPED")

    def test_structural_history_reactivation_rejected(self):
        self.payload["narration_clauses"][0]["clause_type"] = "terminal_structure"
        self.assertRejected("STRUCTURAL_HISTORY_REACTIVATED")


if __name__ == "__main__":
    unittest.main()
