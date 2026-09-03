import unittest
from unittest.mock import patch

from app.teeth_doctrine import build_teeth_doctrine_context
from app.teeth_provenance import (
    NON_GUARANTEE_POLICY,
    SOURCE_LAYER,
    TeethProvenanceValidationError,
    mutated_provenance,
    validate_teeth_rule_provenance,
)
from app.teeth_registry import EXPECTED_CONTENT_REVISION, EXPECTED_DOCTRINE_VERSION


REQUIRED_BINDING_FIELDS = {
    "rule_id",
    "registry_implementation_key",
    "status",
    "active",
    "source_layer",
    "source_event_ids",
    "source_entity_ids",
    "source_spans",
    "gate_results",
    "candidate_disposition",
    "warning_role",
    "safety_boundary",
}

REQUIRED_WARNING_FIELDS = {
    "warning_id",
    "source_rule_ids",
    "modifier_rule_ids",
    "source_event_ids",
    "source_entity_ids",
    "source_spans",
    "owner_id",
    "warning_count",
    "doctrine_version",
    "registry_contract_version",
    "sheet_revision",
    "content_revision",
    "certainty_policy",
    "safety_boundaries",
    "release_status",
}


class TeethCTX003RuleProvenanceTests(unittest.TestCase):
    def doctrine(self, dream):
        return build_teeth_doctrine_context(dream)

    def assert_binding_contract(self, doctrine):
        provenance = doctrine["rule_provenance"]
        self.assertTrue(provenance["registry_gate"]["passed"])
        self.assertEqual(EXPECTED_CONTENT_REVISION, provenance["registry_identity"]["content_revision"])
        self.assertEqual(EXPECTED_DOCTRINE_VERSION, provenance["registry_identity"]["decision_id"])
        for binding in provenance["rule_bindings"]:
            self.assertTrue(REQUIRED_BINDING_FIELDS <= set(binding))
            self.assertEqual(SOURCE_LAYER, binding["source_layer"])
            self.assertTrue(binding["source_event_ids"])
            self.assertTrue(binding["source_entity_ids"])
            self.assertTrue(all(binding["source_spans"]))
        for warning in provenance["warning_provenance"]:
            self.assertTrue(REQUIRED_WARNING_FIELDS <= set(warning))
            self.assertEqual(NON_GUARANTEE_POLICY, warning["certainty_policy"])

    def binding_map(self, doctrine):
        self.assert_binding_contract(doctrine)
        return {
            binding["rule_id"]: binding
            for binding in doctrine["rule_provenance"]["rule_bindings"]
        }

    def warning(self, doctrine):
        warnings = doctrine["rule_provenance"]["warning_provenance"]
        self.assertEqual(1, len(warnings))
        return warnings[0]

    def test_registry_one_binds_base_count_event_and_spans(self):
        doctrine = self.doctrine("My tooth fell out.")
        bindings = self.binding_map(doctrine)
        warning = self.warning(doctrine)
        self.assertEqual(["TEETH-FALLOUT-OWN", "TEETH-FALLOUT-ONE"], doctrine["applied_rule_ids"])
        self.assertEqual(["My tooth fell out"], bindings["TEETH-FALLOUT-OWN"]["source_spans"])
        self.assertEqual(["tooth"], bindings["TEETH-FALLOUT-ONE"]["source_spans"])
        self.assertEqual(["TEETH-FALLOUT-OWN"], warning["source_rule_ids"])
        self.assertEqual(["TEETH-FALLOUT-ONE"], warning["modifier_rule_ids"])
        self.assertEqual("one_person", warning["warning_count"])

    def test_registry_multiple_binds_multiple_without_named_arithmetic(self):
        doctrine = self.doctrine("Three of my teeth fell out.")
        bindings = self.binding_map(doctrine)
        warning = self.warning(doctrine)
        self.assertEqual(["TEETH-FALLOUT-OWN", "TEETH-FALLOUT-MULTIPLE"], doctrine["applied_rule_ids"])
        self.assertEqual(["Three of my teeth"], bindings["TEETH-FALLOUT-MULTIPLE"]["source_spans"])
        self.assertEqual("multiple_people", warning["warning_count"])
        self.assertEqual("released_non_guaranteed", warning["release_status"])

    def test_registry_pain_is_modifier_not_base_or_certainty(self):
        doctrine = self.doctrine("My tooth fell out and it hurt badly.")
        bindings = self.binding_map(doctrine)
        warning = self.warning(doctrine)
        self.assertEqual(["hurt badly"], bindings["TEETH-MOD-PAIN"]["source_spans"])
        self.assertEqual(["TEETH-FALLOUT-OWN"], warning["source_rule_ids"])
        self.assertEqual(["TEETH-FALLOUT-ONE", "TEETH-MOD-PAIN"], warning["modifier_rule_ids"])
        self.assertEqual(NON_GUARANTEE_POLICY, warning["certainty_policy"])

    def test_registry_painless_is_modifier_not_base_or_certainty(self):
        doctrine = self.doctrine("My tooth fell out without pain.")
        bindings = self.binding_map(doctrine)
        warning = self.warning(doctrine)
        self.assertEqual(["without pain"], bindings["TEETH-MOD-PAINLESS"]["source_spans"])
        self.assertEqual(["TEETH-FALLOUT-ONE", "TEETH-MOD-PAINLESS"], warning["modifier_rule_ids"])
        self.assertEqual("dreamer", warning["owner_id"])

    def test_registry_gums_releases_only_gum_rule_and_records_negations(self):
        doctrine = self.doctrine("My gums were bleeding, but no tooth was loose and none fell out.")
        bindings = self.binding_map(doctrine)
        warning = self.warning(doctrine)
        dispositions = doctrine["rule_provenance"]["candidate_dispositions"]
        rejected = {item["rule_id"] for item in dispositions if item["disposition"] == "rejected_negated"}
        self.assertEqual(["TEETH-OMEN-GUM-BLOOD"], doctrine["applied_rule_ids"])
        self.assertEqual(["My gums were bleeding"], bindings["TEETH-OMEN-GUM-BLOOD"]["source_spans"])
        self.assertEqual({"TEETH-STATE-LOOSE", "TEETH-FALLOUT-OWN"}, rejected)
        self.assertEqual([], warning["modifier_rule_ids"])

    def test_registry_fallen_tooth_blood_stays_modifier_only(self):
        doctrine = self.doctrine("My tooth fell out and there was blood on the fallen tooth.")
        bindings = self.binding_map(doctrine)
        warning = self.warning(doctrine)
        self.assertEqual(["blood on the fallen tooth"], bindings["TEETH-MOD-BLOOD"]["source_spans"])
        self.assertNotIn("TEETH-MOD-BLOOD", warning["source_rule_ids"])
        self.assertIn("TEETH-MOD-BLOOD", warning["modifier_rule_ids"])
        self.assertEqual(NON_GUARANTEE_POLICY, warning["certainty_policy"])

    def test_registry_other_owner_remains_warning_owner(self):
        doctrine = self.doctrine("My sister's tooth fell out.")
        bindings = self.binding_map(doctrine)
        warning = self.warning(doctrine)
        self.assertEqual(["TEETH-FALLOUT-OTHER", "TEETH-FALLOUT-ONE"], doctrine["applied_rule_ids"])
        self.assertEqual(["My sister's tooth fell out"], bindings["TEETH-FALLOUT-OTHER"]["source_spans"])
        self.assertEqual("sister", warning["owner_id"])
        self.assertNotIn("dreamer", warning["source_entity_ids"])

    def test_registry_external_actor_does_not_replace_owner(self):
        doctrine = self.doctrine("My sister pulled my tooth out.")
        bindings = self.binding_map(doctrine)
        warning = self.warning(doctrine)
        self.assertEqual(["My sister pulled my tooth out"], bindings["TEETH-PULL-EXTERNAL"]["source_spans"])
        self.assertEqual("dreamer", warning["owner_id"])
        self.assertIn("sister", warning["source_entity_ids"])
        self.assertIn("TEETH-PULL-EXTERNAL", warning["modifier_rule_ids"])

    def test_registry_terminal_ending_preserves_loss_and_withholds_consequence(self):
        doctrine = self.doctrine(
            "My tooth fell out, then the same tooth returned firmly to the same socket."
        )
        bindings = self.binding_map(doctrine)
        self.assertEqual(["TEETH-FALLOUT-OWN", "TEETH-FALLOUT-ONE"], doctrine["applied_rule_ids"])
        self.assertEqual(["TEETH-END-TERMINAL"], doctrine["structural_rule_ids"])
        self.assertEqual(["TEETH-END-RETURNED-SAME"], doctrine["unresolved_rule_ids"])
        self.assertEqual("withheld_unresolved", bindings["TEETH-END-RETURNED-SAME"]["candidate_disposition"])
        self.assertEqual([], doctrine["rule_provenance"]["warning_provenance"])
        self.assertEqual("withheld_unresolved", doctrine["rule_provenance"]["release_status"])

    def test_registry_position_is_structural_and_unresolved(self):
        doctrine = self.doctrine("My upper tooth fell out.")
        bindings = self.binding_map(doctrine)
        self.assertEqual(["TEETH-POSITION-MAP"], doctrine["unresolved_rule_ids"])
        self.assertEqual(["upper tooth"], bindings["TEETH-POSITION-MAP"]["source_spans"])
        self.assertEqual("withheld_unresolved", bindings["TEETH-POSITION-MAP"]["candidate_disposition"])
        self.assertEqual(["upper"], doctrine["positions"])

    def test_registry_state_fallout_collision_has_no_invented_winner(self):
        doctrine = self.doctrine("My broken tooth fell out.")
        bindings = self.binding_map(doctrine)
        self.assertEqual(["TEETH-COMBO-STATE-FALLOUT"], doctrine["unresolved_rule_ids"])
        self.assertEqual(["broken tooth fell out"], bindings["TEETH-COMBO-STATE-FALLOUT"]["source_spans"])
        self.assertEqual("withheld_unresolved", bindings["TEETH-COMBO-STATE-FALLOUT"]["candidate_disposition"])

    def test_registry_drift_fails_closed_without_rule_or_warning_release(self):
        failed = {
            "verified": False,
            "content_revision": "fnv1a64:wrong",
            "doctrine_version": EXPECTED_DOCTRINE_VERSION,
            "decision_id": EXPECTED_DOCTRINE_VERSION,
            "rules": {},
            "active_rule_ids": [],
            "unresolved_rule_ids": [],
            "error": "registry_content_revision_mismatch",
        }
        with patch("app.teeth_doctrine.get_teeth_registry_snapshot", return_value=failed):
            doctrine = self.doctrine("My tooth fell out.")
        provenance = doctrine["rule_provenance"]
        self.assertFalse(provenance["registry_gate"]["passed"])
        self.assertEqual([], doctrine["applied_rule_ids"])
        self.assertEqual([], provenance["rule_bindings"])
        self.assertEqual([], provenance["warning_provenance"])
        self.assertEqual(["REGISTRY_CONTENT_REVISION_MISMATCH"], provenance["registry_gate"]["reason_codes"])
        self.assertEqual("withheld_registry_unverified", provenance["release_status"])


class TeethCTX003UnsafeMutationTests(unittest.TestCase):
    def setUp(self):
        self.baseline = build_teeth_doctrine_context(
            "My tooth fell out and it hurt badly."
        )["rule_provenance"]

    def assert_rejected(self, provenance, code):
        with self.assertRaises(TeethProvenanceValidationError) as raised:
            validate_teeth_rule_provenance(provenance)
        self.assertEqual(code, raised.exception.code)

    def test_rejects_warning_release_after_registry_failure(self):
        value = mutated_provenance(self.baseline)
        value["registry_identity"]["verified"] = False
        self.assert_rejected(value, "UNVERIFIED_REGISTRY_FAIL_CLOSED")

    def test_rejects_registry_content_drift(self):
        value = mutated_provenance(self.baseline)
        value["registry_identity"]["content_revision"] = "fnv1a64:wrong"
        self.assert_rejected(value, "REGISTRY_CONTENT_REVISION_MISMATCH")

    def test_rejects_registry_activation_drift(self):
        value = mutated_provenance(self.baseline)
        active = value["registry_identity"]["active_rule_ids"]
        unresolved = value["registry_identity"]["unresolved_rule_ids"]
        unresolved.remove("TEETH-POSITION-MAP")
        active.append("TEETH-POSITION-MAP")
        self.assert_rejected(value, "REGISTRY_ACTIVATION_DRIFT")

    def test_rejects_registry_rule_count_drift(self):
        value = mutated_provenance(self.baseline)
        value["registry_identity"]["active_rule_ids"].pop()
        self.assert_rejected(value, "REGISTRY_RULE_COUNT_MISMATCH")

    def test_rejects_unresolved_rule_as_applied(self):
        value = build_teeth_doctrine_context("My upper tooth fell out.")["rule_provenance"]
        value = mutated_provenance(value)
        pending = next(item for item in value["rule_bindings"] if item["rule_id"] == "TEETH-POSITION-MAP")
        pending["candidate_disposition"] = "applied"
        pending["active"] = True
        self.assert_rejected(value, "UNRESOLVED_RULE_APPLIED")

    def test_rejects_unknown_rule_id(self):
        value = mutated_provenance(self.baseline)
        value["rule_bindings"][0]["rule_id"] = "TEETH-UNKNOWN"
        self.assert_rejected(value, "UNKNOWN_RULE_ID")

    def test_rejects_eventless_rule_binding(self):
        value = mutated_provenance(self.baseline)
        value["rule_bindings"][0]["source_event_ids"] = []
        self.assert_rejected(value, "RULE_EVENT_PROVENANCE_MISSING")

    def test_rejects_spanless_rule_binding(self):
        value = mutated_provenance(self.baseline)
        value["rule_bindings"][0]["source_spans"] = []
        self.assert_rejected(value, "RULE_SPAN_PROVENANCE_MISSING")

    def test_rejects_wrong_source_layer(self):
        value = mutated_provenance(self.baseline)
        value["rule_bindings"][0]["source_layer"] = "universal_fact"
        self.assert_rejected(value, "RULE_SOURCE_LAYER_MISMATCH")

    def test_rejects_modifier_as_base_warning(self):
        value = mutated_provenance(self.baseline)
        value["warning_provenance"][0]["source_rule_ids"] = ["TEETH-MOD-PAIN"]
        self.assert_rejected(value, "WARNING_BASE_MODIFIER_CONFUSION")

    def test_rejects_predictive_certainty_from_registry(self):
        value = mutated_provenance(self.baseline)
        value["warning_provenance"][0]["certainty_policy"] = "guaranteed_death"
        self.assert_rejected(value, "REGISTRY_NOT_PREDICTIVE_CERTAINTY")

    def test_rejects_stale_decision_binding(self):
        value = mutated_provenance(self.baseline)
        value["registry_identity"]["decision_id"] = "DEC-TEETH-2026-09-02-04"
        self.assert_rejected(value, "REGISTRY_DECISION_MISMATCH")


if __name__ == "__main__":
    unittest.main()
