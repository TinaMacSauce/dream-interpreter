import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from flask import Flask

from app.config import Config
from app.routes.qa import TEETH_QA_CASES, qa_bp
from app.teeth_doctrine import (
    build_teeth_doctrine_context,
    build_teeth_narration_facts,
)
from scripts.verify_production_teeth_contract import validate as validate_production_contract


CASE_DREAMS = dict(TEETH_QA_CASES)


class TeethQAReleaseGateTests(unittest.TestCase):
    def doctrine(self, case_id):
        return build_teeth_doctrine_context(CASE_DREAMS[case_id])

    def narration(self, case_id):
        return build_teeth_narration_facts(CASE_DREAMS[case_id])

    def test_quantity_one_and_multiple_are_distinct(self):
        one = self.doctrine("quantity_one")
        multiple = self.doctrine("quantity_multiple")

        self.assertEqual("one_person", one["warning_count"])
        self.assertIn("TEETH-FALLOUT-ONE", one["applied_rule_ids"])
        self.assertNotIn("TEETH-FALLOUT-MULTIPLE", one["applied_rule_ids"])
        self.assertEqual("multiple_people", multiple["warning_count"])
        self.assertIn("TEETH-FALLOUT-MULTIPLE", multiple["applied_rule_ids"])
        self.assertNotIn("TEETH-FALLOUT-ONE", multiple["applied_rule_ids"])

        one_detail = " ".join(self.narration("quantity_one")["details"])
        multiple_detail = " ".join(self.narration("quantity_multiple")["details"])
        self.assertIn("this was your own tooth", one_detail.lower())
        self.assertNotIn("these were your own teeth", one_detail.lower())
        self.assertIn("these were your own teeth", multiple_detail.lower())
        self.assertNotIn("this was your own tooth", multiple_detail.lower())

    def test_ownership_is_bound_to_tooth_not_nearby_actor(self):
        other = self.doctrine("ownership_other")
        actor = self.doctrine("ownership_external_actor")

        self.assertEqual("other", other["owner"])
        self.assertEqual("sister", other["subject_scope"])
        self.assertIn("TEETH-FALLOUT-OTHER", other["applied_rule_ids"])
        self.assertEqual("dreamer", actor["owner"])
        self.assertEqual("other", actor["removal_actor"])
        self.assertEqual("external_interference", actor["pull_modifier"])
        self.assertIn("TEETH-FALLOUT-OWN", actor["applied_rule_ids"])

    def test_pain_and_painless_change_proximity_not_outcome_certainty(self):
        painful = self.doctrine("painful_loss")
        painless = self.doctrine("painless_loss")

        self.assertEqual("painful", painful["pain"])
        self.assertEqual("very_close_or_close_relative", painful["proximity"])
        self.assertEqual("heightened", painful["emotional_intensity"])
        self.assertIn("TEETH-MOD-PAIN", painful["applied_rule_ids"])
        self.assertEqual("painless", painless["pain"])
        self.assertEqual("friend_acquaintance_or_more_distant", painless["proximity"])
        self.assertEqual("", painless["emotional_intensity"])
        self.assertIn("TEETH-MOD-PAINLESS", painless["applied_rule_ids"])

    def test_blood_after_loss_is_severity_only(self):
        doctrine = self.doctrine("blood_after_loss")

        self.assertTrue(doctrine["blood_on_fallen_tooth"])
        self.assertEqual("increased", doctrine["severity_modifier"])
        self.assertIn("TEETH-MOD-BLOOD", doctrine["applied_rule_ids"])
        details = " ".join(self.narration("blood_after_loss")["details"]).lower()
        self.assertIn("emotional depth only", details)
        self.assertIn("does not determine", details)

    def test_bleeding_gums_respects_loose_and_loss_negations(self):
        doctrine = self.doctrine("bleeding_gums_with_negations")

        self.assertFalse(doctrine["active_fallout"])
        self.assertFalse(doctrine["loose_warning"])
        self.assertTrue(doctrine["bleeding_gums_warning"])
        self.assertEqual("bleeding_gums", doctrine["warning_kind"])
        self.assertEqual(["TEETH-OMEN-GUM-BLOOD"], doctrine["applied_rule_ids"])

    def test_loose_tooth_without_loss_is_sickness_warning_only(self):
        doctrine = self.doctrine("loose_without_loss")

        self.assertFalse(doctrine["active_fallout"])
        self.assertTrue(doctrine["loose_warning"])
        self.assertEqual("loose_sickness", doctrine["warning_kind"])
        self.assertEqual(["TEETH-STATE-LOOSE"], doctrine["applied_rule_ids"])

    def test_negated_and_hypothetical_loss_do_not_activate_fallout(self):
        for case_id in ("negated_loss", "hypothetical_loss"):
            with self.subTest(case_id=case_id):
                doctrine = self.doctrine(case_id)
                self.assertFalse(doctrine["active_doctrine"])
                self.assertFalse(doctrine["active_fallout"])
                self.assertEqual([], doctrine["applied_rule_ids"])

    def test_only_genuine_terminal_ending_seals(self):
        genuine = self.doctrine("genuine_terminal_ending")
        attempted = self.doctrine("attempted_ending")

        self.assertTrue(genuine["ending_precedence"])
        self.assertEqual("same_tooth_returned_firm", genuine["terminal_ending"])
        self.assertEqual("unresolved", genuine["outcome_resolution"])
        self.assertFalse(genuine["restoration_attempted"])
        self.assertEqual(["TEETH-END-TERMINAL"], genuine["applied_rule_ids"])
        self.assertIn("TEETH-END-RETURNED-SAME", genuine["unresolved_rule_ids"])
        self.assertFalse(attempted["ending_precedence"])
        self.assertEqual("", attempted["terminal_ending"])
        self.assertTrue(attempted["restoration_attempted"])
        self.assertIn("TEETH-FALLOUT-ONE", attempted["applied_rule_ids"])
        self.assertNotIn("TEETH-END-TERMINAL", attempted["applied_rule_ids"])

        genuine_text = " ".join(self.narration("genuine_terminal_ending")["details"]).lower()
        attempted_text = " ".join(self.narration("attempted_ending")["details"]).lower()
        self.assertNotIn("attempt to put the tooth back", genuine_text)
        self.assertIn("attempt to put the tooth back", attempted_text)
        self.assertIn("not treated as a completed restoration or terminal ending", attempted_text)

    def test_negated_reinsertion_attempt_is_not_recorded(self):
        doctrine = build_teeth_doctrine_context(
            "My tooth fell out, but I never tried to put it back."
        )

        self.assertTrue(doctrine["active_warning"])
        self.assertFalse(doctrine["restoration_attempted"])
        self.assertFalse(doctrine["ending_precedence"])

    def test_all_active_narration_is_culture_scoped_and_fear_safe(self):
        forbidden = ("will die", "is going to die", "will get sick", "definitely")
        for case_id in CASE_DREAMS:
            narration = self.narration(case_id)
            if not narration["active"]:
                continue
            text = " ".join([narration["lead"], *narration["details"]]).lower()
            with self.subTest(case_id=case_id):
                self.assertTrue(
                    "jamaican" in text
                    or "caribbean" in text
                    or "cultural consequence" in text
                )
                for phrase in forbidden:
                    self.assertNotIn(phrase, text)

    def test_public_contract_is_fixed_non_billable_and_versioned(self):
        app = Flask(__name__)
        app.register_blueprint(qa_bp)

        with patch.dict(os.environ, {"RENDER_GIT_COMMIT": "qa-route-sha"}, clear=False):
            response = app.test_client().get("/qa/teeth-regression")

        payload = response.get_json()
        self.assertEqual(200, response.status_code)
        self.assertEqual("teeth-qa-contract-v2", payload["contract_version"])
        self.assertEqual(len(TEETH_QA_CASES), payload["case_count"])
        self.assertEqual("qa-route-sha", payload["release"]["build_commit"])
        self.assertTrue(payload["doctrine_registry"]["verified"])
        self.assertEqual(23, payload["doctrine_registry"]["rule_count"])
        self.assertEqual(17, payload["doctrine_registry"]["active_rule_count"])
        self.assertEqual(6, payload["doctrine_registry"]["unresolved_rule_count"])
        self.assertEqual("no-store", response.headers["Cache-Control"])
        self.assertNotIn("Set-Cookie", response.headers)
        payload["doctrine_registry"]["loaded_from"] = "canonical_sheet"
        evidence = validate_production_contract(payload, expected_commit="qa-route-sha")
        self.assertTrue(evidence["verified"], evidence["errors"])

    def test_production_verifier_rejects_singular_plural_narration_drift(self):
        app = Flask(__name__)
        app.register_blueprint(qa_bp)

        with patch.dict(os.environ, {"RENDER_GIT_COMMIT": "qa-route-sha"}, clear=False):
            payload = app.test_client().get("/qa/teeth-regression").get_json()

        payload["doctrine_registry"]["loaded_from"] = "canonical_sheet"
        one = next(item for item in payload["cases"] if item["case_id"] == "quantity_one")
        one["narration"]["details"] = [
            "Because these were your own teeth, the warning concerns the relationship circle."
        ]
        evidence = validate_production_contract(payload, expected_commit="qa-route-sha")
        self.assertFalse(evidence["verified"])
        self.assertTrue(
            any("quantity_one: narration" in error for error in evidence["errors"]),
            evidence["errors"],
        )

    def test_production_verifier_rejects_missing_attempt_acknowledgement(self):
        app = Flask(__name__)
        app.register_blueprint(qa_bp)

        with patch.dict(os.environ, {"RENDER_GIT_COMMIT": "qa-route-sha"}, clear=False):
            payload = app.test_client().get("/qa/teeth-regression").get_json()

        payload["doctrine_registry"]["loaded_from"] = "canonical_sheet"
        attempted = next(
            item for item in payload["cases"] if item["case_id"] == "attempted_ending"
        )
        attempted["narration"]["details"] = []
        evidence = validate_production_contract(payload, expected_commit="qa-route-sha")
        self.assertFalse(evidence["verified"])
        self.assertTrue(
            any("attempted_ending: narration" in error for error in evidence["errors"]),
            evidence["errors"],
        )

    def test_qa_status_exposes_protected_non_billable_access_contract(self):
        app = Flask(__name__)
        app.register_blueprint(qa_bp)

        with TemporaryDirectory() as data_dir:
            qa_grants = str(Path(data_dir) / "qa_grants.json")
            with (
                patch.dict(
                    os.environ,
                    {"RENDER_GIT_COMMIT": "qa-status-sha"},
                    clear=False,
                ),
                patch.object(Config, "ADMIN_KEY", "configured-secret"),
                patch.object(Config, "QA_GRANTS_FILE", qa_grants),
            ):
                response = app.test_client().get("/qa/status")

        payload = response.get_json()
        access = payload["qa_access"]
        self.assertEqual(200, response.status_code)
        self.assertTrue(payload["ready"])
        self.assertEqual("qa-status-sha", payload["release"]["build_commit"])
        self.assertTrue(access["configured"])
        self.assertTrue(access["storage_ready"])
        self.assertEqual("/qa/interpret", access["interpret_route"])
        self.assertEqual("/interpret", access["application_route"])
        self.assertTrue(access["non_billable"])
        self.assertFalse(access["customer_credits_consumed"])
        self.assertFalse(access["customer_entitlement_store_used"])
        self.assertEqual("sha256_hash_only", access["token_storage"])
        self.assertTrue(access["revocable"])
        self.assertEqual("no-store", response.headers["Cache-Control"])

    def test_qa_interpret_route_requires_an_active_token(self):
        app = Flask(__name__)
        app.register_blueprint(qa_bp)

        response = app.test_client().post(
            "/qa/interpret",
            json={"dream": "My tooth fell out."},
        )

        payload = response.get_json()
        self.assertEqual(403, response.status_code)
        self.assertTrue(payload["blocked"])
        self.assertEqual("missing_token", payload["reason"])
        self.assertEqual("no-store", response.headers["Cache-Control"])
        self.assertNotIn("Set-Cookie", response.headers)


if __name__ == "__main__":
    unittest.main()
