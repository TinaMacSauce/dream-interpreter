import os
import unittest
from unittest.mock import patch

from flask import Flask

from app.routes.qa import TEETH_QA_CASES, qa_bp
from app.teeth_doctrine import (
    build_teeth_doctrine_context,
    build_teeth_narration_facts,
)


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
        self.assertEqual(["TEETH-END-TERMINAL"], genuine["applied_rule_ids"])
        self.assertIn("TEETH-END-RETURNED-SAME", genuine["unresolved_rule_ids"])
        self.assertFalse(attempted["ending_precedence"])
        self.assertEqual("", attempted["terminal_ending"])
        self.assertIn("TEETH-FALLOUT-ONE", attempted["applied_rule_ids"])

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
        self.assertEqual("teeth-qa-contract-v1", payload["contract_version"])
        self.assertEqual(len(TEETH_QA_CASES), payload["case_count"])
        self.assertEqual("qa-route-sha", payload["release"]["build_commit"])
        self.assertEqual("no-store", response.headers["Cache-Control"])
        self.assertNotIn("Set-Cookie", response.headers)


if __name__ == "__main__":
    unittest.main()
