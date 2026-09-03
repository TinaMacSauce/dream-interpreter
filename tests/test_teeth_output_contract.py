import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from app import create_app
from app.config import Config
from app.qa_access import issue_qa_grant
from app.services.narration_service import build_doctrine_bound_summary
from app.teeth_integration import (
    attach_teeth_narration_facts,
    bind_teeth_output_contract,
    build_teeth_output_assessment,
)


class TeethOutputContractTests(unittest.TestCase):
    def _active_facts(self, dream: str):
        return attach_teeth_narration_facts(
            dream,
            {
                "risk": "Low",
                "event_context": {
                    "primary_action": {
                        "name": "fell out",
                        "meaning": "generic emotional loss",
                    },
                    "primary_subject": "Teeth",
                    "primary_ending": {},
                },
            },
        )

    def test_active_teeth_suppresses_ambiguous_risk_and_separates_assessment(self):
        facts = self._active_facts("My tooth fell out.")

        seal, interpretation, full = bind_teeth_output_contract(
            doctrine_facts=facts,
            seal={"status": "Live", "type": "Focused", "risk": "Low"},
            interpretation={
                "spiritual_meaning": "Generic emotional loss.",
                "effects_in_physical_realm": "This may reflect emotional loss.",
                "what_to_do": "Reflect and pray without panic.",
            },
            full_interpretation="Generic duplicated output.",
        )

        self.assertEqual("", seal["risk"])
        self.assertEqual("", seal["risk_label"])
        self.assertTrue(seal["legacy_risk_suppressed"])
        assessment = seal["warning_assessment"]
        self.assertTrue(assessment["warning_present"])
        self.assertEqual("not_scaled", assessment["warning_severity"])
        self.assertEqual("approved_rule_match", assessment["interpretation_confidence"])
        self.assertEqual("none", assessment["predictive_certainty"])

        self.assertIn("one fallen tooth", interpretation["spiritual_meaning"])
        self.assertIn("tradition-based guidance", interpretation["effects_in_physical_realm"])
        self.assertNotIn("generic emotional loss", full.lower())
        self.assertNotIn("low risk", full.lower())
        self.assertNotIn("as this may", full.lower())

    def test_blood_changes_warning_severity_without_confidence_or_certainty(self):
        facts = self._active_facts(
            "My tooth fell out and there was blood on the fallen tooth."
        )

        assessment = build_teeth_output_assessment(facts["teeth_narration"])

        self.assertEqual("heightened", assessment["warning_severity"])
        self.assertEqual("approved_rule_match", assessment["interpretation_confidence"])
        self.assertEqual("none", assessment["predictive_certainty"])

    def test_narration_uses_only_concise_approved_teeth_parts(self):
        facts = self._active_facts("One of my teeth fell out without pain.")

        summary = build_doctrine_bound_summary(
            facts,
            {
                "spiritual_meaning": "This can show up as this may reflect emotional loss.",
                "effects_in_physical_realm": "Duplicated emotional loss wording.",
                "what_to_do": "Pray.",
            },
        )

        self.assertIn("one fallen tooth", summary.lower())
        self.assertIn("painless loss", summary.lower())
        self.assertNotIn("as this may", summary.lower())
        self.assertNotIn("duplicated emotional loss", summary.lower())
        self.assertNotIn("low risk", summary.lower())

    def test_inactive_teeth_does_not_rewrite_non_teeth_output(self):
        facts = attach_teeth_narration_facts(
            "My tooth did not fall out.",
            {"risk": "Low"},
        )
        original_seal = {"status": "Live", "type": "Focused", "risk": "Low"}
        original_interpretation = {
            "spiritual_meaning": "No active Teeth warning.",
            "effects_in_physical_realm": "None.",
            "what_to_do": "Reflect.",
        }

        seal, interpretation, full = bind_teeth_output_contract(
            doctrine_facts=facts,
            seal=original_seal,
            interpretation=original_interpretation,
            full_interpretation="Original full output.",
        )

        self.assertEqual(original_seal, seal)
        self.assertEqual(original_interpretation, interpretation)
        self.assertEqual("Original full output.", full)

    def test_protected_real_interpreter_path_returns_clean_teeth_contract(self):
        with TemporaryDirectory() as data_dir:
            data_path = Path(data_dir)
            config = patch.multiple(
                Config,
                ADMIN_KEY="admin-test-secret",
                DOCTRINE_MODE=True,
                QA_GRANTS_FILE=str(data_path / "qa_grants.json"),
                SUBSCRIBERS_FILE=str(data_path / "subscribers.json"),
                COUNTS_FILE=str(data_path / "usage_counts.json"),
                DREAM_PACKS_FILE=str(data_path / "dream_packs.json"),
            )
            base_match = (
                {
                    "symbol": "Teeth",
                    "spiritual_meaning": "generic emotional loss",
                    "effects_in_physical_realm": "duplicated emotional loss",
                    "what_to_do": "Reflect and pray without panic.",
                },
                100,
                {},
            )
            built = {
                "interpretation": {
                    "spiritual_meaning": "Generic emotional loss.",
                    "effects_in_physical_realm": (
                        "This can show up as this may reflect emotional loss."
                    ),
                    "what_to_do": "Reflect and pray without panic.",
                },
                "full_interpretation": "Generic duplicated emotional loss.",
                "top_symbols": ["Teeth"],
                "doctrine_facts": {
                    "lead_message": "Generic emotional loss",
                    "top_symbols": ["Teeth"],
                    "risk": "Low",
                },
            }

            with config:
                grant = issue_qa_grant(
                    email="output-contract@qa.jamaicantruestories.com",
                    uses=2,
                    hours=1,
                )
                app = create_app()
                with (
                    patch(
                        "app.services.interpreter_service.doctrine_available",
                        return_value=True,
                    ),
                    patch(
                        "app.services.interpreter_service.load_doctrine_sheets",
                        return_value={},
                    ),
                    patch(
                        "app.services.interpreter_service._load_layered_combinations",
                        return_value=[],
                    ),
                    patch(
                        "app.services.interpreter_service.detect_rule_hits",
                        return_value=[],
                    ),
                    patch(
                        "app.services.interpreter_service.match_base_symbols_doctrine",
                        return_value=[base_match],
                    ),
                    patch(
                        "app.services.interpreter_service.apply_override_rules",
                        return_value=None,
                    ),
                    patch(
                        "app.services.interpreter_service.compute_doctrine_seal",
                        return_value={
                            "status": "Live",
                            "type": "Focused",
                            "risk": "Low",
                            "message": "One symbolic message.",
                        },
                    ),
                    patch(
                        "app.services.interpreter_service.build_doctrine_interpretation",
                        return_value=built,
                    ),
                ):
                    response = app.test_client().post(
                        "/qa/interpret",
                        json={"dream": "My tooth fell out."},
                        headers={"X-QA-Token": grant["token"]},
                    )

        self.assertEqual(200, response.status_code)
        payload = response.get_json()
        self.assertEqual("temporary_qa", payload["access"])
        self.assertEqual("", payload["seal"]["risk"])
        self.assertTrue(payload["seal"]["legacy_risk_suppressed"])
        self.assertEqual(
            "approved_rule_match",
            payload["seal"]["warning_assessment"]["interpretation_confidence"],
        )
        self.assertEqual(
            "none",
            payload["seal"]["warning_assessment"]["predictive_certainty"],
        )
        combined = " ".join(
            [
                payload["narration"]["readable_summary"],
                payload["interpretation"]["spiritual_meaning"],
                payload["interpretation"]["effects_in_physical_realm"],
                payload["full_interpretation"],
            ]
        ).lower()
        self.assertIn("one fallen tooth", combined)
        self.assertIn("this was your own tooth", combined)
        self.assertNotIn("these were your own teeth", combined)
        self.assertNotIn("as this may", combined)
        self.assertNotIn("low risk", combined)
        self.assertTrue(payload["qa_access"]["non_billable"])
        self.assertFalse(payload["qa_access"]["customer_credits_consumed"])
        self.assertEqual(1, payload["qa_access"]["uses_remaining"])
        self.assertFalse((data_path / "subscribers.json").exists())
        self.assertFalse((data_path / "usage_counts.json").exists())


if __name__ == "__main__":
    unittest.main()
