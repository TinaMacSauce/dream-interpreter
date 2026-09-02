import unittest
from unittest.mock import patch

from flask import Flask

from app.services import interpreter_service as service


class TeethLivePathTests(unittest.TestCase):
    def setUp(self):
        self.app = Flask(__name__)
        self.app.secret_key = "teeth-live-path-test"

    def test_teeth_facts_are_attached_before_narration_and_returned_in_payload(self):
        dream = "One of my teeth fell out with pain."
        built = {
            "doctrine_facts": {"existing": "preserved"},
            "top_symbols": ["Teeth"],
            "interpretation": {
                "spiritual_meaning": "",
                "effects_in_physical_realm": "",
                "what_to_do": "",
            },
            "full_interpretation": "",
        }
        enriched = {
            "existing": "preserved",
            "event_context": {
                "priority_order": ["action", "subject", "place", "context", "ending"],
                "primary_action": {},
                "primary_subject": "",
                "primary_place": {},
                "primary_state": {},
                "primary_relationship": {},
                "primary_context": {},
                "primary_ending": {},
                "subjects": [],
            },
            "top_symbols": ["Teeth"],
            "teeth_narration": {
                "active": True,
                "warning_count": "one_person",
                "relationship_scope": "relative_or_close_friend",
                "proximity": "very_close_or_close_relative",
                "lead": "approved-teeth-fact",
                "details": [],
            },
        }

        patches = (
            patch.object(service.Config, "DOCTRINE_MODE", True),
            patch.object(service, "validate_dream_text", return_value=None),
            patch.object(service, "has_active_access", return_value=(True, {"type": "subscription"})),
            patch.object(service, "get_session_email", return_value="test@example.com"),
            patch.object(service, "persist_email_to_session", return_value=None),
            patch.object(service, "get_dream_pack_status", return_value={}),
            patch.object(service, "doctrine_available", return_value=True),
            patch.object(service, "load_doctrine_sheets", return_value={}),
            patch.object(service, "_load_layered_combinations", return_value=[]),
            patch.object(service, "detect_rule_hits", return_value=[]),
            patch.object(service, "match_base_symbols_doctrine", return_value=[]),
            patch.object(service, "apply_override_rules", return_value=None),
            patch.object(service, "compute_doctrine_seal", return_value={}),
            patch.object(service, "build_doctrine_interpretation", return_value=built),
            patch.object(service, "attach_teeth_narration_facts", return_value=enriched),
            patch.object(service, "build_narration_result", return_value={"mode": "deterministic_event"}),
        )

        with self.app.test_request_context("/interpret", method="POST", json={"dream": dream}):
            managers = [item.start() for item in patches]
            try:
                response = service.run_interpretation()
                payload = response.get_json()
                attach_mock = managers[-2]
                narration_mock = managers[-1]
            finally:
                for item in reversed(patches):
                    item.stop()

        attach_mock.assert_called_once()
        self.assertEqual(attach_mock.call_args.args[0], dream)
        self.assertEqual(attach_mock.call_args.args[1]["existing"], "preserved")
        narration_mock.assert_called_once()
        self.assertEqual(narration_mock.call_args.kwargs["doctrine_facts"], enriched)
        self.assertEqual(payload["doctrine_facts"], enriched)
        self.assertEqual(payload["doctrine_facts"]["teeth_narration"]["warning_count"], "one_person")


if __name__ == "__main__":
    unittest.main()
