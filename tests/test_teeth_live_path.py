import unittest
from unittest.mock import patch

from flask import Flask

from app.services import interpreter_service as service


class TeethLivePathTests(unittest.TestCase):
    def setUp(self):
        self.app = Flask(__name__)
        self.app.secret_key = "teeth-live-path-test"

    def _run_dream(self, dream):
        built = {
            "doctrine_facts": {},
            "top_symbols": ["Teeth"],
            "interpretation": {
                "spiritual_meaning": "",
                "effects_in_physical_realm": "",
                "what_to_do": "",
            },
            "full_interpretation": "",
        }

        patches = (
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
            patch.object(service, "build_narration_result", return_value={}),
        )

        with self.app.test_request_context("/interpret", method="POST", json={"dream": dream}):
            managers = [item.start() for item in patches]
            try:
                response = service.run_interpretation()
                payload = response.get_json()
                narration_mock = managers[-1]
                narration_facts = narration_mock.call_args.kwargs["doctrine_facts"]
            finally:
                for item in reversed(patches):
                    item.stop()

        return payload, narration_facts

    def test_one_painful_tooth_reaches_live_narration_with_approved_scope(self):
        payload, facts = self._run_dream("One of my teeth fell out with pain.")

        teeth = facts["teeth_narration"]
        self.assertTrue(teeth["active"])
        self.assertEqual(teeth["warning_count"], "one_person")
        self.assertEqual(teeth["relationship_scope"], "relative_or_close_friend")
        self.assertEqual(teeth["proximity"], "very_close_or_close_relative")
        self.assertEqual(payload["doctrine_facts"]["teeth_narration"], teeth)

    def test_multiple_painless_teeth_reach_live_narration_without_positional_leakage(self):
        _, facts = self._run_dream("Several of my lower teeth fell out without pain.")

        teeth = facts["teeth_narration"]
        self.assertTrue(teeth["active"])
        self.assertEqual(teeth["warning_count"], "multiple_people")
        self.assertEqual(teeth["proximity"], "friend_acquaintance_or_more_distant")
        rendered = str(teeth).lower()
        self.assertNotIn("blood relative", rendered)
        self.assertNotIn("parent", rendered)
        self.assertNotIn("child", rendered)
        self.assertNotIn("older relative", rendered)
        self.assertNotIn("younger relative", rendered)

    def test_negated_tooth_loss_stays_inactive_on_live_path(self):
        _, facts = self._run_dream("My tooth did not fall out and it hurt.")

        teeth = facts["teeth_narration"]
        self.assertFalse(teeth["active"])
        self.assertEqual(teeth["lead"], "")
        self.assertEqual(teeth["details"], [])


if __name__ == "__main__":
    unittest.main()
