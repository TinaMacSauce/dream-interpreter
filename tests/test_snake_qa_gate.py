import os
import unittest
from unittest.mock import patch

from flask import Flask

from app.routes.qa import SNAKE_QA_CASES, qa_bp


class SnakeQAReleaseGateTests(unittest.TestCase):
    def test_fixed_contract_is_non_billable_and_covers_all_cases(self):
        app = Flask(__name__)
        app.register_blueprint(qa_bp)
        with patch.dict(os.environ, {"RENDER_GIT_COMMIT": "snake-qa-sha"}, clear=False):
            response = app.test_client().get("/qa/snake-regression")
        payload = response.get_json()
        self.assertEqual(200, response.status_code)
        self.assertEqual("snake-qa-contract-v1", payload["contract_version"])
        self.assertEqual(len(SNAKE_QA_CASES), payload["case_count"])
        self.assertTrue(payload["non_billable"])
        self.assertFalse(payload["customer_credits_consumed"])
        self.assertTrue(payload["doctrine_registry"]["verified"])

    def test_every_active_case_is_safety_scoped(self):
        app = Flask(__name__)
        app.register_blueprint(qa_bp)
        payload = app.test_client().get("/qa/snake-regression").get_json()
        forbidden = ("definitely", "will happen", "is the enemy", "will get sick")
        for case in payload["cases"]:
            narration = case["narration"]
            if narration.get("active") is not True:
                continue
            text = narration["narration_text"].lower()
            with self.subTest(case_id=case["case_id"]):
                self.assertIn("spiritual tradition", text)
                for phrase in forbidden:
                    self.assertNotIn(phrase, text)


if __name__ == "__main__":
    unittest.main()
