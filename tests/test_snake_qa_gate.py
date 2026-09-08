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
        self.assertEqual("snake-qa-contract-v3", payload["contract_version"])
        self.assertEqual(len(SNAKE_QA_CASES), payload["case_count"])
        self.assertEqual(67, payload["case_count"])
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

    def test_all_independent_ordinary_language_regressions_are_permanent(self):
        app = Flask(__name__)
        app.register_blueprint(qa_bp)
        payload = app.test_client().get("/qa/snake-regression").get_json()
        cases = {case["case_id"]: case for case in payload["cases"]}
        expected = {
            "REG-SNAKE-ATTACK-001",
            "REG-SNAKE-BITE-ATTEMPT-001",
            "REG-SNAKE-BITE-DREAMER-001",
            "REG-SNAKE-CHASE-ESCAPE-001",
            "REG-SNAKE-CHASE-CAPTURE-001",
            "REG-SNAKE-DEFEAT-001",
            "REG-SNAKE-HYPOTHETICAL-001",
            "REG-SNAKE-MULTI-ACTION-001",
            "REG-SNAKE-MIXED-ENDINGS-001",
            "REG-SNAKE-NEGATION-001",
            "REG-SNAKE-PROTECT-OTHER-001",
            "REG-SNAKE-SIZE-SPECIES-001",
            "REG-SNAKE-TRANSFORM-001",
            "REG-SNAKE-TRANSFORM-ACCUSATION-001",
            "REG-SNAKE-VENOM-ABSENT-001",
            "REG-SNAKE-QUOTED-001",
        }
        self.assertTrue(expected.issubset(cases))
        for case_id in expected:
            with self.subTest(case_id=case_id):
                integrity = cases[case_id]["doctrine"]["event_graph"]["graph_integrity"]
                self.assertEqual({"verified": True, "reason_codes": []}, integrity)


if __name__ == "__main__":
    unittest.main()
