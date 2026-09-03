import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from flask import Flask

from app.config import Config
from app.qa_access import consume_qa_grant, get_qa_grant_status, issue_qa_grant
from app.routes.admin import admin_bp
from app.services.interpreter_service import run_interpretation


class QaAccessContractTests(unittest.TestCase):
    def _config(self, data_path: Path):
        return patch.multiple(
            Config,
            ADMIN_KEY="admin-test-secret",
            QA_GRANTS_FILE=str(data_path / "qa_grants.json"),
            QA_EMAIL_DOMAIN="qa.jamaicantruestories.com",
            QA_DEFAULT_USES=25,
            QA_MAX_USES=50,
            QA_DEFAULT_HOURS=2,
            QA_MAX_HOURS=6,
            SUBSCRIBERS_FILE=str(data_path / "subscribers.json"),
            COUNTS_FILE=str(data_path / "usage_counts.json"),
        )

    def test_admin_grant_is_protected_bounded_isolated_and_revocable(self):
        with TemporaryDirectory() as data_dir:
            data_path = Path(data_dir)
            app = Flask(__name__)
            app.secret_key = "test-secret"
            app.register_blueprint(admin_bp)

            with self._config(data_path):
                client = app.test_client()
                denied = client.post(
                    "/admin/qa-grant",
                    json={"email": "regression@qa.jamaicantruestories.com"},
                )
                self.assertEqual(403, denied.status_code)

                wrong_domain = client.post(
                    "/admin/qa-grant",
                    headers={"X-Admin-Key": "admin-test-secret"},
                    json={"email": "customer@example.com"},
                )
                self.assertEqual(400, wrong_domain.status_code)

                granted = client.post(
                    "/admin/qa-grant",
                    headers={"X-Admin-Key": "admin-test-secret"},
                    json={
                        "email": "regression@qa.jamaicantruestories.com",
                        "uses": 999,
                        "hours": 999,
                    },
                )
                self.assertEqual(200, granted.status_code)
                grant = granted.get_json()
                self.assertTrue(grant["non_billable"])
                self.assertFalse(grant["customer_credits_consumed"])
                self.assertEqual(50, grant["uses_remaining"])
                self.assertTrue(grant["token"])

                grants_path = data_path / "qa_grants.json"
                stored_text = grants_path.read_text(encoding="utf-8")
                stored = json.loads(stored_text)
                record = stored["grants"][grant["grant_id"]]
                self.assertNotIn(grant["token"], stored_text)
                self.assertEqual(64, len(record["token_hash"]))
                self.assertFalse((data_path / "subscribers.json").exists())
                self.assertFalse((data_path / "usage_counts.json").exists())

                revoked = client.post(
                    "/admin/qa-revoke",
                    headers={"X-Admin-Key": "admin-test-secret"},
                    json={"grant_id": grant["grant_id"]},
                )
                self.assertEqual(200, revoked.status_code)
                status = get_qa_grant_status(grant["token"])
                self.assertFalse(status["active"])
                self.assertEqual("revoked", status["reason"])

    def test_grant_consumption_is_bounded(self):
        with TemporaryDirectory() as data_dir:
            data_path = Path(data_dir)
            with self._config(data_path):
                grant = issue_qa_grant(
                    email="bounded@qa.jamaicantruestories.com",
                    uses=1,
                    hours=1,
                )
                before = get_qa_grant_status(grant["token"])
                after = consume_qa_grant(grant["token"])
                denied = consume_qa_grant(grant["token"])

            self.assertTrue(before["active"])
            self.assertEqual(1, before["uses_remaining"])
            self.assertFalse(after["active"])
            self.assertTrue(after["consumed"])
            self.assertEqual(0, after["uses_remaining"])
            self.assertEqual("exhausted", after["reason"])
            self.assertFalse(denied["consumed"])
            self.assertEqual(0, denied["uses_remaining"])

    def test_invalid_qa_token_fails_closed_before_customer_access(self):
        with TemporaryDirectory() as data_dir:
            data_path = Path(data_dir)
            app = Flask(__name__)
            app.secret_key = "test-secret"

            with (
                self._config(data_path),
                app.test_request_context(
                    "/interpret",
                    method="POST",
                    json={"dream": "My tooth fell out."},
                    headers={"X-QA-Token": "invalid-token"},
                ),
                patch(
                    "app.services.interpreter_service.has_active_access"
                ) as customer_access,
            ):
                response, status = run_interpretation()

            self.assertEqual(403, status)
            self.assertEqual("invalid_token", response.get_json()["reason"])
            customer_access.assert_not_called()

    def test_valid_qa_token_uses_normal_interpreter_without_customer_credit(self):
        with TemporaryDirectory() as data_dir:
            data_path = Path(data_dir)
            app = Flask(__name__)
            app.secret_key = "test-secret"
            interpretation = {
                "spiritual_meaning": "Culturally scoped test meaning.",
                "effects_in_physical_realm": "No certain outcome.",
                "what_to_do": "Reflect without treating this as prediction.",
            }

            with self._config(data_path):
                grant = issue_qa_grant(
                    email="full-path@qa.jamaicantruestories.com",
                    uses=2,
                    hours=1,
                )

                with (
                    app.test_request_context(
                        "/interpret",
                        method="POST",
                        json={"dream": "My tooth fell out."},
                        headers={"X-QA-Token": grant["token"]},
                    ),
                    patch.object(Config, "DOCTRINE_MODE", False),
                    patch(
                        "app.services.interpreter_service.load_legacy_rows",
                        return_value=[],
                    ),
                    patch(
                        "app.services.interpreter_service.match_symbols_legacy",
                        return_value=[],
                    ),
                    patch(
                        "app.services.interpreter_service.build_legacy_interpretation",
                        return_value=interpretation,
                    ),
                    patch(
                        "app.services.interpreter_service.has_active_access"
                    ) as customer_access,
                    patch(
                        "app.services.interpreter_service.get_dream_pack_status"
                    ) as dream_pack,
                    patch(
                        "app.services.interpreter_service.shadow_increment"
                    ) as free_credit,
                    patch(
                        "app.services.interpreter_service.consume_dream_pack_use"
                    ) as customer_credit,
                ):
                    response = run_interpretation()

            payload = response.get_json()
            self.assertEqual(200, response.status_code)
            self.assertEqual("temporary_qa", payload["access"])
            self.assertTrue(payload["qa_access"]["non_billable"])
            self.assertFalse(payload["qa_access"]["customer_credits_consumed"])
            self.assertEqual(1, payload["qa_access"]["uses_remaining"])
            customer_access.assert_not_called()
            dream_pack.assert_not_called()
            free_credit.assert_not_called()
            customer_credit.assert_not_called()
            self.assertFalse((data_path / "subscribers.json").exists())
            self.assertFalse((data_path / "usage_counts.json").exists())


if __name__ == "__main__":
    unittest.main()
