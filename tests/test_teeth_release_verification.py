import unittest

from app.release_verification import (
    validate_health_payload,
    validate_live_payload,
    validate_qa_status_payload,
    validate_release_metadata,
    validate_version_payload,
)


COMMIT = "3f37f1085421d04ede2420cae63c4abb63a2202d"
RELEASE = {
    "build_commit": COMMIT,
    "release_id": "teeth-registry-v1",
    "repository": "TinaMacSauce/dream-interpreter",
    "repository_url": "https://github.com/TinaMacSauce/dream-interpreter",
    "commit_url": f"https://github.com/TinaMacSauce/dream-interpreter/commit/{COMMIT}",
    "production_url": "https://interpreter.jamaicantruestories.com",
    "version_endpoint": "https://interpreter.jamaicantruestories.com/version",
    "qa_status_endpoint": "https://interpreter.jamaicantruestories.com/qa/status",
    "teeth_doctrine_version": "DEC-TEETH-2026-09-03-05",
    "teeth_context_version": "teeth-context-v2",
    "doctrine_registry": "Dream Symbol Dictionary!DoctrineRegistry",
    "teeth_registry_sheet_revision": "6134",
    "teeth_registry_content_revision": "fnv1a64:c51447de5d35bd59",
    "teeth_registry_contract_version": "teeth-doctrine-registry-v1",
}

REGISTRY = {
    "verified": True,
    "contract_version": "teeth-doctrine-registry-v1",
    "sheet_revision": "6134",
    "content_revision": "fnv1a64:c51447de5d35bd59",
    "doctrine_version": "DEC-TEETH-2026-09-03-05",
    "rule_count": 23,
    "active_rule_count": 17,
    "unresolved_rule_count": 6,
    "loaded_from": "canonical_sheet",
}

QA_ACCESS = {
    "configured": True,
    "storage_ready": True,
    "grant_route": "/admin/qa-grant",
    "revoke_route": "/admin/qa-revoke",
    "interpret_route": "/qa/interpret",
    "application_route": "/interpret",
    "fixed_contract_route": "/qa/teeth-regression",
    "grant_authentication": "X-Admin-Key",
    "interpret_authentication": "X-QA-Token or Authorization Bearer",
    "non_billable": True,
    "customer_credits_consumed": False,
    "customer_entitlement_store_used": False,
    "token_storage": "sha256_hash_only",
    "revocable": True,
}


class TeethProductionReleaseVerificationTests(unittest.TestCase):
    def test_valid_release_metadata_passes(self):
        self.assertEqual(
            [],
            validate_release_metadata(RELEASE, expected_commit=COMMIT),
        )

    def test_wrong_commit_fails_closed(self):
        errors = validate_release_metadata(RELEASE, expected_commit="newer-commit")

        self.assertEqual(2, len(errors))
        self.assertIn("release.build_commit", " ".join(errors))
        self.assertIn("release.commit_url", " ".join(errors))

    def test_missing_release_metadata_fails_closed(self):
        self.assertTrue(
            validate_release_metadata(None, expected_commit=COMMIT)
        )

    def test_valid_live_payload_passes(self):
        payload = {
            "alive": True,
            "service": "dream-interpreter",
            "release": RELEASE,
        }

        self.assertEqual([], validate_live_payload(payload, expected_commit=COMMIT))

    def test_live_payload_rejects_wrong_service_and_doctrine(self):
        release = dict(RELEASE, teeth_doctrine_version="superseded")
        payload = {"alive": True, "service": "other", "release": release}

        errors = validate_live_payload(payload, expected_commit=COMMIT)

        self.assertEqual(2, len(errors))

    def test_valid_health_payload_passes(self):
        payload = {
            "service": "dream-interpreter",
            "status": "healthy",
            "spreadsheet_connected": True,
            "doctrine_sheets_available": True,
            "teeth_registry": REGISTRY,
            "release": RELEASE,
        }

        self.assertEqual([], validate_health_payload(payload, expected_commit=COMMIT))

    def test_health_payload_rejects_dependency_failure(self):
        payload = {
            "service": "dream-interpreter",
            "status": "degraded",
            "spreadsheet_connected": False,
            "doctrine_sheets_available": False,
            "teeth_registry": REGISTRY,
            "release": RELEASE,
        }

        errors = validate_health_payload(payload, expected_commit=COMMIT)

        self.assertEqual(3, len(errors))

    def test_health_payload_rejects_unverified_registry(self):
        payload = {
            "service": "dream-interpreter",
            "status": "degraded",
            "spreadsheet_connected": True,
            "doctrine_sheets_available": True,
            "teeth_registry": dict(REGISTRY, verified=False),
            "release": RELEASE,
        }

        errors = validate_health_payload(payload, expected_commit=COMMIT)

        self.assertIn("teeth_registry.verified", " ".join(errors))

    def test_valid_version_payload_passes(self):
        payload = {
            "service": "dream-interpreter",
            "production_url": "https://interpreter.jamaicantruestories.com",
            "release": RELEASE,
        }

        self.assertEqual([], validate_version_payload(payload, expected_commit=COMMIT))

    def test_valid_qa_status_payload_passes(self):
        payload = {
            "service": "dream-interpreter",
            "ready": True,
            "release": RELEASE,
            "qa_access": QA_ACCESS,
            "doctrine_registry": REGISTRY,
        }

        self.assertEqual([], validate_qa_status_payload(payload, expected_commit=COMMIT))

    def test_qa_status_rejects_customer_entitlement_use(self):
        payload = {
            "service": "dream-interpreter",
            "ready": True,
            "release": RELEASE,
            "qa_access": dict(QA_ACCESS, customer_entitlement_store_used=True),
            "doctrine_registry": REGISTRY,
        }

        errors = validate_qa_status_payload(payload, expected_commit=COMMIT)

        self.assertIn("customer_entitlement_store_used", " ".join(errors))


if __name__ == "__main__":
    unittest.main()
