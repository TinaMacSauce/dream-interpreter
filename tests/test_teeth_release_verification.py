import unittest

from app.release_verification import (
    validate_health_payload,
    validate_live_payload,
    validate_release_metadata,
)


COMMIT = "3f37f1085421d04ede2420cae63c4abb63a2202d"
RELEASE = {
    "build_commit": COMMIT,
    "release_id": "teeth-registry-v1",
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


class TeethProductionReleaseVerificationTests(unittest.TestCase):
    def test_valid_release_metadata_passes(self):
        self.assertEqual(
            [],
            validate_release_metadata(RELEASE, expected_commit=COMMIT),
        )

    def test_wrong_commit_fails_closed(self):
        errors = validate_release_metadata(RELEASE, expected_commit="newer-commit")

        self.assertEqual(1, len(errors))
        self.assertIn("release.build_commit", errors[0])

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


if __name__ == "__main__":
    unittest.main()
