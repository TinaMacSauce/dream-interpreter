import unittest
from unittest.mock import patch

from app.cache import TEETH_REGISTRY_CACHE
from app.config import Config
from app.teeth_doctrine import build_teeth_doctrine_context
from app.teeth_registry import (
    EXPECTED_DOCTRINE_VERSION,
    EXPECTED_RULES,
    REQUIRED_HEADERS,
    get_teeth_registry_snapshot,
    registry_content_revision,
    validate_registry_values,
)


def registry_values():
    rows = [list(REQUIRED_HEADERS)]
    for key, (rule_id, status, active) in EXPECTED_RULES.items():
        rows.append(
            [
                rule_id,
                EXPECTED_DOCTRINE_VERSION,
                "Teeth",
                f"trigger for {key}",
                f"meaning for {key}",
                f"precedence for {key}",
                f"boundary for {key}",
                status,
                "Tina, latest explicit decision",
                EXPECTED_DOCTRINE_VERSION,
                key,
                "2026-09-03T18:43:07Z",
                "TRUE" if active else "FALSE",
            ]
        )
    return rows


class _Worksheet:
    def __init__(self, values):
        self.values = values

    def get_all_values(self):
        return self.values


class _Spreadsheet:
    def __init__(self, values):
        self.values = values
        self.requested_sheet = ""

    def worksheet(self, name):
        self.requested_sheet = name
        return _Worksheet(self.values)


class TeethRegistryRuntimeTests(unittest.TestCase):
    def setUp(self):
        TEETH_REGISTRY_CACHE["loaded_at"] = 0.0
        TEETH_REGISTRY_CACHE["snapshot"] = None

    def test_manifest_preserves_17_active_and_6_unresolved_rules(self):
        values = registry_values()
        snapshot = validate_registry_values(
            values,
            expected_content_revision=registry_content_revision(values),
        )

        self.assertTrue(snapshot["verified"])
        self.assertEqual(23, snapshot["rule_count"])
        self.assertEqual(17, snapshot["active_rule_count"])
        self.assertEqual(6, snapshot["unresolved_rule_count"])
        self.assertIn("TEETH-FALLOUT-ONE", snapshot["active_rule_ids"])
        self.assertIn("TEETH-SPITTING", snapshot["unresolved_rule_ids"])

    def test_activation_drift_fails_closed_even_with_matching_content_hash(self):
        values = registry_values()
        values[1][-1] = "FALSE"

        with self.assertRaisesRegex(RuntimeError, "registry_activation_mismatch"):
            validate_registry_values(
                values,
                expected_content_revision=registry_content_revision(values),
            )

    def test_content_revision_drift_fails_closed(self):
        values = registry_values()

        with self.assertRaisesRegex(RuntimeError, "registry_content_revision_mismatch"):
            validate_registry_values(values, expected_content_revision="fnv1a64:wrong")

    def test_production_loader_reads_exact_canonical_tab(self):
        values = registry_values()
        spreadsheet = _Spreadsheet(values)
        revision = registry_content_revision(values)

        with (
            patch.object(Config, "APP_ENV", "production"),
            patch("app.teeth_registry.EXPECTED_CONTENT_REVISION", revision),
            patch("app.teeth_registry.get_spreadsheet", return_value=spreadsheet),
        ):
            snapshot = get_teeth_registry_snapshot(force=True)

        self.assertTrue(snapshot["verified"])
        self.assertEqual("DoctrineRegistry", spreadsheet.requested_sheet)
        self.assertEqual("canonical_sheet", snapshot["loaded_from"])

    def test_registry_failure_withholds_all_doctrine_rules(self):
        failed = {
            "verified": False,
            "doctrine_version": EXPECTED_DOCTRINE_VERSION,
            "rules": {},
            "active_rule_ids": [],
            "unresolved_rule_ids": [],
            "error": "registry_content_revision_mismatch",
        }

        with patch("app.teeth_doctrine.get_teeth_registry_snapshot", return_value=failed):
            doctrine = build_teeth_doctrine_context("My tooth fell out.")

        self.assertFalse(doctrine["active_doctrine"])
        self.assertFalse(doctrine["active_warning"])
        self.assertEqual([], doctrine["applied_rule_ids"])
        self.assertFalse(doctrine["doctrine_registry"]["verified"])


if __name__ == "__main__":
    unittest.main()
