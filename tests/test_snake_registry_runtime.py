import unittest

from app.snake_registry import (
    EXPECTED_AUTHORITY,
    EXPECTED_DOCTRINE_VERSION,
    EXPECTED_RULES,
    EXPECTED_UPDATED_AT_UTC,
    validate_snake_registry_values,
)
from app.teeth_registry import REQUIRED_HEADERS, registry_content_revision


def snake_registry_values():
    values = [list(REQUIRED_HEADERS)]
    for key, (rule_id, status, active) in EXPECTED_RULES.items():
        values.append([
            rule_id, EXPECTED_DOCTRINE_VERSION, "Snake", f"trigger {key}",
            f"meaning {key}", f"precedence {key}", f"boundary {key}",
            status, EXPECTED_AUTHORITY, EXPECTED_DOCTRINE_VERSION, key,
            EXPECTED_UPDATED_AT_UTC, "TRUE" if active else "FALSE",
        ])
    return values


class SnakeRegistryRuntimeTests(unittest.TestCase):
    def test_exact_18_rule_manifest_is_verified(self):
        values = snake_registry_values()
        snapshot = validate_snake_registry_values(
            values,
            expected_content_revision=registry_content_revision(values),
        )
        self.assertTrue(snapshot["verified"])
        self.assertEqual(18, snapshot["rule_count"])
        self.assertEqual(18, snapshot["active_rule_count"])
        self.assertEqual("DoctrineRegistry!A25:M42", snapshot["sheet_range"])

    def test_activation_or_unknown_rule_drift_fails_closed(self):
        values = snake_registry_values()
        values[1][-1] = "FALSE"
        with self.assertRaisesRegex(RuntimeError, "snake_registry_activation_mismatch"):
            validate_snake_registry_values(
                values,
                expected_content_revision=registry_content_revision(values),
            )

        values = snake_registry_values()
        values[1][10] = "unknown_key"
        with self.assertRaisesRegex(RuntimeError, "snake_registry_implementation_key_mismatch"):
            validate_snake_registry_values(
                values,
                expected_content_revision=registry_content_revision(values),
            )

    def test_teeth_rows_do_not_change_snake_cluster_fingerprint(self):
        values = snake_registry_values()
        mixed = [values[0], [
            "TEETH-FALLOUT-OWN", "DEC-TEETH-2026-09-03-05", "Teeth", "x",
            "x", "x", "x", "APPROVED", "Tina, latest explicit decision",
            "DEC-TEETH-2026-09-03-05", "own_fallout",
            "2026-09-03T18:43:07Z", "TRUE",
        ], *values[1:]]
        snapshot = validate_snake_registry_values(
            mixed,
            expected_content_revision=registry_content_revision(values),
        )
        self.assertTrue(snapshot["verified"])


if __name__ == "__main__":
    unittest.main()
