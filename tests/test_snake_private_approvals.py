"""Synthetic registry examples only: no canonical captures or founder wording."""
import copy
import itertools
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from app.cache import SNAKE_REGISTRY_CACHE
from app.config import Config
from app.registry_approvals import (
    MAX_MANIFEST_BYTES, METADATA_FIELDS, load_approval_manifest,
    registry_snapshot_sha256, validate_approval_manifest,
)
from app.snake_doctrine import build_snake_narration_facts
from app.snake_integration import bind_snake_output_contract
from app.snake_registry import (
    EXPECTED_RULES, get_snake_registry_snapshot, public_snake_registry_metadata,
    validate_snake_registry_values,
)
from app.teeth_registry import REQUIRED_HEADERS, cluster_registry_values, registry_content_revision
from tests.test_snake_registry_runtime import snake_registry_values


MOCK_VERSION = "MOCK-REVISION-V2"
MOCK_AMENDED_KEYS = (
    "snake_location", "snake_location_house_life", "snake_location_bedroom_intimacy",
    "snake_location_kitchen_productivity_healing_replenishment",
)


def synthetic_values(amended=True):
    values = [list(REQUIRED_HEADERS)]
    for index, (key, (rule_id, status, active)) in enumerate(EXPECTED_RULES.items()):
        revised = amended and key in MOCK_AMENDED_KEYS
        version = MOCK_VERSION if revised else "MOCK-BASE-V1"
        marker = f"Mock {'revised' if revised else 'original'} annotation {index}."
        values.append([
            rule_id, version, "Snake", f"Mock trigger {index}.", marker,
            f"Mock precedence {index}.", f"Mock safety constraint {index}.",
            status, "Mock approval authority", version, key,
            "2000-02-02T00:00:00Z" if revised else "2000-01-01T00:00:00Z",
            "TRUE" if active else "FALSE",
        ])
    return values


def synthetic_manifest(*snapshots):
    """Create mock approvals for tests; never generate runtime trust from a Sheet."""
    profiles = []
    for values in snapshots:
        scoped = cluster_registry_values(values, "Snake")
        rows = [dict(zip(REQUIRED_HEADERS, row)) for row in scoped[1:]]
        profiles.append({
            "snapshot_sha256": registry_snapshot_sha256(scoped),
            "doctrine_version": MOCK_VERSION if any(row["doctrine_version"] == MOCK_VERSION for row in rows) else rows[0]["doctrine_version"],
            "sheet_revision": "",
            "rule_metadata": {row["implementation_key"]: {field: row[field] for field in METADATA_FIELDS} for row in rows},
        })
    return {"schema_version": 1, "profiles": profiles}


class SnakePrivateApprovalTests(unittest.TestCase):
    def setUp(self):
        self.old, self.new = synthetic_values(False), synthetic_values()
        self.manifest = synthetic_manifest(self.old, self.new)

    def validate(self, values=None, manifest=None):
        return validate_snake_registry_values(
            self.new if values is None else values,
            approval_manifest=self.manifest if manifest is None else manifest,
        )

    def render(self, dream, values=None):
        snapshot = self.validate(values)
        with patch("app.snake_doctrine.get_snake_registry_snapshot", return_value=snapshot):
            return build_snake_narration_facts(dream)

    def test_complete_approved_versions_accepted_without_digest_override(self):
        for values in (self.old, self.new):
            snapshot = self.validate(values)
            self.assertTrue(snapshot["verified"])
            self.assertEqual((29, 24, 5), tuple(snapshot[key] for key in (
                "rule_count", "active_rule_count", "unresolved_rule_count",
            )))
            self.assertTrue(snapshot["canonical_location_text"])

    def test_unapproved_data_and_digest_override_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "content_revision_mismatch"):
            validate_snake_registry_values(self.new)
        with self.assertRaisesRegex(RuntimeError, "approval_override_conflict"):
            validate_snake_registry_values(self.new, approval_manifest=self.manifest,
                                           expected_content_revision=registry_content_revision(self.new))

    def test_legacy_validation_and_narration_remain_available(self):
        values = snake_registry_values()
        # Replace the legacy pin with a synthetic fixture's hash, not live data.
        with patch("app.snake_registry.EXPECTED_CONTENT_REVISION", registry_content_revision(values)):
            for manifest in (None, self.manifest):
                snapshot = validate_snake_registry_values(values, approval_manifest=manifest)
                self.assertFalse(snapshot["canonical_location_text"])
                self.assertTrue(snapshot["verified"])
                with patch("app.snake_doctrine.get_snake_registry_snapshot", return_value=snapshot):
                    result = build_snake_narration_facts("A snake was in my kitchen.")
                self.assertTrue(result["active"])
                self.assertEqual("", result["location_governing_meaning"])

    def test_all_partial_update_combinations_fail(self):
        indices = [i for i, row in enumerate(self.old) if row[10] in MOCK_AMENDED_KEYS]
        for count in (1, 2, 3):
            for subset in itertools.combinations(indices, count):
                values = copy.deepcopy(self.old)
                for index in subset:
                    values[index] = self.new[index]
                with self.subTest(rows=subset), self.assertRaisesRegex(RuntimeError, "content_revision_mismatch"):
                    self.validate(values)

    def test_every_amended_cell_tamper_is_rejected(self):
        for index, row in enumerate(self.new):
            if row[10] not in MOCK_AMENDED_KEYS:
                continue
            for column in range(len(REQUIRED_HEADERS)):
                values = copy.deepcopy(self.new)
                values[index][column] = "Mock unauthorized edit"
                with self.subTest(row=index, column=column), self.assertRaises(RuntimeError):
                    self.validate(values)

    def test_schema_missing_duplicate_extra_and_unknown_rows_rejected(self):
        for mutation in ("header", "missing", "duplicate", "extra_cell", "short_row", "unknown_key", "wrong_id"):
            values = copy.deepcopy(self.new)
            if mutation == "header": values[0].pop()
            elif mutation == "missing": values.pop()
            elif mutation == "duplicate": values.append(copy.deepcopy(values[1]))
            elif mutation == "extra_cell": values[1].append("Mock extra")
            elif mutation == "short_row": values[1].pop()
            elif mutation == "unknown_key": values[1][10] = "mock_unknown_key"
            else: values[1][0] = "MOCK-UNKNOWN-RULE"
            manifest = copy.deepcopy(self.manifest)
            manifest["profiles"][1]["snapshot_sha256"] = registry_snapshot_sha256(cluster_registry_values(values, "Snake"))
            with self.subTest(mutation=mutation), self.assertRaises(RuntimeError):
                self.validate(values, manifest)

    def test_required_content_and_safety_fields_cannot_be_empty(self):
        for field in ("trigger_context", "governing_meaning", "precedence", "safety_boundary"):
            values = copy.deepcopy(self.new)
            values[1][REQUIRED_HEADERS.index(field)] = ""
            with self.subTest(field=field), self.assertRaisesRegex(RuntimeError, "required_field_missing"):
                self.validate(values, synthetic_manifest(values))

    def test_unresolved_activation_and_malformed_booleans_fail_even_with_digest(self):
        for index, row in enumerate(self.new[1:], 1):
            if row[7] != "UNRESOLVED": continue
            for value in ("TRUE", "unknown", ""):
                values = copy.deepcopy(self.new)
                values[index][12] = value
                with self.subTest(row=index, value=value), self.assertRaisesRegex(RuntimeError, "activation_mismatch"):
                    self.validate(values, synthetic_manifest(values))
        self.assertEqual(5, len(self.validate()["unresolved_rule_ids"]))

    def test_exact_row_metadata_is_enforced_independently_of_digest(self):
        for field in METADATA_FIELDS:
            values = copy.deepcopy(self.new)
            values[1][REQUIRED_HEADERS.index(field)] = "Mock wrong metadata"
            manifest = copy.deepcopy(self.manifest)
            manifest["profiles"][1]["snapshot_sha256"] = registry_snapshot_sha256(values)
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                self.validate(values, manifest)

    def test_mixed_versions_preserve_per_row_provenance(self):
        snapshot = self.validate()
        self.assertEqual(MOCK_VERSION, snapshot["doctrine_version"])
        self.assertEqual(["MOCK-BASE-V1", MOCK_VERSION], snapshot["decision_ids"])
        self.assertEqual("", snapshot["sheet_revision"])
        for row in self.new[1:]:
            stored = snapshot["rules"][row[10]]
            for field in METADATA_FIELDS:
                self.assertEqual(row[REQUIRED_HEADERS.index(field)], stored[field])

    def test_other_clusters_do_not_change_snapshot_or_join_runtime_rules(self):
        values = [self.new[0], ["MOCK-OTHER", "MOCK-BASE-V1", "Other"], *self.new[1:]]
        self.assertEqual(self.validate()["content_revision"], self.validate(values)["content_revision"])
        self.assertNotIn("MOCK-OTHER", self.validate(values)["active_rule_ids"])

    def test_malformed_manifest_profiles_fail_closed(self):
        for mutation in ("version", "boolean_version", "unknown_field", "empty", "duplicate",
                         "digest", "missing_key", "unknown_key", "metadata_field", "metadata_empty",
                         "decision", "timestamp", "top_version", "sheet_revision"):
            manifest = copy.deepcopy(self.manifest)
            profile = manifest["profiles"][1]
            metadata = profile["rule_metadata"][MOCK_AMENDED_KEYS[0]]
            if mutation == "version": manifest["schema_version"] = 2
            elif mutation == "boolean_version": manifest["schema_version"] = True
            elif mutation == "unknown_field": profile["governing_meaning"] = "Mock forbidden field"
            elif mutation == "empty": manifest["profiles"] = []
            elif mutation == "duplicate": manifest["profiles"].append(copy.deepcopy(profile))
            elif mutation == "digest": profile["snapshot_sha256"] = "sha256:bad"
            elif mutation == "missing_key": profile["rule_metadata"].pop(MOCK_AMENDED_KEYS[0])
            elif mutation == "unknown_key": profile["rule_metadata"]["mock_unknown"] = metadata
            elif mutation == "metadata_field": metadata["active"] = True
            elif mutation == "metadata_empty": metadata["authority"] = " "
            elif mutation == "decision": metadata["decision_id"] = "MOCK-WRONG"
            elif mutation == "timestamp": metadata["updated_at_utc"] = "2000-99-99T00:00:00Z"
            elif mutation == "top_version": profile["doctrine_version"] = "MOCK-MISSING"
            else: profile["sheet_revision"] = "Mock predicted revision"
            with self.subTest(mutation=mutation), self.assertRaisesRegex(RuntimeError, "approval_manifest_invalid"):
                self.validate(manifest=manifest)

    def test_config_file_rejects_null_duplicate_keys_invalid_json_and_oversize(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mock.private.json"
            for raw in ("null", "[]", "{", '{"schema_version":1,"schema_version":1}', " " * (MAX_MANIFEST_BYTES + 1)):
                path.write_text(raw)
                with self.assertRaisesRegex(RuntimeError, "approval_manifest_invalid"):
                    load_approval_manifest(str(path))
            path.unlink()
            with self.assertRaisesRegex(RuntimeError, "approval_manifest_invalid"):
                load_approval_manifest(str(path))
        self.assertIsNone(load_approval_manifest(""))

    def test_private_row_text_reaches_narration_once_with_provenance(self):
        for place, key in zip(("house", "bedroom", "kitchen"), MOCK_AMENDED_KEYS[1:]):
            result = self.render(f"A snake was in my {place}.")
            meaning = self.validate()["rules"][key]["governing_meaning"]
            self.assertEqual(1, result["narration_text"].count(meaning))
            self.assertEqual(MOCK_VERSION, result["location_rule_provenance"]["decision_id"])
            self.assertIn("spiritual tradition", result["narration_text"])
            self.assertEqual("none", result["predictive_certainty"])
            self.assertIn("culprit", result["narration_text"])
            self.assertEqual(len(result["applied_rule_ids"]), len(set(result["applied_rule_ids"])))

    def test_home_alias_and_specific_location_precedence(self):
        home, house = self.render("A snake was at home."), self.render("A snake was in my house.")
        self.assertEqual(home["location_governing_meaning"], house["location_governing_meaning"])
        for place, key in zip(("bedroom", "kitchen"), MOCK_AMENDED_KEYS[2:]):
            result = self.render(f"At home a snake was in my {place}.")
            self.assertEqual(self.validate()["rules"][key]["governing_meaning"], result["location_governing_meaning"])
            self.assertNotIn(EXPECTED_RULES[MOCK_AMENDED_KEYS[1]][0], result["applied_rule_ids"])

    def test_unresolved_or_unmapped_locations_remain_physical_only(self):
        for place in ("yard", "bathroom", "living room"):
            for prefix in ("", "At home "):
                with self.subTest(place=place, prefix=prefix):
                    result = self.render(f"{prefix}a snake was in my {place}.")
                    self.assertEqual("", result["location_scope"])
                    self.assertEqual("", result["location_governing_meaning"])
                    self.assertNotIn(EXPECTED_RULES["snake_location"][0], result["applied_rule_ids"])

    def test_location_text_does_not_change_actions_targets_or_endings(self):
        for dream in ("A snake watched my sister in the kitchen.",
                      "The snake bit me in my house, then I killed the snake.",
                      "A snake watched me at work."):
            before, after = self.render(dream, self.old), self.render(dream)
            for field in ("action", "action_target", "outcome", "completed_bite", "venom", "applied_rule_ids", "event_graph"):
                self.assertEqual(before[field], after[field])

    def test_output_binding_consumes_canonical_text_and_keeps_safety(self):
        result = self.render("A snake was in my kitchen.")
        _, output, full = bind_snake_output_contract(
            doctrine_facts={"snake_narration": result}, seal={},
            interpretation={"spiritual_meaning": "Mock stale output"}, full_interpretation="Mock stale output",
        )
        self.assertIn(result["location_governing_meaning"], output["spiritual_meaning"])
        self.assertIn("not proof", full)
        self.assertNotIn("Mock stale output", full)

    def test_inactive_or_unverified_registry_cannot_supply_location_text(self):
        for mutation in ("inactive", "unverified"):
            snapshot = self.validate()
            if mutation == "inactive": snapshot["rules"][MOCK_AMENDED_KEYS[1]]["active"] = False
            else: snapshot["verified"] = False
            with patch("app.snake_doctrine.get_snake_registry_snapshot", return_value=snapshot):
                result = build_snake_narration_facts("A snake was in my house.")
            self.assertEqual("", result["location_governing_meaning"])

    def test_public_metadata_does_not_contain_source_text_or_approval_manifest(self):
        metadata = public_snake_registry_metadata(self.validate(), include_rule_ids=True)
        self.assertNotIn("rules", metadata)
        self.assertNotIn("rule_metadata", metadata)
        self.assertNotIn("Mock revised annotation", json.dumps(metadata))
        self.assertNotIn("Mock approval authority", json.dumps(metadata))

    def test_real_loader_uses_private_file_for_cutover_and_rollback(self):
        spreadsheet, worksheet = Mock(), Mock()
        spreadsheet.worksheet.return_value = worksheet
        try:
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "mock.private.json"
                path.write_text(json.dumps(self.manifest))
                with patch.object(Config, "APP_ENV", "production"), patch.object(Config, "SNAKE_REGISTRY_APPROVALS_FILE", str(path)), patch("app.snake_registry.get_spreadsheet", return_value=spreadsheet):
                    for values in (self.old, self.new, self.old):
                        worksheet.get_all_values.return_value = values
                        snapshot = get_snake_registry_snapshot(force=True)
                        self.assertTrue(snapshot["verified"])
                        self.assertEqual(registry_content_revision(values), snapshot["content_revision"])
                    self.assertIs(snapshot, get_snake_registry_snapshot())
                    path.write_text("null")
                    failed = get_snake_registry_snapshot(force=True)
                    self.assertFalse(failed["verified"])
                    self.assertEqual({}, failed["rules"])
                    self.assertNotIn(str(path), failed["error"])
                    path.write_text(json.dumps(self.manifest))
                    worksheet.get_all_values.side_effect = RuntimeError("mock source unavailable")
                    self.assertFalse(get_snake_registry_snapshot(force=True)["verified"])
        finally:
            SNAKE_REGISTRY_CACHE.clear()

    def test_unpublished_profile_keeps_release_gate_closed(self):
        from app.release_verification import validate_health_payload
        errors = validate_health_payload({"snake_registry": self.validate()}, expected_commit="mock-sha")
        self.assertTrue(any("snake_registry.sheet_revision" in error for error in errors))
        self.assertTrue(any("snake_registry.content_revision" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
