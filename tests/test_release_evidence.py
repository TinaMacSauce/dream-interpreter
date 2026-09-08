import unittest
from datetime import datetime, timezone

from app.release_evidence import (
    PARITY_DEPLOYING,
    PARITY_DRIFT,
    PARITY_MATCHED,
    VERIFICATION_HEADERS,
    build_verification_row,
    classify_deployment_parity,
    record_verification_row,
    successful_named_runs,
)


COMMIT = "a" * 40


class FakeSpreadsheet:
    def __init__(self):
        self.batches = []

    def batch_update(self, body):
        self.batches.append(body)


class FakeWorksheet:
    id = 2046090306

    def __init__(self, values, readback=None):
        self.values = values
        self.readback = readback

    def get_all_values(self):
        return self.values

    def row_values(self, row):
        return self.readback or []


def run(name, run_id):
    return {
        "id": run_id,
        "name": name,
        "head_sha": COMMIT,
        "event": "push",
        "status": "completed",
        "conclusion": "success",
    }


class ReleaseEvidenceTests(unittest.TestCase):
    def test_parity_states_are_exact_and_time_bounded(self):
        self.assertEqual(
            PARITY_MATCHED,
            classify_deployment_parity(
                COMMIT, COMMIT, elapsed_seconds=9999, deployment_window_seconds=600
            ),
        )
        self.assertEqual(
            PARITY_DEPLOYING,
            classify_deployment_parity(
                COMMIT, "b" * 40, elapsed_seconds=599, deployment_window_seconds=600
            ),
        )
        self.assertEqual(
            PARITY_DRIFT,
            classify_deployment_parity(
                COMMIT, "b" * 40, elapsed_seconds=601, deployment_window_seconds=600
            ),
        )

    def test_named_runs_require_exact_sha_successful_main_pushes(self):
        selected = successful_named_runs(
            [run("Python tests", 1), run("QA release gate", 2)],
            expected_commit=COMMIT,
            names=("Python tests", "QA release gate"),
        )
        self.assertEqual({"Python tests", "QA release gate"}, set(selected))

        bad = run("QA release gate", 2)
        bad["conclusion"] = "failure"
        with self.assertRaisesRegex(RuntimeError, "QA release gate"):
            successful_named_runs(
                [run("Python tests", 1), bad],
                expected_commit=COMMIT,
                names=("Python tests", "QA release gate"),
            )

    def test_build_row_preserves_registry_and_nonbillable_evidence(self):
        release = {"build_commit": COMMIT, "release_id": "release-observability-v1"}
        registry = {
            "verified": True,
            "sheet_revision": "6138",
            "content_revision": "fnv1a64:teeth",
            "doctrine_version": "DEC-TEETH",
            "active_rule_count": 17,
            "unresolved_rule_count": 6,
        }
        snake = dict(
            registry,
            content_revision="fnv1a64:snake",
            doctrine_version="DEC-SNAKE",
            active_rule_count=18,
            unresolved_rule_count=0,
        )
        qa_status = {
            "doctrine_registry": registry,
            "snake_doctrine_registry": snake,
            "qa_access": {
                "non_billable": True,
                "customer_credits_consumed": False,
            },
        }
        row = build_verification_row(
            expected_commit=COMMIT,
            release=release,
            qa_status=qa_status,
            teeth_case_count=61,
            snake_case_count=20,
            ci_runs={"Python tests": run("Python tests", 10), "QA release gate": run("QA release gate", 11)},
            smoke_run={"id": 12, "html_url": "https://example.test/12"},
            verified_at=datetime(2026, 9, 8, 9, 0, tzinfo=timezone.utc),
        )
        self.assertEqual(len(VERIFICATION_HEADERS), len(row))
        self.assertEqual(35, row[7])
        self.assertEqual(6, row[8])
        self.assertIn("MATCHED", row[12])
        self.assertIn("No customer credits consumed", row[14])

    def test_append_is_atomic_formatted_and_idempotent(self):
        row = [f"v{i}" for i in range(len(VERIFICATION_HEADERS))]
        row[7] = 35
        row[8] = 6
        sheet = FakeSpreadsheet()
        worksheet = FakeWorksheet([VERIFICATION_HEADERS, ["old"]], [str(v) for v in row])
        result = record_verification_row(sheet, worksheet, row)
        self.assertEqual("APPENDED", result["status"])
        requests = sheet.batches[0]["requests"]
        self.assertEqual("PASTE_FORMAT", requests[0]["copyPaste"]["pasteType"])
        self.assertEqual(35, requests[1]["updateCells"]["rows"][0]["values"][7]["userEnteredValue"]["numberValue"])

        existing = [""] * len(VERIFICATION_HEADERS)
        existing[10] = str(row[10])
        duplicate_sheet = FakeSpreadsheet()
        duplicate = record_verification_row(
            duplicate_sheet,
            FakeWorksheet([VERIFICATION_HEADERS, existing]),
            row,
        )
        self.assertEqual("ALREADY_RECORDED", duplicate["status"])
        self.assertEqual([], duplicate_sheet.batches)

    def test_conflicting_id_fails_closed(self):
        row = [f"v{i}" for i in range(len(VERIFICATION_HEADERS))]
        existing = [""] * len(VERIFICATION_HEADERS)
        existing[0] = row[0]
        existing[10] = "different"
        with self.assertRaisesRegex(RuntimeError, "Conflicting"):
            record_verification_row(
                FakeSpreadsheet(),
                FakeWorksheet([VERIFICATION_HEADERS, existing]),
                row,
            )


if __name__ == "__main__":
    unittest.main()
