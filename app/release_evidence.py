from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Sequence


PARITY_MATCHED = "MATCHED"
PARITY_DEPLOYING = "DEPLOYING"
PARITY_DRIFT = "DRIFT"

VERIFICATION_HEADERS = [
    "evidence_id",
    "verified_at_utc",
    "work_item",
    "doctrine_range",
    "doctrine_version",
    "registry_source_revision",
    "content_revision",
    "approved_rule_count",
    "unresolved_rule_count",
    "doctrine_verification",
    "accessible_commit",
    "test_result",
    "deployed_version",
    "completion_status",
    "notes",
]


def classify_deployment_parity(
    expected_commit: str,
    reported_commit: str,
    *,
    elapsed_seconds: float,
    deployment_window_seconds: float,
) -> str:
    """Classify exact deployment parity without initiating a deployment."""
    expected = (expected_commit or "").strip().lower()
    reported = (reported_commit or "").strip().lower()
    if expected and reported and expected == reported:
        return PARITY_MATCHED
    if max(0.0, elapsed_seconds) <= max(0.0, deployment_window_seconds):
        return PARITY_DEPLOYING
    return PARITY_DRIFT


def successful_named_runs(
    runs: Iterable[Mapping[str, Any]],
    *,
    expected_commit: str,
    names: Sequence[str],
) -> Dict[str, Mapping[str, Any]]:
    """Return exact-SHA successful main runs for every required workflow."""
    selected: Dict[str, Mapping[str, Any]] = {}
    for run in runs:
        name = str(run.get("name") or "")
        if name not in names or name in selected:
            continue
        if str(run.get("head_sha") or "") != expected_commit:
            continue
        if run.get("event") != "push":
            continue
        if run.get("status") != "completed" or run.get("conclusion") != "success":
            continue
        selected[name] = run

    missing = [name for name in names if name not in selected]
    if missing:
        raise RuntimeError(
            "Missing successful exact-SHA main workflow runs: " + ", ".join(missing)
        )
    return selected


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{label} is missing or is not an object")
    return value


def build_verification_row(
    *,
    expected_commit: str,
    release: Mapping[str, Any],
    qa_status: Mapping[str, Any],
    teeth_case_count: int,
    snake_case_count: int,
    ci_runs: Mapping[str, Mapping[str, Any]],
    smoke_run: Mapping[str, Any],
    verified_at: datetime | None = None,
) -> List[Any]:
    """Build one append-only canonical release evidence row."""
    if release.get("build_commit") != expected_commit:
        raise RuntimeError("Production release commit does not match expected main commit")

    teeth = _require_mapping(qa_status.get("doctrine_registry"), "Teeth registry")
    snake = _require_mapping(
        qa_status.get("snake_doctrine_registry"),
        "Snake registry",
    )
    if teeth.get("verified") is not True or snake.get("verified") is not True:
        raise RuntimeError("Canonical doctrine registries are not verified")

    qa_access = _require_mapping(qa_status.get("qa_access"), "QA access")
    if qa_access.get("non_billable") is not True:
        raise RuntimeError("QA access is not non-billable")
    if qa_access.get("customer_credits_consumed") is not False:
        raise RuntimeError("QA evidence indicates customer credit consumption")

    timestamp = verified_at or datetime.now(timezone.utc)
    verified_at_utc = timestamp.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    commit_url = f"https://github.com/TinaMacSauce/dream-interpreter/commit/{expected_commit}"

    source_revisions = {
        str(teeth.get("sheet_revision") or ""),
        str(snake.get("sheet_revision") or ""),
    }
    source_revisions.discard("")
    if len(source_revisions) != 1:
        raise RuntimeError("Teeth and Snake registry source revisions do not match")
    source_revision = next(iter(source_revisions))

    python_run = ci_runs["Python tests"]
    qa_run = ci_runs["QA release gate"]
    smoke_url = str(smoke_run.get("html_url") or "")
    return [
        f"RELEASE-{expected_commit[:12].upper()}",
        verified_at_utc,
        f"Automated production verification: {release.get('release_id')}",
        "DoctrineRegistry!A1:M42",
        f"{teeth.get('doctrine_version')}; {snake.get('doctrine_version')}",
        source_revision,
        (
            f"Teeth={teeth.get('content_revision')}; "
            f"Snake={snake.get('content_revision')}"
        ),
        int(teeth.get("active_rule_count") or 0)
        + int(snake.get("active_rule_count") or 0),
        int(teeth.get("unresolved_rule_count") or 0)
        + int(snake.get("unresolved_rule_count") or 0),
        "PASS: exact cluster-scoped Teeth and Snake registries verified from the canonical Sheet",
        commit_url,
        (
            f"PASS: exact main SHA; Python tests {python_run.get('id')} and "
            f"QA release gate {qa_run.get('id')}"
        ),
        (
            f"PASS: MATCHED production {expected_commit}; /version, /live, /health, "
            f"/qa/status, Teeth {teeth_case_count}/{teeth_case_count}, "
            f"Snake {snake_case_count}/{snake_case_count}; smoke {smoke_run.get('id')} "
            f"{smoke_url}"
        ),
        "RELEASE VERIFIED / INDEPENDENT QA PENDING",
        (
            "Automatically appended only after exact-SHA main CI, deployment identity, "
            "production smoke, and both fixed non-billable regressions passed. No customer "
            "credits consumed. Token-protected /qa/interpret remains separate independent QA."
        ),
    ]


def record_verification_row(spreadsheet: Any, worksheet: Any, row: Sequence[Any]) -> Dict[str, Any]:
    """Append one row atomically, preserving format and refusing conflicting evidence."""
    values = worksheet.get_all_values()
    if not values or values[0][: len(VERIFICATION_HEADERS)] != VERIFICATION_HEADERS:
        raise RuntimeError("VerificationLog header contract mismatch")
    if len(row) != len(VERIFICATION_HEADERS):
        raise RuntimeError("VerificationLog row width mismatch")

    evidence_id = str(row[0])
    commit_url = str(row[10])
    for index, existing in enumerate(values[1:], start=2):
        existing_id = existing[0] if existing else ""
        existing_commit = existing[10] if len(existing) > 10 else ""
        if existing_id == evidence_id and existing_commit != commit_url:
            raise RuntimeError(f"Conflicting VerificationLog evidence ID at row {index}")
        if existing_commit == commit_url:
            return {"status": "ALREADY_RECORDED", "row": index}

    target_row = len(values) + 1
    sheet_id = worksheet.id
    request_values = []
    for value in row:
        if isinstance(value, bool):
            entered = {"boolValue": value}
        elif isinstance(value, (int, float)):
            entered = {"numberValue": value}
        else:
            entered = {"stringValue": str(value)}
        request_values.append({"userEnteredValue": entered})

    requests: List[Dict[str, Any]] = []
    if target_row > 2:
        requests.append(
            {
                "copyPaste": {
                    "source": {
                        "sheetId": sheet_id,
                        "startRowIndex": target_row - 2,
                        "endRowIndex": target_row - 1,
                        "startColumnIndex": 0,
                        "endColumnIndex": len(VERIFICATION_HEADERS),
                    },
                    "destination": {
                        "sheetId": sheet_id,
                        "startRowIndex": target_row - 1,
                        "endRowIndex": target_row,
                        "startColumnIndex": 0,
                        "endColumnIndex": len(VERIFICATION_HEADERS),
                    },
                    "pasteType": "PASTE_FORMAT",
                    "pasteOrientation": "NORMAL",
                }
            }
        )
    requests.append(
        {
            "updateCells": {
                "range": {
                    "sheetId": sheet_id,
                    "startRowIndex": target_row - 1,
                    "endRowIndex": target_row,
                    "startColumnIndex": 0,
                    "endColumnIndex": len(VERIFICATION_HEADERS),
                },
                "rows": [{"values": request_values}],
                "fields": "userEnteredValue",
            }
        }
    )
    spreadsheet.batch_update({"requests": requests})

    written = worksheet.row_values(target_row)
    if written[: len(VERIFICATION_HEADERS)] != [str(value) for value in row]:
        raise RuntimeError(f"VerificationLog readback mismatch at row {target_row}")
    return {"status": "APPENDED", "row": target_row}
