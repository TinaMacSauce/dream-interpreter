#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
from typing import Any, Dict, Tuple
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import gspread
from google.oauth2.service_account import Credentials



def load_module(name: str, relative_path: str):
    path = Path(__file__).resolve().parents[1] / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_EVIDENCE = load_module("jts_release_evidence_recorder", "app/release_evidence.py")
_VALIDATION = load_module("jts_release_validation_recorder", "app/release_verification.py")
VERIFICATION_HEADERS = _EVIDENCE.VERIFICATION_HEADERS
build_verification_row = _EVIDENCE.build_verification_row
record_verification_row = _EVIDENCE.record_verification_row
successful_named_runs = _EVIDENCE.successful_named_runs
validate_health_payload = _VALIDATION.validate_health_payload
validate_live_payload = _VALIDATION.validate_live_payload
validate_qa_status_payload = _VALIDATION.validate_qa_status_payload
validate_version_payload = _VALIDATION.validate_version_payload


GOOGLE_SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]


def fetch_json(url: str, *, token: str = "", timeout: float = 30.0) -> Tuple[int, Dict[str, Any]]:
    headers = {"Accept": "application/json", "User-Agent": "jts-release-verifier"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
        headers["X-GitHub-Api-Version"] = "2022-11-28"
    request = Request(url, headers=headers)
    try:
        with urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        payload = json.loads(error.read().decode("utf-8"))
        return error.code, payload


def google_client() -> gspread.Client:
    raw = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON", "").strip()
    if not raw:
        raise RuntimeError("GOOGLE_SERVICE_ACCOUNT_JSON is required for release evidence")
    try:
        info = json.loads(raw)
        private_key = info.get("private_key")
        if isinstance(private_key, str):
            info["private_key"] = private_key.replace("\\n", "\n")
        credentials = Credentials.from_service_account_info(info, scopes=GOOGLE_SCOPES)
    except Exception as exc:
        raise RuntimeError(f"Invalid GOOGLE_SERVICE_ACCOUNT_JSON: {exc}") from exc
    return gspread.authorize(credentials)


def open_verification_log(spreadsheet_id: str, worksheet_name: str):
    spreadsheet = google_client().open_by_key(spreadsheet_id)
    worksheet = spreadsheet.worksheet(worksheet_name)
    headers = worksheet.row_values(1)
    if headers[: len(VERIFICATION_HEADERS)] != VERIFICATION_HEADERS:
        raise RuntimeError("VerificationLog header contract mismatch")
    return spreadsheet, worksheet


def require_ok(status: int, payload: Dict[str, Any], label: str) -> Dict[str, Any]:
    if status != 200:
        raise RuntimeError(f"{label} returned HTTP {status}: {payload}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Append exact release evidence after smoke success.")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--base-url", default="https://interpreter.jamaicantruestories.com")
    parser.add_argument("--expected-commit")
    parser.add_argument("--smoke-run-id", type=int)
    parser.add_argument("--repository", default="TinaMacSauce/dream-interpreter")
    parser.add_argument("--spreadsheet-id", default=os.getenv("SPREADSHEET_ID", ""))
    parser.add_argument("--worksheet", default="VerificationLog")
    args = parser.parse_args()

    if not args.spreadsheet_id:
        raise RuntimeError("SPREADSHEET_ID is required")
    spreadsheet, worksheet = open_verification_log(args.spreadsheet_id, args.worksheet)
    if args.preflight:
        print(json.dumps({"verified": True, "worksheet": args.worksheet}))
        return 0
    if not args.expected_commit or not args.smoke_run_id:
        raise RuntimeError("--expected-commit and --smoke-run-id are required")

    token = os.getenv("GITHUB_TOKEN", "").strip()
    api_root = f"https://api.github.com/repos/{args.repository}"
    main_status, main_commit = fetch_json(f"{api_root}/commits/main", token=token)
    require_ok(main_status, main_commit, "GitHub main commit")
    if main_commit.get("sha") != args.expected_commit:
        raise RuntimeError(
            f"DRIFT: GitHub main is {main_commit.get('sha')}, expected {args.expected_commit}"
        )

    query = urlencode({"head_sha": args.expected_commit, "per_page": 100})
    runs_status, runs_payload = fetch_json(f"{api_root}/actions/runs?{query}", token=token)
    require_ok(runs_status, runs_payload, "GitHub workflow runs")
    ci_runs = successful_named_runs(
        runs_payload.get("workflow_runs", []),
        expected_commit=args.expected_commit,
        names=("Python tests", "QA release gate"),
    )

    smoke_status, smoke_run = fetch_json(
        f"{api_root}/actions/runs/{args.smoke_run_id}",
        token=token,
    )
    require_ok(smoke_status, smoke_run, "GitHub production smoke run")
    if (
        smoke_run.get("name") != "Production release smoke"
        or smoke_run.get("head_sha") != args.expected_commit
        or smoke_run.get("status") != "completed"
        or smoke_run.get("conclusion") != "success"
    ):
        raise RuntimeError("Production release smoke is not successful on the exact SHA")

    base = args.base_url.rstrip("/")
    production: Dict[str, Dict[str, Any]] = {}
    validators = {
        "version": validate_version_payload,
        "live": validate_live_payload,
        "health": validate_health_payload,
        "qa/status": validate_qa_status_payload,
    }
    for path, validator in validators.items():
        status, payload = fetch_json(f"{base}/{path}")
        require_ok(status, payload, path)
        errors = validator(payload, expected_commit=args.expected_commit)
        if errors:
            raise RuntimeError(f"{path} contract failed: {errors}")
        production[path] = payload

    case_counts: Dict[str, int] = {}
    for label, path in (("teeth", "qa/teeth-regression"), ("snake", "qa/snake-regression")):
        status, payload = fetch_json(f"{base}/{path}", timeout=120.0)
        require_ok(status, payload, path)
        if label == "snake" and payload.get("contract_pass") is not True:
            raise RuntimeError(
                f"{path} contract failed: {payload.get('failure_reason') or 'unspecified'}"
            )
        release = payload.get("release") or {}
        if release.get("build_commit") != args.expected_commit:
            raise RuntimeError(f"{path} is not serving the expected commit")
        count = payload.get("case_count")
        if not isinstance(count, int) or count < 1 or len(payload.get("cases") or []) != count:
            raise RuntimeError(f"{path} case contract is incomplete")
        if label == "snake" and (
            payload.get("non_billable") is not True
            or payload.get("customer_credits_consumed") is not False
        ):
            raise RuntimeError("Snake regression is not explicitly non-billable")
        case_counts[label] = count

    row = build_verification_row(
        expected_commit=args.expected_commit,
        release=production["version"]["release"],
        qa_status=production["qa/status"],
        teeth_case_count=case_counts["teeth"],
        snake_case_count=case_counts["snake"],
        ci_runs=ci_runs,
        smoke_run=smoke_run,
    )
    result = record_verification_row(spreadsheet, worksheet, row)
    print(json.dumps({"verified": True, "parity": "MATCHED", **result}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
