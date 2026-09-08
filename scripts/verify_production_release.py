#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


_VALIDATOR_PATH = Path(__file__).resolve().parents[1] / "app" / "release_verification.py"
_VALIDATOR_SPEC = importlib.util.spec_from_file_location(
    "jts_release_verification",
    _VALIDATOR_PATH,
)
if _VALIDATOR_SPEC is None or _VALIDATOR_SPEC.loader is None:
    raise RuntimeError(f"Could not load release validator from {_VALIDATOR_PATH}")
_VALIDATOR_MODULE = importlib.util.module_from_spec(_VALIDATOR_SPEC)
_VALIDATOR_SPEC.loader.exec_module(_VALIDATOR_MODULE)
validate_health_payload = _VALIDATOR_MODULE.validate_health_payload
validate_live_payload = _VALIDATOR_MODULE.validate_live_payload
validate_qa_status_payload = _VALIDATOR_MODULE.validate_qa_status_payload
validate_version_payload = _VALIDATOR_MODULE.validate_version_payload

_PARITY_PATH = Path(__file__).resolve().parents[1] / "app" / "release_evidence.py"
_PARITY_SPEC = importlib.util.spec_from_file_location("jts_release_evidence", _PARITY_PATH)
if _PARITY_SPEC is None or _PARITY_SPEC.loader is None:
    raise RuntimeError(f"Could not load parity classifier from {_PARITY_PATH}")
_PARITY_MODULE = importlib.util.module_from_spec(_PARITY_SPEC)
_PARITY_SPEC.loader.exec_module(_PARITY_MODULE)
classify_deployment_parity = _PARITY_MODULE.classify_deployment_parity


Validator = Callable[..., List[str]]


def fetch_json(
    url: str,
    timeout: float,
    *,
    method: str = "GET",
    body: Dict[str, Any] | None = None,
) -> Tuple[int, Dict[str, Any]]:
    data = None
    headers = {"Accept": "application/json"}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = Request(url, headers=headers, data=data, method=method)
    try:
        with urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
            return response.status, payload
    except HTTPError as error:
        payload = json.loads(error.read().decode("utf-8"))
        return error.code, payload


def probe(
    *,
    base_url: str,
    expected_commit: str,
    timeout: float,
    elapsed_seconds: float = 0.0,
    deployment_window_seconds: float = 600.0,
) -> Tuple[bool, Dict[str, Any]]:
    evidence: Dict[str, Any] = {
        "base_url": base_url,
        "expected_commit": expected_commit,
        "probes": {},
    }
    checks: Tuple[Tuple[str, Validator], ...] = (
        ("live", validate_live_payload),
        ("health", validate_health_payload),
        ("version", validate_version_payload),
        ("qa/status", validate_qa_status_payload),
    )
    all_errors: List[str] = []

    for path, validator in checks:
        url = f"{base_url.rstrip('/')}/{path}"
        try:
            status, payload = fetch_json(url, timeout)
            errors = validator(payload, expected_commit=expected_commit)
            if status != 200:
                errors.insert(0, f"HTTP status expected 200, got {status}")
            evidence["probes"][path] = {
                "url": url,
                "http_status": status,
                "payload": payload,
                "errors": errors,
            }
            all_errors.extend(f"{path}: {error}" for error in errors)
        except (URLError, TimeoutError, json.JSONDecodeError) as exc:
            error = f"{type(exc).__name__}: {exc}"
            evidence["probes"][path] = {"url": url, "errors": [error]}
            all_errors.append(f"{path}: {error}")

    version_release = evidence["probes"].get("version", {}).get("payload", {}).get("release", {})
    reported_commit = (
        version_release.get("build_commit", "")
        if isinstance(version_release, dict)
        else ""
    )
    parity_state = classify_deployment_parity(
        expected_commit,
        reported_commit,
        elapsed_seconds=elapsed_seconds,
        deployment_window_seconds=deployment_window_seconds,
    )
    evidence["deployment_parity"] = {
        "state": parity_state,
        "expected_commit": expected_commit,
        "reported_commit": reported_commit,
        "elapsed_seconds": round(max(0.0, elapsed_seconds), 3),
        "deployment_window_seconds": deployment_window_seconds,
    }

    denial_checks = (
        (
            "qa/grant-denied",
            "admin/qa-grant",
            {"email": "release-probe@qa.jamaicantruestories.com"},
            lambda payload: (
                payload.get("ok") is False and payload.get("error") == "Forbidden"
            ),
        ),
        (
            "qa/interpret-denied",
            "qa/interpret",
            {"dream": "My tooth fell out."},
            lambda payload: (
                payload.get("blocked") is True
                and payload.get("reason") == "missing_token"
            ),
        ),
    )
    for label, path, body, payload_is_denied in denial_checks:
        denied_url = f"{base_url.rstrip('/')}/{path}"
        try:
            status, payload = fetch_json(
                denied_url,
                timeout,
                method="POST",
                body=body,
            )
            errors = []
            if status != 403:
                errors.append(f"HTTP status expected 403, got {status}")
            if not payload_is_denied(payload):
                errors.append("unauthenticated request was not rejected safely")
            evidence["probes"][label] = {
                "url": denied_url,
                "http_status": status,
                "payload": payload,
                "errors": errors,
            }
            all_errors.extend(f"{label}: {error}" for error in errors)
        except (URLError, TimeoutError, json.JSONDecodeError) as exc:
            error = f"{type(exc).__name__}: {exc}"
            evidence["probes"][label] = {
                "url": denied_url,
                "errors": [error],
            }
            all_errors.append(f"{label}: {error}")

    evidence["errors"] = all_errors
    evidence["verified"] = not all_errors
    return not all_errors, evidence


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Wait for Render to serve an exact JTS release commit."
    )
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--attempts", type=int, default=40)
    parser.add_argument("--delay", type=float, default=15.0)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument(
        "--deployment-window",
        type=float,
        default=600.0,
        help="Normal Render deployment window in seconds; default 600 (10 minutes).",
    )
    parser.add_argument(
        "--deployment-started-at",
        help="UTC ISO-8601 timestamp for the release start; defaults to verifier start.",
    )
    args = parser.parse_args()

    verifier_started = datetime.now(timezone.utc)
    deployment_started = verifier_started
    if args.deployment_started_at:
        deployment_started = datetime.fromisoformat(
            args.deployment_started_at.replace("Z", "+00:00")
        )
        if deployment_started.tzinfo is None:
            deployment_started = deployment_started.replace(tzinfo=timezone.utc)

    last_evidence: Dict[str, Any] = {}
    for attempt in range(1, args.attempts + 1):
        verified, evidence = probe(
            base_url=args.base_url,
            expected_commit=args.expected_commit,
            timeout=args.timeout,
            elapsed_seconds=(datetime.now(timezone.utc) - deployment_started).total_seconds(),
            deployment_window_seconds=args.deployment_window,
        )
        evidence["attempt"] = attempt
        evidence["attempts_allowed"] = args.attempts
        print(json.dumps(evidence, sort_keys=True), flush=True)
        last_evidence = evidence
        if verified:
            return 0
        if evidence.get("deployment_parity", {}).get("state") == "DRIFT":
            break
        if attempt < args.attempts:
            time.sleep(args.delay)

    print(
        "Production did not reach the expected release contract: "
        + json.dumps(last_evidence.get("errors", [])),
        flush=True,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
