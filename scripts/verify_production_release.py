#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import time
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
    args = parser.parse_args()

    last_evidence: Dict[str, Any] = {}
    for attempt in range(1, args.attempts + 1):
        verified, evidence = probe(
            base_url=args.base_url,
            expected_commit=args.expected_commit,
            timeout=args.timeout,
        )
        evidence["attempt"] = attempt
        evidence["attempts_allowed"] = args.attempts
        print(json.dumps(evidence, sort_keys=True), flush=True)
        last_evidence = evidence
        if verified:
            return 0
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
