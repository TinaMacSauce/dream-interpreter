from __future__ import annotations

import hashlib
import os
import secrets
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterator

from flask import request

from app.config import Config
from app.storage import ensure_json_file, load_dict_file, write_json_file_atomic
from app.utils import normalize_email


EMPTY_QA_ACCESS = {
    "active": False,
    "consumed": False,
    "grant_id": "",
    "email": "",
    "uses_remaining": 0,
    "expires_at": "",
    "reason": "missing_token",
}


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso_z(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_iso_z(value: Any) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value or "").replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _grants_path() -> Path:
    return Path(Config.QA_GRANTS_FILE)


@contextmanager
def _file_lock(path: Path) -> Iterator[None]:
    lock_path = Path(str(path) + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    with open(lock_path, "a+", encoding="utf-8") as lock_file:
        if os.name == "nt":
            import msvcrt

            lock_file.seek(0)
            lock_file.write("0")
            lock_file.flush()
            lock_file.seek(0)
            msvcrt.locking(lock_file.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                lock_file.seek(0)
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _load_unlocked(path: Path) -> Dict[str, Any]:
    ensure_json_file(path, {"grants": {}})
    payload = load_dict_file(path)
    grants = payload.get("grants") if isinstance(payload, dict) else None
    return {"grants": grants if isinstance(grants, dict) else {}}


def _save_unlocked(path: Path, payload: Dict[str, Any]) -> None:
    write_json_file_atomic(path, payload)


def issue_qa_grant(*, email: str, uses: int, hours: int) -> Dict[str, Any]:
    """Create an isolated QA entitlement and return its secret token once."""
    email_n = normalize_email(email)
    token = secrets.token_urlsafe(32)
    grant_id = f"qa-{secrets.token_hex(8)}"
    created_at = _utc_now()
    expires_at = created_at + timedelta(hours=hours)
    path = _grants_path()

    record = {
        "grant_id": grant_id,
        "email": email_n,
        "token_hash": _token_hash(token),
        "uses_remaining": int(uses),
        "created_at": _iso_z(created_at),
        "expires_at": _iso_z(expires_at),
        "revoked_at": "",
    }

    with _file_lock(path):
        payload = _load_unlocked(path)
        payload["grants"][grant_id] = record
        _save_unlocked(path, payload)

    return {
        "active": True,
        "grant_id": grant_id,
        "email": email_n,
        "token": token,
        "uses_remaining": int(uses),
        "expires_at": record["expires_at"],
    }


def qa_token_from_request() -> str:
    token = (request.headers.get("X-QA-Token") or "").strip()
    if token:
        return token

    authorization = (request.headers.get("Authorization") or "").strip()
    scheme, _, credential = authorization.partition(" ")
    if scheme.lower() == "bearer":
        return credential.strip()
    return ""


def _status_from_record(record: Dict[str, Any], now: datetime) -> Dict[str, Any]:
    if record.get("revoked_at"):
        reason = "revoked"
    elif int(record.get("uses_remaining") or 0) <= 0:
        reason = "exhausted"
    else:
        expires_at = _parse_iso_z(record.get("expires_at"))
        reason = "" if expires_at and now < expires_at else "expired"

    return {
        "active": not reason,
        "consumed": False,
        "grant_id": str(record.get("grant_id") or ""),
        "email": normalize_email(record.get("email") or ""),
        "uses_remaining": max(0, int(record.get("uses_remaining") or 0)),
        "expires_at": str(record.get("expires_at") or ""),
        "reason": reason,
    }


def get_qa_grant_status(token: str) -> Dict[str, Any]:
    token = (token or "").strip()
    if not token:
        return dict(EMPTY_QA_ACCESS)

    supplied_hash = _token_hash(token)
    path = _grants_path()
    now = _utc_now()

    with _file_lock(path):
        payload = _load_unlocked(path)
        for record in payload["grants"].values():
            stored_hash = str((record or {}).get("token_hash") or "")
            if stored_hash and secrets.compare_digest(stored_hash, supplied_hash):
                return _status_from_record(record, now)

    status = dict(EMPTY_QA_ACCESS)
    status["reason"] = "invalid_token"
    return status


def consume_qa_grant(token: str) -> Dict[str, Any]:
    """Consume one QA use after a successful interpretation only."""
    token = (token or "").strip()
    supplied_hash = _token_hash(token) if token else ""
    path = _grants_path()
    now = _utc_now()

    with _file_lock(path):
        payload = _load_unlocked(path)
        for record in payload["grants"].values():
            stored_hash = str((record or {}).get("token_hash") or "")
            if supplied_hash and stored_hash and secrets.compare_digest(stored_hash, supplied_hash):
                status = _status_from_record(record, now)
                if not status["active"]:
                    return status
                record["uses_remaining"] = status["uses_remaining"] - 1
                _save_unlocked(path, payload)
                consumed = _status_from_record(record, now)
                consumed["consumed"] = True
                return consumed

    status = dict(EMPTY_QA_ACCESS)
    status["reason"] = "invalid_token"
    return status


def revoke_qa_grant(grant_id: str) -> bool:
    grant_id = str(grant_id or "").strip()
    if not grant_id:
        return False

    path = _grants_path()
    with _file_lock(path):
        payload = _load_unlocked(path)
        record = payload["grants"].get(grant_id)
        if not isinstance(record, dict) or record.get("revoked_at"):
            return False
        record["revoked_at"] = _iso_z(_utc_now())
        _save_unlocked(path, payload)
        return True


def qa_storage_ready() -> bool:
    """Verify the isolated QA store is readable and writable without a grant."""
    path = _grants_path()
    try:
        with _file_lock(path):
            payload = _load_unlocked(path)
            _save_unlocked(path, payload)
        return True
    except (OSError, ValueError, TypeError):
        return False


def public_qa_access_metadata() -> Dict[str, Any]:
    return {
        "configured": bool(Config.ADMIN_KEY),
        "storage_ready": qa_storage_ready(),
        "grant_route": "/admin/qa-grant",
        "revoke_route": "/admin/qa-revoke",
        "interpret_route": "/qa/interpret",
        "application_route": "/interpret",
        "fixed_contract_route": "/qa/teeth-regression",
        "grant_authentication": "X-Admin-Key",
        "interpret_authentication": "X-QA-Token or Authorization Bearer",
        "email_domain": Config.QA_EMAIL_DOMAIN,
        "default_uses": Config.QA_DEFAULT_USES,
        "max_uses": Config.QA_MAX_USES,
        "default_hours": Config.QA_DEFAULT_HOURS,
        "max_hours": Config.QA_MAX_HOURS,
        "non_billable": True,
        "customer_credits_consumed": False,
        "customer_entitlement_store_used": False,
        "token_storage": "sha256_hash_only",
        "revocable": True,
    }
