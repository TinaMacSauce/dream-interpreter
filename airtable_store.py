# airtable_store.py
# Airtable storage layer for JTS Dream Interpreter
# - Subscribers (paid users)
# - Usage (free usage counts + email gate)
# - Receipts (viral dream receipts)

import os
import time
import requests
from typing import Any, Dict, Optional

AIRTABLE_API_KEY = os.environ.get("AIRTABLE_API_KEY", "").strip()
AIRTABLE_BASE_ID = os.environ.get("AIRTABLE_BASE_ID", "").strip()

SUBSCRIBERS_TABLE = os.environ.get("AIRTABLE_SUBSCRIBERS_TABLE", "Subscribers").strip()
USAGE_TABLE = os.environ.get("AIRTABLE_USAGE_TABLE", "Usage").strip()
RECEIPTS_TABLE = os.environ.get("AIRTABLE_RECEIPTS_TABLE", "Receipts").strip()

API_ROOT = "https://api.airtable.com/v0"


def _assert_config():
    if not AIRTABLE_API_KEY:
        raise RuntimeError("Missing AIRTABLE_API_KEY env var.")
    if not AIRTABLE_BASE_ID:
        raise RuntimeError("Missing AIRTABLE_BASE_ID env var.")


def _headers() -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {AIRTABLE_API_KEY}",
        "Content-Type": "application/json",
    }


def _table_url(table_name: str) -> str:
    # Airtable accepts raw names; requests will handle encoding
    return f"{API_ROOT}/{AIRTABLE_BASE_ID}/{table_name}"


def _request(method: str, url: str, **kwargs) -> requests.Response:
    # light retry for Airtable throttles
    for attempt in range(3):
        resp = requests.request(method, url, headers=_headers(), timeout=20, **kwargs)
        if resp.status_code in (429, 500, 502, 503, 504):
            time.sleep(0.8 * (attempt + 1))
            continue
        return resp
    return resp


def _find_one_by_field(table: str, field_name: str, value: str) -> Optional[Dict[str, Any]]:
    _assert_config()
    value = (value or "").strip()
    if not value:
        return None

    # filterByFormula exact match:
    # {email}="test@example.com"
    formula = f'{{{field_name}}}="{value}"'
    url = _table_url(table)
    resp = _request("GET", url, params={"maxRecords": 1, "filterByFormula": formula})
    if resp.status_code != 200:
        return None
    data = resp.json()
    records = data.get("records", [])
    return records[0] if records else None


def _create_record(table: str, fields: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    _assert_config()
    url = _table_url(table)
    resp = _request("POST", url, json={"fields": fields})
    if resp.status_code not in (200, 201):
        return None
    return resp.json()


def _update_record(table: str, record_id: str, fields: Dict[str, Any]) -> bool:
    _assert_config()
    url = f"{_table_url(table)}/{record_id}"
    resp = _request("PATCH", url, json={"fields": fields})
    return resp.status_code in (200, 201)


# =========================
# Subscribers
# =========================
def is_paid(email: str) -> bool:
    email = (email or "").strip().lower()
    if not email:
        return False

    rec = _find_one_by_field(SUBSCRIBERS_TABLE, "email", email)
    if not rec:
        return False

    fields = rec.get("fields", {})
    status = str(fields.get("status", "")).strip().lower()
    return status == "pro"


def mark_paid(email: str) -> bool:
    email = (email or "").strip().lower()
    if not email:
        return False

    rec = _find_one_by_field(SUBSCRIBERS_TABLE, "email", email)
    if rec:
        return _update_record(SUBSCRIBERS_TABLE, rec["id"], {"status": "pro"})
    created = _create_record(SUBSCRIBERS_TABLE, {"email": email, "status": "pro"})
    return bool(created)


# =========================
# Usage
# =========================
def get_usage(email: str) -> Dict[str, Any]:
    """
    Returns: {"free_used": int, "email_captured": bool}
    Expects Usage table fields:
      - email (text)
      - free_used (number)
      - email_captured (checkbox or text)
    """
    email = (email or "").strip().lower()
    if not email:
        return {"free_used": 0, "email_captured": False}

    rec = _find_one_by_field(USAGE_TABLE, "email", email)
    if not rec:
        return {"free_used": 0, "email_captured": True}  # they provided email, so captured

    fields = rec.get("fields", {})
    free_used = int(fields.get("free_used") or 0)
    email_captured = bool(fields.get("email_captured", True))
    return {"free_used": free_used, "email_captured": email_captured}


def set_email_captured(email: str) -> bool:
    email = (email or "").strip().lower()
    if not email:
        return False

    rec = _find_one_by_field(USAGE_TABLE, "email", email)
    if rec:
        return _update_record(USAGE_TABLE, rec["id"], {"email_captured": True})
    created = _create_record(USAGE_TABLE, {"email": email, "free_used": 0, "email_captured": True})
    return bool(created)


def increment_free_used(email: str) -> int:
    """
    Increments Usage.free_used by 1 and returns the new value.
    """
    email = (email or "").strip().lower()
    if not email:
        return 0

    rec = _find_one_by_field(USAGE_TABLE, "email", email)
    if not rec:
        created = _create_record(USAGE_TABLE, {"email": email, "free_used": 1, "email_captured": True})
        if not created:
            return 0
        return 1

    fields = rec.get("fields", {})
    current = int(fields.get("free_used") or 0)
    new_val = current + 1
    ok = _update_record(USAGE_TABLE, rec["id"], {"free_used": new_val, "email_captured": True})
    return new_val if ok else current


# =========================
# Receipts
# =========================
def save_receipt(receipt: Dict[str, Any], email: Optional[str] = None) -> Optional[str]:
    """
    Saves a receipt to Airtable.
    Expects Receipts table fields (recommended):
      - receipt_id (text)
      - email (text) [optional]
      - dream_type (text)
      - risk_level (text)
      - code_name (text)
      - seal_summary (long text)
      - share_phrase (long text)
      - created_at (date/time) [optional; Airtable can auto-add Created time]
    """
    rid = str(receipt.get("receipt_id") or "").strip()
    if not rid:
        return None

    fields = {
        "receipt_id": rid,
        "email": (email or "").strip().lower() or None,
        "dream_type": receipt.get("dream_type"),
        "risk_level": receipt.get("risk_level"),
        "code_name": receipt.get("code_name"),
        "seal_summary": receipt.get("seal_summary"),
        "share_phrase": receipt.get("share_phrase"),
    }

    # remove None values so Airtable doesn't complain about type mismatches
    fields = {k: v for k, v in fields.items() if v is not None}

    rec = _find_one_by_field(RECEIPTS_TABLE, "receipt_id", rid)
    if rec:
        _update_record(RECEIPTS_TABLE, rec["id"], fields)
        return rid

    created = _create_record(RECEIPTS_TABLE, fields)
    return rid if created else None
