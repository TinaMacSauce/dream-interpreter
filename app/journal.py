from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List

import secrets

from app.sheets import get_journal_worksheet
from app.utils import normalize_email


MAX_HISTORY_LIMIT = 100
DEFAULT_HISTORY_LIMIT = 50


def utc_now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _safe_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [
            str(x).strip()
            for x in value
            if str(x).strip()
        ]

    if isinstance(value, str):
        return [
            x.strip()
            for x in value.split(",")
            if x.strip()
        ]

    return []


def _build_entry_id() -> str:
    return f"DJ-{secrets.token_hex(6).upper()}"


def append_dream_journal_entry(
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    ws = get_journal_worksheet()

    entry_id = (
        _safe_text(
            payload.get("entry_id")
        )
        or _build_entry_id()
    )

    created_at = utc_now_iso()

    top_symbols = _safe_list(
        payload.get("top_symbols")
    )

    row = [
        entry_id,
        created_at,
        normalize_email(
            payload.get("email") or ""
        ),
        _safe_text(
            payload.get("dream_text")
        ),
        _safe_text(
            payload.get("spiritual_meaning")
        ),
        _safe_text(
            payload.get(
                "effects_in_physical_realm"
            )
        ),
        _safe_text(
            payload.get("what_to_do")
        ),
        _safe_text(
            payload.get(
                "full_interpretation"
            )
        ),
        _safe_text(
            payload.get("receipt_id")
        ),
        ", ".join(top_symbols),
        _safe_text(
            payload.get("seal_status")
        ),
        _safe_text(
            payload.get("seal_type")
        ),
        _safe_text(
            payload.get("seal_risk")
        ),
        _safe_text(
            payload.get("engine_mode")
        ),
        _safe_text(
            payload.get("access_type")
        ),
        _safe_text(
            payload.get("is_saved")
            or "yes"
        ),
        _safe_text(
            payload.get("notes")
        ),
    ]

    ws.append_row(
        row,
        value_input_option="RAW",
    )

    return {
        "ok": True,
        "entry_id": entry_id,
        "created_at": created_at,
    }


def _normalize_history_row(
    row: Dict[str, Any],
) -> Dict[str, Any]:
    top_symbols = _safe_list(
        row.get("top_symbols")
    )

    return {
        "entry_id": _safe_text(
            row.get("entry_id")
        ),
        "created_at": _safe_text(
            row.get("created_at")
        ),
        "email": normalize_email(
            row.get("email") or ""
        ),
        "dream_text": _safe_text(
            row.get("dream_text")
        ),
        "spiritual_meaning": _safe_text(
            row.get("spiritual_meaning")
        ),
        "effects_in_physical_realm": _safe_text(
            row.get(
                "effects_in_physical_realm"
            )
        ),
        "what_to_do": _safe_text(
            row.get("what_to_do")
        ),
        "full_interpretation": _safe_text(
            row.get(
                "full_interpretation"
            )
        ),
        "receipt_id": _safe_text(
            row.get("receipt_id")
        ),
        "top_symbols": top_symbols,
        "seal_status": _safe_text(
            row.get("seal_status")
        ),
        "seal_type": _safe_text(
            row.get("seal_type")
        ),
        "seal_risk": _safe_text(
            row.get("seal_risk")
        ),
        "engine_mode": _safe_text(
            row.get("engine_mode")
        ),
        "access_type": _safe_text(
            row.get("access_type")
        ),
        "notes": _safe_text(
            row.get("notes")
        ),
    }


def get_journal_history(
    email: str,
    limit: int = DEFAULT_HISTORY_LIMIT,
) -> Dict[str, Any]:
    ws = get_journal_worksheet()

    rows = ws.get_all_records()

    email_n = normalize_email(email)

    try:
        limit = int(limit)
    except Exception:
        limit = DEFAULT_HISTORY_LIMIT

    limit = max(
        1,
        min(limit, MAX_HISTORY_LIMIT),
    )

    matches: List[Dict[str, Any]] = []

    for row in rows:
        if not isinstance(row, dict):
            continue

        row_email = normalize_email(
            row.get("email") or ""
        )

        if row_email != email_n:
            continue

        matches.append(
            _normalize_history_row(row)
        )

    matches.sort(
        key=lambda x: (
            x.get("created_at") or ""
        ),
        reverse=True,
    )

    return {
        "ok": True,
        "email": email_n,
        "count": len(matches),
        "entries": matches[:limit],
    }
