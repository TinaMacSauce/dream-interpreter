from typing import Any, Dict, List

from app.sheets import get_journal_worksheet
from app.utils import normalize_email


def append_dream_journal_entry(payload: Dict[str, Any]) -> Dict[str, Any]:
    from datetime import datetime
    import secrets

    ws = get_journal_worksheet()

    entry_id = payload.get("entry_id") or f"DJ-{secrets.token_hex(6).upper()}"
    created_at = datetime.utcnow().isoformat() + "Z"

    top_symbols = payload.get("top_symbols") or []
    if not isinstance(top_symbols, list):
        top_symbols = []

    row = [
        entry_id,
        created_at,
        normalize_email(payload.get("email") or ""),
        (payload.get("dream_text") or "").strip(),
        (payload.get("spiritual_meaning") or "").strip(),
        (payload.get("effects_in_physical_realm") or "").strip(),
        (payload.get("what_to_do") or "").strip(),
        (payload.get("full_interpretation") or "").strip(),
        (payload.get("receipt_id") or "").strip(),
        ", ".join([str(x).strip() for x in top_symbols if str(x).strip()]),
        (payload.get("seal_status") or "").strip(),
        (payload.get("seal_type") or "").strip(),
        (payload.get("seal_risk") or "").strip(),
        (payload.get("engine_mode") or "").strip(),
        (payload.get("access_type") or "").strip(),
        (payload.get("is_saved") or "yes").strip(),
        (payload.get("notes") or "").strip(),
    ]

    ws.append_row(row, value_input_option="RAW")
    return {"ok": True, "entry_id": entry_id, "created_at": created_at}


def get_journal_history(email: str, limit: int = 50) -> Dict[str, Any]:
    ws = get_journal_worksheet()
    rows = ws.get_all_records()

    email_n = normalize_email(email)
    matches: List[Dict[str, Any]] = []

    for row in rows:
        row_email = normalize_email(row.get("email") or "")
        if row_email != email_n:
            continue

        top_symbols_raw = (row.get("top_symbols") or "").strip()
        top_symbols = [x.strip() for x in top_symbols_raw.split(",") if x.strip()]

        matches.append(
            {
                "entry_id": row.get("entry_id", ""),
                "created_at": row.get("created_at", ""),
                "email": row_email,
                "dream_text": row.get("dream_text", ""),
                "spiritual_meaning": row.get("spiritual_meaning", ""),
                "effects_in_physical_realm": row.get("effects_in_physical_realm", ""),
                "what_to_do": row.get("what_to_do", ""),
                "full_interpretation": row.get("full_interpretation", ""),
                "receipt_id": row.get("receipt_id", ""),
                "top_symbols": top_symbols,
                "seal_status": row.get("seal_status", ""),
                "seal_type": row.get("seal_type", ""),
                "seal_risk": row.get("seal_risk", ""),
                "engine_mode": row.get("engine_mode", ""),
                "access_type": row.get("access_type", ""),
                "notes": row.get("notes", ""),
            }
        )

    matches.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return {"ok": True, "email": email_n, "entries": matches[:limit]}
