from typing import Any, Dict, List, Optional

import gspread
from flask import make_response, request, Response

from app.cache import invalidate_all_caches
from app.config import Config
from app.sheets import get_or_create_worksheet
from app.utils import normalize_header, normalize_text


def get_admin_key_from_request() -> str:
    query_key = (request.args.get("key") or "").strip()
    header_key = (request.headers.get("X-Admin-Key") or "").strip()
    return query_key or header_key


def require_admin() -> Optional[Response]:
    if not Config.ADMIN_KEY:
        return make_response("Admin is not configured.", 403)

    provided = get_admin_key_from_request()
    if not provided or provided != Config.ADMIN_KEY:
        return make_response("Forbidden", 403)

    return None


def build_col_map(header_row: List[str]) -> Dict[str, int]:
    col_map: Dict[str, int] = {}
    for idx, header in enumerate(header_row, start=1):
        col_map[normalize_header(header)] = idx
    return col_map


def find_existing_row_index_by_column(ws, lookup_value: str, lookup_col: int) -> Optional[int]:
    target = normalize_text(lookup_value)
    if not target:
        return None

    col_vals = ws.col_values(lookup_col)
    for i, value in enumerate(col_vals[1:], start=2):
        if normalize_text(value) == target:
            return i

    return None


def sheet_primary_lookup_headers(sheet_name: str) -> List[str]:
    if sheet_name == Config.SHEET_BASE_SYMBOLS:
        return ["symbol", "input"]
    if sheet_name == Config.SHEET_BEHAVIOR_RULES:
        return ["behavior_name", "behavior"]
    if sheet_name == Config.SHEET_SIZE_STATE_RULES:
        return ["state_name", "state"]
    if sheet_name == Config.SHEET_LOCATION_RULES:
        return ["location_name", "location"]
    if sheet_name == Config.SHEET_RELATIONSHIP_RULES:
        return ["relationship_name", "relationship"]
    if sheet_name == Config.SHEET_OVERRIDE_RULES:
        return ["override_name", "condition"]
    if sheet_name == Config.SHEET_OUTPUT_TEMPLATES:
        return ["template_type"]
    return ["input", "symbol"]


def sanitize_keywords_for_storage(raw: str) -> str:
    import re

    parts = re.split(r"[,|;]+", raw or "")
    cleaned = []
    seen = set()

    for part in parts:
        token = normalize_text(part)
        if not token:
            continue
        if token in seen:
            continue
        seen.add(token)
        cleaned.append(token)

    return ", ".join(cleaned)


def admin_upsert_generic_sheet(sheet_name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    ws = get_or_create_worksheet(sheet_name, rows=2000, cols=30)
    header_row = ws.row_values(1)

    if not header_row:
        raise RuntimeError(f"Sheet {sheet_name} has no header row.")

    col_map = build_col_map(header_row)
    mode = (payload.get("mode") or "upsert").strip().lower()

    lookup_col = None
    lookup_value = ""

    for key in sheet_primary_lookup_headers(sheet_name):
        nk = normalize_header(key)
        if col_map.get(nk) and str(payload.get(key, "")).strip():
            lookup_col = col_map[nk]
            lookup_value = str(payload.get(key, "")).strip()
            break

    if not lookup_col or not lookup_value:
        raise RuntimeError(f"Missing primary lookup field for sheet {sheet_name}.")

    existing_row = None
    if mode != "add":
        existing_row = find_existing_row_index_by_column(ws, lookup_value, lookup_col)

    row_index = existing_row if existing_row else len(ws.get_all_values()) + 1
    action = "updated" if existing_row else "added"

    updates = []
    for raw_header in header_row:
        normalized_header = normalize_header(raw_header)
        chosen_value = None

        for payload_key, payload_val in payload.items():
            if normalize_header(payload_key) == normalized_header:
                chosen_value = payload_val
                break

        if chosen_value is None:
            continue

        if normalized_header == "keywords":
            chosen_value = sanitize_keywords_for_storage(str(chosen_value))
        else:
            chosen_value = str(chosen_value).strip()

        col_index = col_map.get(normalized_header)
        if col_index:
            updates.append((row_index, col_index, chosen_value))

    if not updates:
        raise RuntimeError("No matching payload fields found for target sheet headers.")

    ws.update_cells(
        [gspread.Cell(r, c, v) for r, c, v in updates],
        value_input_option="RAW",
    )

    invalidate_all_caches()

    return {
        "ok": True,
        "action": action,
        "written_row": row_index,
        "worksheet_name": sheet_name,
    }


def admin_upsert_to_sheet(payload: Dict[str, Any]) -> Dict[str, Any]:
    target_sheet = (payload.get("target_sheet") or "legacy").strip()

    if target_sheet == "legacy":
        return admin_upsert_generic_sheet(Config.WORKSHEET_NAME, payload)

    sheet_alias_map = {
        "BaseSymbols": Config.SHEET_BASE_SYMBOLS,
        "BehaviorRules": Config.SHEET_BEHAVIOR_RULES,
        "SizeStateRules": Config.SHEET_SIZE_STATE_RULES,
        "LocationRules": Config.SHEET_LOCATION_RULES,
        "RelationshipRules": Config.SHEET_RELATIONSHIP_RULES,
        "OverrideRules": Config.SHEET_OVERRIDE_RULES,
        "OutputTemplates": Config.SHEET_OUTPUT_TEMPLATES,
    }

    real_sheet_name = sheet_alias_map.get(target_sheet, target_sheet)
    return admin_upsert_generic_sheet(real_sheet_name, payload)
