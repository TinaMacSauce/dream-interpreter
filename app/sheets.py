import json
import time
import os
from typing import Dict, List, Tuple

import gspread
from google.oauth2.service_account import Credentials

from app.config import Config
from app.utils import normalize_header
from app.cache import LEGACY_CACHE, DOCTRINE_CACHE


# ============================================================
# Credentials
# ============================================================
def get_credentials() -> Credentials:
    raw = Config.GOOGLE_SERVICE_ACCOUNT_JSON

    if raw:
        raw_fixed = raw.replace("\\n", "\n")
        info = json.loads(raw_fixed)
        return Credentials.from_service_account_info(info, scopes=Config.GOOGLE_SCOPES)

    if not os.path.exists(Config.GOOGLE_SERVICE_ACCOUNT_FILE):
        raise RuntimeError("Missing Google credentials.")

    return Credentials.from_service_account_file(
        Config.GOOGLE_SERVICE_ACCOUNT_FILE,
        scopes=Config.GOOGLE_SCOPES,
    )


def get_spreadsheet():
    creds = get_credentials()
    gc = gspread.authorize(creds)
    return gc.open_by_key(Config.SPREADSHEET_ID)


# ============================================================
# Worksheet handling
# ============================================================
def get_or_create_worksheet(sheet_name: str):
    sh = get_spreadsheet()

    try:
        ws = sh.worksheet(sheet_name)
    except Exception:
        ws = sh.add_worksheet(title=sheet_name, rows=2000, cols=30)

    ensure_expected_headers(ws, sheet_name)
    return ws


def ensure_expected_headers(ws, sheet_name: str):
    expected = Config.EXPECTED_HEADERS.get(sheet_name)
    if not expected:
        return

    current = ws.row_values(1)
    current_norm = [normalize_header(x) for x in current]
    expected_norm = [normalize_header(x) for x in expected]

    from app.utils import col_to_a1
    end_col = col_to_a1(len(expected))

    if not current or current_norm[:len(expected_norm)] != expected_norm:
        ws.update(f"A1:{end_col}1", [expected])

# ============================================================
# Sheet → rows
# ============================================================
def worksheet_to_rows(ws) -> Tuple[List[str], List[Dict[str, str]]]:
    values = ws.get_all_values()

    if not values:
        return [], []

    raw_headers = values[0]
    headers = []
    seen = {}

    for h in raw_headers:
        nh = normalize_header(h) or "col"
        count = seen.get(nh, 0)
        seen[nh] = count + 1
        headers.append(nh if count == 0 else f"{nh}__{count + 1}")

    rows: List[Dict[str, str]] = []

    for r in values[1:]:
        if len(r) < len(headers):
            r = r + [""] * (len(headers) - len(r))

        rows.append({
            headers[i]: (r[i] or "").strip()
            for i in range(len(headers))
        })

    return headers, rows


# ============================================================
# Legacy sheet
# ============================================================
def load_legacy_rows(force: bool = False) -> List[Dict]:
    now = time.time()

    if (
        not force
        and LEGACY_CACHE["rows"]
        and (now - LEGACY_CACHE["loaded_at"] < Config.CACHE_TTL_SECONDS)
    ):
        return LEGACY_CACHE["rows"]

    ws = get_or_create_worksheet(Config.WORKSHEET_NAME)
    headers, rows = worksheet_to_rows(ws)

    LEGACY_CACHE["rows"] = rows
    LEGACY_CACHE["headers"] = headers
    LEGACY_CACHE["loaded_at"] = now

    return rows


# ============================================================
# Doctrine sheets
# ============================================================
def load_doctrine_sheets(force: bool = False) -> Dict[str, List[Dict]]:
    now = time.time()

    if (
        not force
        and DOCTRINE_CACHE["sheets"]
        and (now - DOCTRINE_CACHE["loaded_at"] < Config.CACHE_TTL_SECONDS)
    ):
        return DOCTRINE_CACHE["sheets"]

    sheets_data: Dict[str, List[Dict]] = {}
    headers_map: Dict[str, List[str]] = {}

    for name in Config.DOCTRINE_SHEET_NAMES:
        try:
            ws = get_or_create_worksheet(name)
            headers, rows = worksheet_to_rows(ws)

            sheets_data[name] = rows
            headers_map[name] = headers
        except Exception:
            sheets_data[name] = []
            headers_map[name] = []

    DOCTRINE_CACHE["sheets"] = sheets_data
    DOCTRINE_CACHE["headers"] = headers_map
    DOCTRINE_CACHE["loaded_at"] = now

    return sheets_data


def doctrine_available() -> bool:
    try:
        sheets = load_doctrine_sheets()
        return len(sheets.get(Config.SHEET_BASE_SYMBOLS, [])) > 0
    except Exception:
        return False


# ============================================================
# Journal sheet
# ============================================================
def get_journal_worksheet():
    return get_or_create_worksheet(Config.SHEET_DREAM_JOURNAL)
