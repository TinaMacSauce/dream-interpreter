from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Tuple

import gspread
from google.oauth2.service_account import Credentials

from app.cache import (
    DOCTRINE_CACHE,
    LEGACY_CACHE,
)
from app.config import Config
from app.utils import (
    col_to_a1,
    normalize_header,
)


# ============================================================
# CLIENT CACHE
# ============================================================

_GSPREAD_CLIENT = None
_SPREADSHEET = None


# ============================================================
# CREDENTIALS
# ============================================================

def get_credentials() -> Credentials:
    raw = (
        Config.GOOGLE_SERVICE_ACCOUNT_JSON
        or ""
    ).strip()

    if raw:
        try:
            raw_fixed = raw.replace(
                "\\n",
                "\n",
            )

            info = json.loads(raw_fixed)

            return Credentials.from_service_account_info(
                info,
                scopes=Config.GOOGLE_SCOPES,
            )

        except Exception as e:
            raise RuntimeError(
                f"Invalid GOOGLE_SERVICE_ACCOUNT_JSON: {str(e)}"
            )

    creds_path = (
        Config.GOOGLE_SERVICE_ACCOUNT_FILE
        or ""
    ).strip()

    if not creds_path:
        raise RuntimeError(
            "Google credentials path missing."
        )

    if not os.path.exists(creds_path):
        raise RuntimeError(
            f"Missing Google credentials file: {creds_path}"
        )

    try:
        return Credentials.from_service_account_file(
            creds_path,
            scopes=Config.GOOGLE_SCOPES,
        )

    except Exception as e:
        raise RuntimeError(
            f"Failed loading credentials file: {str(e)}"
        )


def get_gspread_client():
    global _GSPREAD_CLIENT

    if _GSPREAD_CLIENT is not None:
        return _GSPREAD_CLIENT

    creds = get_credentials()

    _GSPREAD_CLIENT = gspread.authorize(
        creds
    )

    return _GSPREAD_CLIENT


def get_spreadsheet():
    global _SPREADSHEET

    if _SPREADSHEET is not None:
        return _SPREADSHEET

    client = get_gspread_client()

    try:
        _SPREADSHEET = client.open_by_key(
            Config.SPREADSHEET_ID
        )

    except Exception as e:
        raise RuntimeError(
            f"Failed opening spreadsheet: {str(e)}"
        )

    return _SPREADSHEET


# ============================================================
# WORKSHEETS
# ============================================================

def get_or_create_worksheet(
    sheet_name: str,
    rows: int = 3000,
    cols: int = 40,
):
    sh = get_spreadsheet()

    try:
        ws = sh.worksheet(sheet_name)

    except Exception:
        ws = sh.add_worksheet(
            title=sheet_name,
            rows=rows,
            cols=cols,
        )

    ensure_expected_headers(
        ws,
        sheet_name,
    )

    return ws


def ensure_expected_headers(
    ws,
    sheet_name: str,
) -> None:
    expected = (
        Config.EXPECTED_HEADERS.get(
            sheet_name
        )
        or []
    )

    if not expected:
        return

    try:
        current = ws.row_values(1)

    except Exception:
        current = []

    current_norm = [
        normalize_header(x)
        for x in current
    ]

    expected_norm = [
        normalize_header(x)
        for x in expected
    ]

    needs_update = (
        not current
        or current_norm[: len(expected_norm)]
        != expected_norm
    )

    if not needs_update:
        return

    end_col = col_to_a1(len(expected))

    ws.update(
        f"A1:{end_col}1",
        [expected],
        value_input_option="RAW",
    )


# ============================================================
# ROW NORMALIZATION
# ============================================================

def worksheet_to_rows(
    ws,
) -> Tuple[List[str], List[Dict[str, str]]]:
    try:
        values = ws.get_all_values()

    except Exception as e:
        raise RuntimeError(
            f"Failed reading worksheet values: {str(e)}"
        )

    if not values:
        return [], []

    raw_headers = values[0]

    headers: List[str] = []

    seen: Dict[str, int] = {}

    for h in raw_headers:
        nh = (
            normalize_header(h)
            or "col"
        )

        count = seen.get(nh, 0)

        seen[nh] = count + 1

        headers.append(
            nh
            if count == 0
            else f"{nh}__{count + 1}"
        )

    rows: List[Dict[str, str]] = []

    for r in values[1:]:
        if len(r) < len(headers):
            r = r + (
                [""] * (
                    len(headers) - len(r)
                )
            )

        row = {
            headers[i]: (
                r[i] or ""
            ).strip()
            for i in range(len(headers))
        }

        if not any(row.values()):
            continue

        rows.append(row)

    return headers, rows


# ============================================================
# LEGACY CACHE
# ============================================================

def load_legacy_rows(
    force: bool = False,
) -> List[Dict]:
    now = time.time()

    if (
        not force
        and LEGACY_CACHE["rows"]
        and (
            now
            - LEGACY_CACHE["loaded_at"]
            < Config.CACHE_TTL_SECONDS
        )
    ):
        return LEGACY_CACHE["rows"]

    ws = get_or_create_worksheet(
        Config.WORKSHEET_NAME
    )

    headers, rows = worksheet_to_rows(
        ws
    )

    LEGACY_CACHE["rows"] = rows
    LEGACY_CACHE["headers"] = headers
    LEGACY_CACHE["loaded_at"] = now

    return rows


# ============================================================
# DOCTRINE CACHE
# ============================================================

def load_doctrine_sheets(
    force: bool = False,
) -> Dict[str, List[Dict]]:
    now = time.time()

    if (
        not force
        and DOCTRINE_CACHE["sheets"]
        and (
            now
            - DOCTRINE_CACHE["loaded_at"]
            < Config.CACHE_TTL_SECONDS
        )
    ):
        return DOCTRINE_CACHE["sheets"]

    sheets_data: Dict[
        str,
        List[Dict]
    ] = {}

    headers_map: Dict[
        str,
        List[str]
    ] = {}

    errors: Dict[str, str] = {}

    for name in Config.DOCTRINE_SHEET_NAMES:
        try:
            ws = get_or_create_worksheet(
                name
            )

            headers, rows = worksheet_to_rows(
                ws
            )

            sheets_data[name] = rows
            headers_map[name] = headers

        except Exception as e:
            sheets_data[name] = []
            headers_map[name] = []
            errors[name] = str(e)

    DOCTRINE_CACHE["sheets"] = (
        sheets_data
    )

    DOCTRINE_CACHE["headers"] = (
        headers_map
    )

    DOCTRINE_CACHE["errors"] = (
        errors
    )

    DOCTRINE_CACHE["loaded_at"] = now

    return sheets_data


def doctrine_available() -> bool:
    try:
        sheets = load_doctrine_sheets()

        base_rows = sheets.get(
            Config.SHEET_BASE_SYMBOLS,
            [],
        )

        return len(base_rows) > 0

    except Exception:
        return False


def get_doctrine_headers() -> Dict[
    str,
    List[str],
]:
    load_doctrine_sheets()

    return (
        DOCTRINE_CACHE.get("headers")
        or {}
    )


def get_doctrine_load_errors() -> Dict[
    str,
    str,
]:
    load_doctrine_sheets()

    return (
        DOCTRINE_CACHE.get("errors")
        or {}
    )


# ============================================================
# JOURNAL
# ============================================================

def get_journal_worksheet():
    return get_or_create_worksheet(
        Config.SHEET_DREAM_JOURNAL
    )


# ============================================================
# CACHE RESET
# ============================================================

def clear_sheet_caches() -> None:
    LEGACY_CACHE["rows"] = []
    LEGACY_CACHE["headers"] = []
    LEGACY_CACHE["loaded_at"] = 0

    DOCTRINE_CACHE["sheets"] = {}
    DOCTRINE_CACHE["headers"] = {}
    DOCTRINE_CACHE["errors"] = {}
    DOCTRINE_CACHE["loaded_at"] = 0
