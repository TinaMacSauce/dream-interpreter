# app.py — Jamaican True Stories Dream Interpreter (Phase 2)
# Reads from Google Sheets tab: "Sheet1" (5-column doctrine sheet)
# Fixes:
# - NO oauth2client (deprecated) — uses google-auth service account credentials
# - Accepts BOTH payload keys: { "text": "..." } OR { "dream": "..." }
# - CORS configured for Shopify domain
#
# Required ENV Vars (Render):
#   SPREADSHEET_ID=your_google_sheet_id
# Optional:
#   WORKSHEET_NAME=Sheet1
#   GOOGLE_SERVICE_ACCOUNT_JSON={"type":"service_account",...}  (full JSON as a single line)
#   ALLOWED_ORIGINS=https://jamaicantruestories.com,https://www.jamaicantruestories.com
#
# requirements.txt should include (you already have most):
# flask, flask-cors, gunicorn, gspread, google-auth, rapidfuzz, python-dotenv

import os
import json
import time
import re
from typing import Dict, List, Tuple

from flask import Flask, request, jsonify
from flask_cors import CORS

import gspread
from google.oauth2.service_account import Credentials
from rapidfuzz import fuzz

# ----------------------------
# App setup
# ----------------------------
app = Flask(__name__)

DEFAULT_ALLOWED = [
    "https://jamaicantruestories.com",
    "https://www.jamaicantruestories.com",
    "https://interpreter.jamaicantruestories.com",
]
allowed_origins_env = os.getenv("ALLOWED_ORIGINS", "")
allowed_origins = [o.strip() for o in allowed_origins_env.split(",") if o.strip()] or DEFAULT_ALLOWED

CORS(app, resources={r"/*": {"origins": allowed_origins}})

SPREADSHEET_ID = os.getenv("SPREADSHEET_ID", "").strip()
WORKSHEET_NAME = os.getenv("WORKSHEET_NAME", "Sheet1").strip()

# Google scopes for Sheets
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets.readonly",
    "https://www.googleapis.com/auth/drive.readonly",
]

# ----------------------------
# Simple cache (prevents reading Google Sheet on every request)
# ----------------------------
_CACHE = {
    "loaded_at": 0.0,
    "rows": [],  # list[dict]
}
CACHE_TTL_SECONDS = int(os.getenv("SHEET_CACHE_TTL", "120"))  # 2 minutes default


def _normalize_header(h: str) -> str:
    h = (h or "").strip().lower()
    h = re.sub(r"\s+", " ", h)
    return h


def _get_credentials() -> Credentials:
    """
    Priority:
    1) GOOGLE_SERVICE_ACCOUNT_JSON env var (recommended on Render)
    2) credentials.json file in repo
    """
    raw = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON", "").strip()
    if raw:
        info = json.loads(raw)
        return Credentials.from_service_account_info(info, scopes=SCOPES)

    # fallback to local file
    cred_path = os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE", "credentials.json")
    if not os.path.exists(cred_path):
        raise RuntimeError(
            "Missing Google credentials. Set GOOGLE_SERVICE_ACCOUNT_JSON in Render "
            "or add credentials.json to the repo."
        )
    return Credentials.from_service_account_file(cred_path, scopes=SCOPES)


def _load_sheet_rows(force: bool = False) -> List[Dict]:
    """
    Loads Sheet1 rows as dicts with keys:
      input, spiritual meaning, physical effects, action, keywords
    """
    now = time.time()
    if not force and _CACHE["rows"] and (now - _CACHE["loaded_at"] < CACHE_TTL_SECONDS):
        return _CACHE["rows"]

    if not SPREADSHEET_ID:
        raise RuntimeError("SPREADSHEET_ID env var is not set.")

    creds = _get_credentials()
    gc = gspread.authorize(creds)

    sh = gc.open_by_key(SPREADSHEET_ID)
    ws = sh.worksheet(WORKSHEET_NAME)

    values = ws.get_all_values()
    if not values or len(values) < 2:
        _CACHE["rows"] = []
        _CACHE["loaded_at"] = now
        return []

    headers = [_normalize_header(h) for h in values[0]]
    rows = []
    for r in values[1:]:
        # pad row
        if len(r) < len(headers):
            r = r + [""] * (len(headers) - len(r))
        item = {headers[i]: (r[i] or "").strip() for i in range(len(headers))}
        rows.append(item)

    _CACHE["rows"] = rows
    _CACHE["loaded_at"] = now
    return rows


def _split_keywords(s: str) -> List[str]:
    if not s:
        return []
    # supports comma/semicolon/pipe separated keywords
    parts = re.split(r"[,\|;]+", s)
    cleaned = []
    for p in parts:
        p = p.strip().lower()
        if p:
            cleaned.append(p)
    return cleaned


def _match_symbols(dream: str, rows: List[Dict], top_k: int = 5) -> List[Tuple[Dict, int]]:
    """
    Returns list of (row_dict, score) sorted by score desc.
    Matching strategy:
      - direct substring hits on input / keywords
      - fuzzy partial ratio for input / keywords for safety
    """
    text = (dream or "").lower()
    scored = []

    for row in rows:
        symbol = (row.get("input") or row.get("symbol") or "").strip()
        if not symbol:
            continue

        symbol_l = symbol.lower()
        keywords = _split_keywords(row.get("keywords", ""))

        score = 0

        # Strong score if symbol appears as substring
        if symbol_l and symbol_l in text:
            score = max(score, 95)

        # Keyword substring hits
        for kw in keywords:
            if kw and kw in text:
                score = max(score, 92)

        # Fuzzy fallback (helps typos like "runing" / "teath")
        if score < 90:
            score = max(score, fuzz.partial_ratio(symbol_l, text))
            for kw in keywords[:6]:  # cap to avoid heavy compute
                score = max(score, fuzz.partial_ratio(kw, text))

        if score >= 70:  # threshold
            scored.append((row, score))

    scored.sort(key=lambda x: x[1], reverse=True)

    # de-dupe by input symbol
    seen = set()
    out = []
    for row, sc in scored:
        sym = (row.get("input") or "").strip().lower()
        if sym in seen:
            continue
        seen.add(sym)
        out.append((row, sc))
        if len(out) >= top_k:
            break

    return out


def _combine_fields(matches: List[Tuple[Dict, int]]) -> Dict[str, str]:
    """
    Combines matched rows into one 3-part interpretation.
    """
    spiritual_parts = []
    physical_parts = []
    action_parts = []

    for row, sc in matches:
        symbol = (row.get("input") or "").strip()
        sm = (row.get("spiritual meaning") or row.get("spiritual_meaning") or "").strip()
        pe = (row.get("physical effects") or row.get("physical_effects") or "").strip()
        ac = (row.get("action") or "").strip()

        # Label by symbol so output stays readable
        if sm:
            spiritual_parts.append(f"{symbol}: {sm}")
        if pe:
            physical_parts.append(f"{symbol}: {pe}")
        if ac:
            action_parts.append(f"{symbol}: {ac}")

    return {
        "spiritual_meaning": "\n\n".join(spiritual_parts).strip(),
        "effects_in_physical_realm": "\n\n".join(physical_parts).strip(),
        "what_to_do": "\n\n".join(action_parts).strip(),
    }


# ----------------------------
# Routes
# ----------------------------
@app.get("/health")
def health():
    return jsonify({"ok": True, "service": "dream-interpreter", "sheet": WORKSHEET_NAME})


@app.post("/interpret")
def interpret():
    data = request.get_json(silent=True) or {}

    # ✅ Accept BOTH keys so frontend/backend can’t mismatch again.
    dream = (data.get("dream") or data.get("text") or "").strip()

    if not dream:
        # Keep error wording compatible with what you saw in your UI:
        return jsonify({"error": "Missing 'text'"}), 400

    try:
        rows = _load_sheet_rows()
    except Exception as e:
        return jsonify({"error": "Sheet load failed", "details": str(e)}), 500

    matches = _match_symbols(dream, rows, top_k=5)

    if not matches:
        # Safe default response when nothing matches yet
        return jsonify({
            "access": "free",
            "is_paid": False,
            "free_uses_left": 3,  # placeholder if you haven’t wired tracking yet
            "interpretation": {
                "spiritual_meaning": "No matching symbols were found yet for this dream in Sheet1.",
                "effects_in_physical_realm": "As you add more symbols/keywords, matches will improve.",
                "what_to_do": "Try adding a few more details (who/where/what happened) and include key objects/animals/people."
            }
        })

    interpretation = _combine_fields(matches)

    # Optional: include top symbols for your receipt UI
    top_symbols = [(row.get("input") or "").strip() for row, _ in matches if (row.get("input") or "").strip()]

    return jsonify({
        "access": "free",
        "is_paid": False,
        "free_uses_left": 3,  # placeholder until your server-tracking is added
        "interpretation": interpretation,
        "receipt": {
            "top_symbols": top_symbols,
            "share_phrase": f"My dream had symbols like: {', '.join(top_symbols[:3])}. I decoded it on Jamaican True Stories."
        }
    })


# ----------------------------
# Local dev entry (Render uses gunicorn)
# ----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=True)
