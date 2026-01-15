# app.py — Jamaican True Stories Dream Interpreter (Phase 2.5)
# Reads from Google Sheets:
#   - Doctrine dictionary: WORKSHEET_NAME (default "Sheet1") — 5 columns:
#       input | Spiritual Meaning | Physical Effects | Action | keywords
#   - Seals table: SEALS_WORKSHEET_NAME (default "SEALS") — recommended columns:
#       seal_name | trigger_keywords | dream_type | risk_level | seal_summary
#
# Fixes / Features:
# - NO oauth2client (deprecated) — uses google-auth service account credentials
# - Accepts BOTH payload keys: { "text": "..." } OR { "dream": "..." }
# - CORS configured for Shopify domain + optional Hoppscotch
# - Sheet caching (fast + cheaper)
# - Better Dream Receipt (code_name + top-5 + caption-ready share phrase)
# - SEAL selection based on triggers and matched symbols
#
# Required ENV Vars (Render):
#   SPREADSHEET_ID=your_google_sheet_id
#
# Optional ENV Vars:
#   WORKSHEET_NAME=Sheet1
#   SEALS_WORKSHEET_NAME=SEALS
#   GOOGLE_SERVICE_ACCOUNT_JSON={"type":"service_account",...}  (full JSON as a single line)
#   GOOGLE_SERVICE_ACCOUNT_FILE=/etc/secrets/credentials.json  (Render secret file path)
#   ALLOWED_ORIGINS=https://jamaicantruestories.com,https://www.jamaicantruestories.com,https://hoppscotch.io
#   SHEET_CACHE_TTL=120
#   MIN_MATCH_SCORE=70
#   TOP_K=5
#   FREE_LIMIT=3
#
# requirements.txt (minimum):
# flask, flask-cors, gunicorn, gspread, google-auth, rapidfuzz

import os
import json
import time
import re
import hashlib
from typing import Dict, List, Tuple, Optional, Any

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
    "https://hoppscotch.io",  # helpful for browser testing
]

allowed_origins_env = os.getenv("ALLOWED_ORIGINS", "")
allowed_origins = [o.strip() for o in allowed_origins_env.split(",") if o.strip()] or DEFAULT_ALLOWED

CORS(app, resources={r"/*": {"origins": allowed_origins}}, supports_credentials=False)

SPREADSHEET_ID = os.getenv("SPREADSHEET_ID", "").strip()
WORKSHEET_NAME = os.getenv("WORKSHEET_NAME", "Sheet1").strip()
SEALS_WORKSHEET_NAME = os.getenv("SEALS_WORKSHEET_NAME", "SEALS").strip()

# Google scopes for Sheets
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets.readonly",
    "https://www.googleapis.com/auth/drive.readonly",
]

# ----------------------------
# Cache + basic free-usage tracking (in-memory)
# ----------------------------
_CACHE = {
    "loaded_at": 0.0,
    "rows": [],   # Sheet1 doctrine rows
    "seals": [],  # SEALS rows
}
CACHE_TTL_SECONDS = int(os.getenv("SHEET_CACHE_TTL", "120"))
MIN_MATCH_SCORE = int(os.getenv("MIN_MATCH_SCORE", "70"))
TOP_K = int(os.getenv("TOP_K", "5"))
FREE_LIMIT = int(os.getenv("FREE_LIMIT", "3"))

# NOTE: This resets when Render restarts. This is okay as a Phase 2 placeholder.
_USAGE: Dict[str, int] = {}  # user_id -> used_count


def _normalize_header(h: str) -> str:
    h = (h or "").strip().lower()
    h = re.sub(r"\s+", " ", h)
    return h


def _norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def _split_keywords(s: str) -> List[str]:
    if not s:
        return []
    parts = re.split(r"[,\|;]+", s)
    cleaned = []
    for p in parts:
        p = p.strip().lower()
        if p:
            cleaned.append(p)
    return cleaned


def _get_credentials() -> Credentials:
    """
    Priority:
    1) GOOGLE_SERVICE_ACCOUNT_JSON env var (recommended on Render)
    2) GOOGLE_SERVICE_ACCOUNT_FILE path (Render secret file)
    3) credentials.json in repo (not recommended for production)
    """
    raw = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON", "").strip()
    if raw:
        info = json.loads(raw)
        return Credentials.from_service_account_info(info, scopes=SCOPES)

    file_path = os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE", "").strip()
    if file_path:
        if not os.path.exists(file_path):
            raise RuntimeError(f"GOOGLE_SERVICE_ACCOUNT_FILE path not found: {file_path}")
        return Credentials.from_service_account_file(file_path, scopes=SCOPES)

    # fallback to local file
    cred_path = "credentials.json"
    if not os.path.exists(cred_path):
        raise RuntimeError(
            "Missing Google credentials. Set GOOGLE_SERVICE_ACCOUNT_JSON in Render "
            "or use a Render Secret File + GOOGLE_SERVICE_ACCOUNT_FILE."
        )
    return Credentials.from_service_account_file(cred_path, scopes=SCOPES)


def _load_sheets(force: bool = False) -> Tuple[List[Dict], List[Dict]]:
    """
    Loads:
      - doctrine rows from WORKSHEET_NAME
      - seals rows from SEALS_WORKSHEET_NAME (optional; empty if missing)
    """
    now = time.time()
    if not force and _CACHE["rows"] and (now - _CACHE["loaded_at"] < CACHE_TTL_SECONDS):
        return _CACHE["rows"], _CACHE["seals"]

    if not SPREADSHEET_ID:
        raise RuntimeError("SPREADSHEET_ID env var is not set.")

    creds = _get_credentials()
    gc = gspread.authorize(creds)
    sh = gc.open_by_key(SPREADSHEET_ID)

    # ---- Sheet1 doctrine
    ws = sh.worksheet(WORKSHEET_NAME)
    values = ws.get_all_values()

    doctrine_rows: List[Dict[str, str]] = []
    if values and len(values) >= 2:
        headers = [_normalize_header(h) for h in values[0]]
        for r in values[1:]:
            if len(r) < len(headers):
                r = r + [""] * (len(headers) - len(r))
            item = {headers[i]: (r[i] or "").strip() for i in range(len(headers))}
            doctrine_rows.append(item)

    # ---- SEALS (optional)
    seals_rows: List[Dict[str, str]] = []
    try:
        seals_ws = sh.worksheet(SEALS_WORKSHEET_NAME)
        seals_vals = seals_ws.get_all_values()
        if seals_vals and len(seals_vals) >= 2:
            s_headers = [_normalize_header(h) for h in seals_vals[0]]
            for r in seals_vals[1:]:
                if len(r) < len(s_headers):
                    r = r + [""] * (len(s_headers) - len(r))
                item = {s_headers[i]: (r[i] or "").strip() for i in range(len(s_headers))}
                seals_rows.append(item)
    except Exception:
        seals_rows = []

    _CACHE["rows"] = doctrine_rows
    _CACHE["seals"] = seals_rows
    _CACHE["loaded_at"] = now

    return doctrine_rows, seals_rows


def _match_symbols(dream: str, rows: List[Dict], top_k: int = TOP_K) -> List[Tuple[Dict, int]]:
    """
    Returns list of (row_dict, score) sorted by score desc.
    Matching strategy:
      - direct substring hits on input / keywords
      - fuzzy partial ratio fallback for input / keywords
    """
    text = _norm_text(dream)
    scored: List[Tuple[Dict, int]] = []

    for row in rows:
        symbol = (row.get("input") or row.get("symbol") or "").strip()
        if not symbol:
            continue

        symbol_l = symbol.lower()
        keywords = _split_keywords(row.get("keywords", ""))

        score = 0

        # direct symbol hit
        if symbol_l and symbol_l in text:
            score = max(score, 95)

        # keyword hit
        for kw in keywords:
            if kw and kw in text:
                score = max(score, 92)

        # fuzzy fallback
        if score < 90:
            score = max(score, fuzz.partial_ratio(symbol_l, text))
            for kw in keywords[:6]:
                score = max(score, fuzz.partial_ratio(kw, text))

        if score >= MIN_MATCH_SCORE:
            scored.append((row, score))

    scored.sort(key=lambda x: x[1], reverse=True)

    # de-dupe by input symbol
    seen = set()
    out: List[Tuple[Dict, int]] = []
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
    Combines matched doctrine rows into one 3-part interpretation.
    """
    spiritual_parts = []
    physical_parts = []
    action_parts = []

    for row, _sc in matches:
        symbol = (row.get("input") or "").strip()
        sm = (row.get("spiritual meaning") or row.get("spiritual_meaning") or "").strip()
        pe = (row.get("physical effects") or row.get("physical_effects") or "").strip()
        ac = (row.get("action") or "").strip()

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


def _make_code_name(dream: str) -> str:
    h = hashlib.sha1(_norm_text(dream).encode("utf-8")).hexdigest()[:8].upper()
    return f"JTS-{h}"


def _pick_seal(matches: List[Tuple[Dict, int]], seals_rows: List[Dict]) -> Optional[Dict[str, Any]]:
    """
    Select a seal based on trigger_keywords matching the matched symbols.
    SEALS recommended columns:
      - seal_name
      - trigger_keywords (comma separated)
      - dream_type
      - risk_level
      - seal_summary
    """
    if not seals_rows:
        return None

    matched_symbols_text = " ".join([(m[0].get("input") or "").strip().lower() for m in matches])

    best = None
    best_score = 0

    for s in seals_rows:
        triggers = s.get("trigger_keywords") or s.get("triggers") or s.get("keywords") or ""
        trigger_list = _split_keywords(triggers)
        score = 0

        for trig in trigger_list:
            if trig and trig in matched_symbols_text:
                score += 1

        if score > best_score:
            best_score = score
            best = s

    if not best or best_score == 0:
        return None

    return {
        "seal_name": best.get("seal_name", "").strip() or "Seal",
        "dream_type": best.get("dream_type", "").strip(),
        "risk_level": best.get("risk_level", "").strip(),
        "seal_summary": best.get("seal_summary", "").strip(),
    }


def _usage_status(user_id: str) -> Dict[str, int]:
    used = _USAGE.get(user_id, 0)
    left = max(FREE_LIMIT - used, 0)
    return {"used": used, "free_uses_left": left}


# ----------------------------
# Routes
# ----------------------------
@app.get("/health")
def health():
    # Make health actually check sheet access lightly (optional but helpful)
    try:
        _load_sheets(force=False)
        return jsonify({"ok": True, "service": "dream-interpreter", "sheet": WORKSHEET_NAME, "seals": SEALS_WORKSHEET_NAME})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.post("/interpret")
def interpret():
    data = request.get_json(silent=True) or {}

    # ✅ Accept BOTH keys so frontend/backend can’t mismatch again.
    dream = (data.get("dream") or data.get("text") or "").strip()
    user_id = str(data.get("user_id") or "anon").strip()

    if not dream:
        return jsonify({"error": "Missing 'dream' or 'text'"}), 400

    # Free limit (Phase 2 placeholder)
    status = _usage_status(user_id)
    if status["free_uses_left"] <= 0:
        return jsonify({
            "paywall": True,
            "access": "free",
            "is_paid": False,
            "free_uses_left": 0,
            "message": "Free limit reached. Subscribe to unlock full interpretations.",
        })

    try:
        rows, seals_rows = _load_sheets()
    except Exception as e:
        return jsonify({"error": "Sheet load failed", "details": str(e)}), 500

    matches = _match_symbols(dream, rows, top_k=TOP_K)

    if not matches:
        return jsonify({
            "access": "free",
            "is_paid": False,
            "free_uses_left": status["free_uses_left"],
            "interpretation": {
                "spiritual_meaning": "No matching symbols were found yet for this dream in Sheet1.",
                "effects_in_physical_realm": "As you add more symbols/keywords, matches will improve.",
                "what_to_do": "Try adding more details (who/where/what happened) and include key objects/animals/people."
            },
            "receipt": {
                "code_name": _make_code_name(dream),
                "top_symbols": [],
                "share_phrase": "I decoded my dream on Jamaican True Stories.",
                "seal_name": "",
                "seal_summary": "",
            }
        })

    interpretation = _combine_fields(matches)

    # Receipt
    top_symbols = [(row.get("input") or "").strip() for row, _ in matches if (row.get("input") or "").strip()]
    code_name = _make_code_name(dream)

    # SEAL selection
    seal = _pick_seal(matches, seals_rows)

    share_phrase = ""
    if top_symbols:
        share_phrase = (
            "My dream had symbols like: "
            + ", ".join(top_symbols[:5])
            + ". I decoded it on Jamaican True Stories."
        )
    else:
        share_phrase = "I decoded my dream on Jamaican True Stories."

    # increment usage only after successful match
    _USAGE[user_id] = _USAGE.get(user_id, 0) + 1
    status = _usage_status(user_id)

    response = {
        "access": "free",
        "is_paid": False,
        "free_uses_left": status["free_uses_left"],
        "interpretation": interpretation,
        "receipt": {
            "code_name": code_name,
            "top_symbols": top_symbols,
            "share_phrase": share_phrase,
            "seal_name": seal["seal_name"] if seal else "",
            "seal_summary": seal.get("seal_summary", "") if seal else "",
        }
    }

    if seal:
        response["seal"] = seal

    return jsonify(response)


# ----------------------------
# Local dev entry (Render uses gunicorn)
# ----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=True)
