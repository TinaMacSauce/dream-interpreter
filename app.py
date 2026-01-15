# app.py — Jamaican True Stories Dream Interpreter (Phase 2.5)
# Reads from Google Sheets tab: "Sheet1" (5-column doctrine sheet)
#
# Fixes:
# - NO oauth2client — uses google-auth service account credentials
# - Accepts BOTH payload keys: { "text": "..." } OR { "dream": "..." }
# - Adds /track endpoint (your Shopify UI is calling it; missing route causes OPTIONS 404)
# - Stronger "old-style" symbol extraction: scan dream text -> pick top 3 best matches
# - Returns: interpretation + receipt + seal in a consistent shape
#
# Required ENV Vars (Render):
#   SPREADSHEET_ID=your_google_sheet_id
#
# Optional:
#   WORKSHEET_NAME=Sheet1
#   GOOGLE_SERVICE_ACCOUNT_JSON={"type":"service_account",...}   (single-line JSON)
#   GOOGLE_SERVICE_ACCOUNT_FILE=credentials.json                (fallback)
#   ALLOWED_ORIGINS=https://jamaicantruestories.com,https://www.jamaicantruestories.com
#   SHEET_CACHE_TTL=120
#   DEBUG_ENDPOINT=0  (set to 1 to enable /debug)
#
# requirements.txt should include:
# flask, flask-cors, gunicorn, gspread, google-auth, rapidfuzz, python-dotenv

import os
import json
import time
import re
import secrets
from typing import Dict, List, Tuple, Any

from flask import Flask, request, jsonify, make_response
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

# Important: allow Content-Type + handle OPTIONS preflight
app.config["CORS_HEADERS"] = "Content-Type"

CORS(
    app,
    resources={r"/*": {"origins": allowed_origins}},
    supports_credentials=True,
)

SPREADSHEET_ID = os.getenv("SPREADSHEET_ID", "").strip()
WORKSHEET_NAME = os.getenv("WORKSHEET_NAME", "Sheet1").strip()

DEBUG_ENDPOINT = os.getenv("DEBUG_ENDPOINT", "0").strip() == "1"

# Google scopes for Sheets
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets.readonly",
    "https://www.googleapis.com/auth/drive.readonly",
]

# ----------------------------
# Simple cache (prevents reading Google Sheet on every request)
# ----------------------------
_CACHE: Dict[str, Any] = {
    "loaded_at": 0.0,
    "rows": [],       # list[dict]
    "headers": [],    # list[str]
}
CACHE_TTL_SECONDS = int(os.getenv("SHEET_CACHE_TTL", "120"))  # 2 minutes default

# ----------------------------
# Helpers
# ----------------------------
def _normalize_header(h: str) -> str:
    h = (h or "").strip().lower()
    h = re.sub(r"\s+", " ", h)
    return h

def _safe_strip(x: Any) -> str:
    return (x or "").strip() if isinstance(x, str) else str(x or "").strip()

def _get_credentials() -> Credentials:
    """
    Priority:
    1) GOOGLE_SERVICE_ACCOUNT_JSON env var (recommended on Render)
    2) credentials.json file in repo
    """
    raw = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON", "").strip()
    if raw:
        # Render env vars sometimes contain literal "\n" sequences; convert if needed
        raw_fixed = raw.replace("\\n", "\n")
        info = json.loads(raw_fixed)
        return Credentials.from_service_account_info(info, scopes=SCOPES)

    cred_path = os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE", "credentials.json")
    if not os.path.exists(cred_path):
        raise RuntimeError(
            "Missing Google credentials. Set GOOGLE_SERVICE_ACCOUNT_JSON in Render "
            "or add credentials.json to the repo."
        )
    return Credentials.from_service_account_file(cred_path, scopes=SCOPES)

def _load_sheet_rows(force: bool = False) -> List[Dict]:
    """
    Loads Sheet1 rows as dicts with keys like:
      input, spiritual meaning, physical effects, action, keywords
    """
    now = time.time()
    if (not force) and _CACHE["rows"] and (now - _CACHE["loaded_at"] < CACHE_TTL_SECONDS):
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
        _CACHE["headers"] = []
        _CACHE["loaded_at"] = now
        return []

    headers = [_normalize_header(h) for h in values[0]]
    rows: List[Dict] = []

    for r in values[1:]:
        if len(r) < len(headers):
            r = r + [""] * (len(headers) - len(r))
        item = {headers[i]: _safe_strip(r[i]) for i in range(len(headers))}
        rows.append(item)

    _CACHE["rows"] = rows
    _CACHE["headers"] = headers
    _CACHE["loaded_at"] = now
    return rows

def _split_keywords(s: str) -> List[str]:
    if not s:
        return []
    parts = re.split(r"[,\|;]+", s)
    out: List[str] = []
    for p in parts:
        p = p.strip().lower()
        if p:
            out.append(p)
    return out

def _word_boundary_regex(phrase: str) -> re.Pattern:
    """
    Try to match phrase as a "clean" substring with loose boundaries.
    Example: "running" shouldn't match "prunning" but should match "running."
    """
    phrase = re.escape(phrase.strip().lower())
    return re.compile(rf"(?<![a-zA-Z]){phrase}(?![a-zA-Z])")

def _score_row(dream_text_l: str, row: Dict) -> int:
    """
    More 'old-style' scoring:
    1) Exact phrase hit on input (high)
    2) Exact keyword hit (high)
    3) Fuzzy fallback (medium)
    """
    symbol = (row.get("input") or row.get("symbol") or "").strip()
    if not symbol:
        return 0

    symbol_l = symbol.lower()
    keywords = _split_keywords(row.get("keywords", ""))

    score = 0

    # Exact-ish word boundary match for input
    if symbol_l and _word_boundary_regex(symbol_l).search(dream_text_l):
        score = max(score, 100)

    # Exact-ish match for any keyword
    for kw in keywords:
        if kw and _word_boundary_regex(kw).search(dream_text_l):
            score = max(score, 96)

    # Fuzzy fallback (typos)
    if score < 90:
        score = max(score, fuzz.partial_ratio(symbol_l, dream_text_l))
        for kw in keywords[:6]:
            score = max(score, fuzz.partial_ratio(kw, dream_text_l))

    return int(score)

def _match_symbols(dream: str, rows: List[Dict], top_k: int = 3) -> List[Tuple[Dict, int]]:
    """
    Returns top_k rows by score, de-duped by input symbol.
    Uses higher threshold so we don't return random long phrases.
    """
    text_l = (dream or "").lower().strip()
    if not text_l:
        return []

    scored: List[Tuple[Dict, int]] = []
    for row in rows:
        sc = _score_row(text_l, row)
        if sc >= 80:
            scored.append((row, sc))

    scored.sort(key=lambda x: x[1], reverse=True)

    seen = set()
    out: List[Tuple[Dict, int]] = []
    for row, sc in scored:
        sym = (row.get("input") or row.get("symbol") or "").strip().lower()
        if not sym or sym in seen:
            continue
        seen.add(sym)
        out.append((row, sc))
        if len(out) >= top_k:
            break

    return out

def _combine_fields(matches: List[Tuple[Dict, int]]) -> Dict[str, str]:
    spiritual_parts: List[str] = []
    physical_parts: List[str] = []
    action_parts: List[str] = []

    for row, sc in matches:
        symbol = (row.get("input") or row.get("symbol") or "").strip()

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

def _make_receipt_id() -> str:
    # Short, readable receipt ID like: JTS-AB12CD3E
    token = secrets.token_hex(4).upper()
    return f"JTS-{token}"

def _compute_seal(matches: List[Tuple[Dict, int]]) -> Dict[str, str]:
    """
    Simple deterministic seal (no OpenAI):
    - Type: Confirmed / Processing / Unclear
    - Risk: Low / Medium / High
    """
    if not matches:
        return {"status": "Delayed", "type": "Unclear", "risk": "High"}

    scores = [sc for _, sc in matches]
    avg = sum(scores) / max(len(scores), 1)

    if avg >= 95:
        return {"status": "Live", "type": "Confirmed", "risk": "Low"}
    if avg >= 88:
        return {"status": "Delayed", "type": "Processing", "risk": "Medium"}
    return {"status": "Delayed", "type": "Processing", "risk": "High"}

def _preflight_ok() -> Any:
    # Useful if some clients hit OPTIONS explicitly.
    resp = make_response("", 204)
    return resp

# ----------------------------
# Routes
# ----------------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "ok": True,
        "service": "dream-interpreter",
        "sheet": WORKSHEET_NAME,
        "has_spreadsheet_id": bool(SPREADSHEET_ID),
        "allowed_origins": allowed_origins,
    })

@app.route("/debug", methods=["GET"])
def debug():
    if not DEBUG_ENDPOINT:
        return jsonify({"error": "Not enabled"}), 404
    try:
        rows = _load_sheet_rows(force=True)
        return jsonify({
            "ok": True,
            "worksheet": WORKSHEET_NAME,
            "row_count": len(rows),
            "headers": _CACHE.get("headers", []),
            # Do NOT expose row data (privacy)
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/interpret", methods=["POST", "OPTIONS"])
def interpret():
    if request.method == "OPTIONS":
        return _preflight_ok()

    data = request.get_json(silent=True) or {}

    # Accept BOTH keys so frontend/backend can’t mismatch again.
    dream = (data.get("dream") or data.get("text") or "").strip()

    if not dream:
        return jsonify({"error": "Missing 'text'"}), 400

    try:
        rows = _load_sheet_rows()
    except Exception as e:
        return jsonify({"error": "Sheet load failed", "details": str(e)}), 500

    matches = _match_symbols(dream, rows, top_k=3)

    if not matches:
        receipt_id = _make_receipt_id()
        seal = _compute_seal(matches)

        return jsonify({
            "access": "free",
            "is_paid": False,
            "free_uses_left": 3,  # placeholder until wired to storage
            "seal": seal,
            "receipt": {
                "id": receipt_id,
                "top_symbols": [],
                "share_phrase": "I decoded my dream on Jamaican True Stories.",
            },
            "interpretation": {
                "spiritual_meaning": "No matching symbols were found yet for this dream in Sheet1.",
                "effects_in_physical_realm": "As you add more symbols/keywords, matches will improve.",
                "what_to_do": "Try adding more key objects/people/places and keep symbols simple (e.g., running, teeth, snake)."
            }
        })

    interpretation = _combine_fields(matches)
    top_symbols = [(row.get("input") or row.get("symbol") or "").strip() for row, _ in matches]
    top_symbols = [s for s in top_symbols if s]

    receipt_id = _make_receipt_id()
    seal = _compute_seal(matches)

    share_list = ", ".join(top_symbols[:3])
    share_phrase = f"My dream had symbols like: {share_list}. I decoded it on Jamaican True Stories."

    return jsonify({
        "access": "free",
        "is_paid": False,
        "free_uses_left": 3,  # placeholder until wired to storage
        "seal": seal,
        "interpretation": interpretation,
        "receipt": {
            "id": receipt_id,
            "top_symbols": top_symbols,
            "share_phrase": share_phrase
        }
    })

@app.route("/track", methods=["POST", "OPTIONS"])
def track():
    """
    Your Shopify page is hitting /track (Render logs show it).
    Without this, browsers send OPTIONS preflight and you get 404s,
    which can break your UI flow.
    """
    if request.method == "OPTIONS":
        return _preflight_ok()

    payload = request.get_json(silent=True) or {}

    # Minimal stub response so your frontend doesn’t break.
    # Later you can store usage in Airtable/DB/etc.
    return jsonify({
        "ok": True,
        "event": payload.get("event", "unknown"),
        "free_uses_left": payload.get("free_uses_left", 3),
    })

# ----------------------------
# Local dev entry (Render uses gunicorn)
# ----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=True)
