# app.py — Jamaican True Stories Dream Interpreter (Phase 2.7 — STRICT SYMBOL MATCH)
# Changes requested:
# - STOP interpreting symbols that are not literally present in the dream text.
# - Removed fuzzy fallback (rapidfuzz) from symbol extraction.
# - Uses strict word-boundary matching for symbols + optional keywords.
# - Keeps “old-style feel” by returning TOP 3 and preferring short canonical symbols.
#
# Notes:
# - This will prevent "airport"/"plane" unless the user literally types those words/phrases.
# - If you still want typo tolerance later, we can add an OPTIONAL "typo_mode" that only triggers
#   when the user’s text contains near-miss spellings of the same word (careful + controlled).

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

# CORS: allow Content-Type, handle browser preflight cleanly
CORS(
    app,
    resources={r"/*": {"origins": allowed_origins}},
    supports_credentials=True,
    allow_headers=["Content-Type", "Authorization"],
    methods=["GET", "POST", "OPTIONS"],
)

SPREADSHEET_ID = os.getenv("SPREADSHEET_ID", "").strip()
WORKSHEET_NAME = os.getenv("WORKSHEET_NAME", "Sheet1").strip()

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets.readonly",
    "https://www.googleapis.com/auth/drive.readonly",
]

CACHE_TTL_SECONDS = int(os.getenv("SHEET_CACHE_TTL", "120"))
_CACHE: Dict[str, Any] = {"loaded_at": 0.0, "rows": [], "headers": []}


# ----------------------------
# Helpers
# ----------------------------
def _preflight_ok():
    return make_response("", 204)


def _normalize_header(h: str) -> str:
    h = (h or "").strip().lower()
    h = re.sub(r"\s+", " ", h)
    return h


def _normalize_text(s: str) -> str:
    """
    Normalize user dream + sheet tokens into comparable text:
    - lowercase
    - replace punctuation with spaces
    - collapse whitespace
    """
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _get_credentials() -> Credentials:
    """
    Priority:
    1) GOOGLE_SERVICE_ACCOUNT_JSON env var (recommended)
    2) credentials.json file in repo (fallback)
    """
    raw = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON", "").strip()
    if raw:
        raw_fixed = raw.replace("\\n", "\n")
        info = json.loads(raw_fixed)
        return Credentials.from_service_account_info(info, scopes=SCOPES)

    cred_path = os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE", "credentials.json")
    if not os.path.exists(cred_path):
        raise RuntimeError(
            "Missing Google credentials. Set GOOGLE_SERVICE_ACCOUNT_JSON in Render "
            "or provide credentials.json via GOOGLE_SERVICE_ACCOUNT_FILE."
        )
    return Credentials.from_service_account_file(cred_path, scopes=SCOPES)


def _load_sheet_rows(force: bool = False) -> List[Dict]:
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
        item = {headers[i]: (r[i] or "").strip() for i in range(len(headers))}
        rows.append(item)

    _CACHE["rows"] = rows
    _CACHE["headers"] = headers
    _CACHE["loaded_at"] = now
    return rows


def _split_keywords(s: str) -> List[str]:
    if not s:
        return []
    parts = re.split(r"[,\|;]+", s)
    out = []
    for p in parts:
        p = _normalize_text(p)
        if p:
            out.append(p)
    return out


def _word_boundary_regex(phrase: str) -> re.Pattern:
    """
    Strict whole-word / whole-phrase match.
    Example:
    - "air" will NOT match "chair"
    - "airport" will match "airport"
    """
    phrase_n = _normalize_text(phrase)
    if not phrase_n:
        # return a pattern that never matches
        return re.compile(r"(?!x)x")
    return re.compile(rf"(?<!\w){re.escape(phrase_n)}(?!\w)")


def _symbol_length_penalty(symbol: str) -> int:
    """
    Old-style bias:
    - short symbols (1-3 words) are preferred
    - long symbols only win if they match extremely strongly
    """
    words = [w for w in _normalize_text(symbol).split() if w]
    n = len(words)
    if n <= 2:
        return 0
    if n == 3:
        return 2
    if n == 4:
        return 6
    if n == 5:
        return 12
    return 18  # 6+ words = heavy penalty


def _score_row_strict(dream_text_norm: str, row: Dict) -> int:
    """
    STRICT scoring:
    - Only scores if symbol or keyword literally appears (word-boundary match).
    - No fuzzy fallback.
    """
    symbol = (row.get("input") or row.get("symbol") or "").strip()
    if not symbol:
        return 0

    symbol_rx = _word_boundary_regex(symbol)
    keywords = _split_keywords(row.get("keywords", ""))

    score = 0

    # Exact(only) hit for symbol
    if symbol_rx.search(dream_text_norm):
        score = max(score, 100)

    # Exact(only) hit for keywords (slightly lower than symbol)
    for kw in keywords:
        if kw and _word_boundary_regex(kw).search(dream_text_norm):
            score = max(score, 96)

    # If nothing matched literally, score stays 0.
    if score <= 0:
        return 0

    # Old-style preference: penalize long "symbols"
    score -= _symbol_length_penalty(symbol)

    return int(score)


def _match_symbols_strict(dream: str, rows: List[Dict], top_k: int = 3) -> List[Tuple[Dict, int]]:
    """
    Returns only symbols that literally appear in the dream text.
    Also tries to avoid duplicates and prefers:
    - higher score
    - shorter symbol strings
    - longer phrases first when both match (handled via sorting + de-dupe)
    """
    dream_norm = _normalize_text(dream)
    if not dream_norm:
        return []

    scored: List[Tuple[Dict, int]] = []

    for row in rows:
        sc = _score_row_strict(dream_norm, row)
        if sc > 0:
            scored.append((row, sc))

    # Sort by score DESC, then shorter symbol first (old-style preference)
    def _sort_key(item: Tuple[Dict, int]):
        row, sc = item
        sym = (row.get("input") or row.get("symbol") or "").strip()
        return (-sc, len(_normalize_text(sym)))

    scored.sort(key=_sort_key)

    seen = set()
    out: List[Tuple[Dict, int]] = []
    for row, sc in scored:
        sym = (row.get("input") or row.get("symbol") or "").strip()
        sym_key = _normalize_text(sym)
        if not sym_key or sym_key in seen:
            continue
        seen.add(sym_key)
        out.append((row, sc))
        if len(out) >= top_k:
            break

    return out


def _combine_fields(matches: List[Tuple[Dict, int]]) -> Dict[str, str]:
    spiritual_parts: List[str] = []
    physical_parts: List[str] = []
    action_parts: List[str] = []

    for row, _sc in matches:
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
    return f"JTS-{secrets.token_hex(4).upper()}"


def _compute_seal(matches: List[Tuple[Dict, int]]) -> Dict[str, str]:
    if not matches:
        return {"status": "Delayed", "type": "Unclear", "risk": "High"}

    scores = [sc for _, sc in matches]
    avg = sum(scores) / max(len(scores), 1)

    if avg >= 95:
        return {"status": "Live", "type": "Confirmed", "risk": "Low"}
    if avg >= 88:
        return {"status": "Delayed", "type": "Processing", "risk": "Medium"}
    return {"status": "Delayed", "type": "Processing", "risk": "High"}


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
        "match_mode": "strict_word_boundary_only",
    })


@app.route("/interpret", methods=["POST", "OPTIONS"])
def interpret():
    if request.method == "OPTIONS":
        return _preflight_ok()

    data = request.get_json(silent=True) or {}
    dream = (data.get("dream") or data.get("text") or "").strip()

    if not dream:
        return jsonify({"error": "Missing 'dream' or 'text'"}), 400

    try:
        rows = _load_sheet_rows()
    except Exception as e:
        return jsonify({"error": "Sheet load failed", "details": str(e)}), 500

    # STRICT matching only (no fuzzy)
    matches = _match_symbols_strict(dream, rows, top_k=3)

    receipt_id = _make_receipt_id()
    seal = _compute_seal(matches)

    if not matches:
        return jsonify({
            "access": "free",
            "is_paid": False,
            "free_uses_left": 3,
            "seal": seal,
            "receipt": {"id": receipt_id, "top_symbols": [], "share_phrase": "I decoded my dream on Jamaican True Stories."},
            "interpretation": {
                "spiritual_meaning": "No matching symbols were found for the exact words in this dream.",
                "effects_in_physical_realm": "Tip: use clear symbol words (people/animals/places/objects) that are in your Symbols sheet.",
                "what_to_do": "Add 1–2 more key symbols (objects/people/animals/places) and try again."
            }
        })

    interpretation = _combine_fields(matches)

    top_symbols = [
        (row.get("input") or row.get("symbol") or "").strip()
        for row, _ in matches
        if (row.get("input") or row.get("symbol") or "").strip()
    ]

    share_phrase = f"My dream had symbols like: {', '.join(top_symbols[:3])}. I decoded it on Jamaican True Stories."

    return jsonify({
        "access": "free",
        "is_paid": False,
        "free_uses_left": 3,
        "seal": seal,
        "interpretation": interpretation,
        "receipt": {"id": receipt_id, "top_symbols": top_symbols, "share_phrase": share_phrase},
    })


@app.route("/track", methods=["POST", "OPTIONS"])
def track():
    if request.method == "OPTIONS":
        return _preflight_ok()

    payload = request.get_json(silent=True) or {}

    # Stub: keeps your Shopify UI from breaking.
    # Later: store usage in Airtable/DB and return real remaining count.
    return jsonify({
        "ok": True,
        "event": payload.get("event", "unknown"),
        "free_uses_left": payload.get("free_uses_left", 3),
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=True)
