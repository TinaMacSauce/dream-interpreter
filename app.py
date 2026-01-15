# app.py — Jamaican True Stories Dream Interpreter (Phase 2: Sheet1)
# Reads from Google Sheet tab "Sheet1" (5-column format)
# Columns expected: input | Spiritual Meaning | Physical Effects | Action | Keywords
#
# Fixes:
# - gspread header row not unique -> uses expected_headers (per gspread docs)
# - Missing 'text' -> accepts text OR dream OR input
# - Basic symbol matching using Keywords + input, with optional fuzzy fallback (rapidfuzz)

import os
import re
import time
from typing import Dict, List, Any, Tuple

from flask import Flask, request, jsonify
from flask_cors import CORS

import gspread
from oauth2client.service_account import ServiceAccountCredentials

try:
    from rapidfuzz import fuzz
except Exception:
    fuzz = None


# -------------------------
# Config
# -------------------------
APP_VERSION = os.getenv("APP_VERSION", "phase2-sheet1-v1")

GOOGLE_SHEET_KEY = os.getenv("GOOGLE_SHEET_KEY")  # spreadsheet ID (the long key)
WORKSHEET_NAME = os.getenv("WORKSHEET_NAME", "Sheet1")  # 5-column tab
CREDS_PATH = os.getenv("CREDS_PATH", "credentials.json")

CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "300"))
MAX_SYMBOLS = int(os.getenv("MAX_SYMBOLS", "5"))
MIN_FUZZY_SCORE = int(os.getenv("MIN_FUZZY_SCORE", "85"))

# If you keep credentials/cookies OFF in the Shopify fetch (recommended),
# we can allow all origins safely.
ALLOW_ALL_ORIGINS = os.getenv("ALLOW_ALL_ORIGINS", "true").lower() == "true"

# If you insist on credentials/cookies in fetch, set this false and provide allowed origins.
ALLOWED_ORIGINS = [
    "https://jamaicantruestories.com",
    "https://www.jamaicantruestories.com",
    # Shopify preview + checkout can be on these domains:
    # Add your exact shop domain if needed:
    # "https://your-shop.myshopify.com",
]


EXPECTED_HEADERS = [
    "input",
    "Spiritual Meaning",
    "Physical Effects",
    "Action",
    "Keywords",
]


# -------------------------
# App
# -------------------------
app = Flask(__name__)

if ALLOW_ALL_ORIGINS:
    # Works best if your frontend fetch does NOT use credentials:"include"
    CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=False)
else:
    # If you use credentials/cookies, you cannot use "*" safely.
    # You must list exact origins.
    CORS(app, resources={r"/*": {"origins": ALLOWED_ORIGINS}}, supports_credentials=True)


# -------------------------
# Google Sheets loader + cache
# -------------------------
_cache = {
    "loaded_at": 0.0,
    "rows": [],
    "index": {},       # input -> row
    "keywords": [],    # list of (keyword, input_key)
    "inputs": [],      # list of input strings (for fuzzy)
}

def _now() -> float:
    return time.time()

def _normalize(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

def _tokenize(text: str) -> List[str]:
    text = _normalize(text)
    # keep letters/numbers/spaces only for matching
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    return text.split()

def _connect_gspread():
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive.file",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = ServiceAccountCredentials.from_json_keyfile_name(CREDS_PATH, scope)
    client = gspread.authorize(creds)
    return client

def _build_indexes(rows: List[Dict[str, Any]]) -> Tuple[Dict[str, Dict[str, Any]], List[Tuple[str, str]], List[str]]:
    index: Dict[str, Dict[str, Any]] = {}
    kw_pairs: List[Tuple[str, str]] = []
    inputs: List[str] = []

    for r in rows:
        inp = _normalize(r.get("input", ""))
        if not inp:
            continue

        # store the canonical row
        index[inp] = r
        inputs.append(inp)

        # keywords can include comma-separated terms and multi-phrases
        kws_raw = r.get("Keywords", "") or ""
        kws = [k.strip().lower() for k in kws_raw.split(",") if k.strip()]
        # always include the input phrase itself as a keyword
        kws.append(inp)

        for k in kws:
            k = _normalize(k)
            if k:
                kw_pairs.append((k, inp))

    # Sort longer keywords first so "running in murky waters" hits before "running"
    kw_pairs.sort(key=lambda x: len(x[0]), reverse=True)

    return index, kw_pairs, inputs

def load_sheet_data(force: bool = False) -> None:
    global _cache

    if (not force) and (_cache["rows"]) and (_now() - _cache["loaded_at"] < CACHE_TTL_SECONDS):
        return

    if not GOOGLE_SHEET_KEY:
        raise RuntimeError("GOOGLE_SHEET_KEY is not set in environment variables.")

    client = _connect_gspread()
    sh = client.open_by_key(GOOGLE_SHEET_KEY)

    ws = sh.worksheet(WORKSHEET_NAME)

    # Use expected_headers to avoid the "header row is not unique" crash.
    # gspread docs: expected_headers must be unique. :contentReference[oaicite:3]{index=3}
    rows = ws.get_all_records(
        expected_headers=EXPECTED_HEADERS,
        head=1
    )

    index, kw_pairs, inputs = _build_indexes(rows)

    _cache = {
        "loaded_at": _now(),
        "rows": rows,
        "index": index,
        "keywords": kw_pairs,
        "inputs": inputs,
    }


# -------------------------
# Matching logic
# -------------------------
def match_symbols(dream_text: str) -> List[Dict[str, Any]]:
    """
    Returns top matched rows from Sheet1.
    Priority:
      1) keyword substring hits (including multi-phrase keywords)
      2) fuzzy fallback if no hits and rapidfuzz available
    """
    load_sheet_data()

    dream_norm = _normalize(dream_text)
    if not dream_norm:
        return []

    hits: List[str] = []
    used = set()

    # 1) keyword substring matches
    for kw, inp in _cache["keywords"]:
        if inp in used:
            continue
        if kw and kw in dream_norm:
            hits.append(inp)
            used.add(inp)
            if len(hits) >= MAX_SYMBOLS:
                break

    # 2) fuzzy fallback (only if no keyword hits)
    if not hits and fuzz and _cache["inputs"]:
        # Compare dream against each input phrase
        scored = []
        for inp in _cache["inputs"]:
            score = fuzz.partial_ratio(dream_norm, inp)
            if score >= MIN_FUZZY_SCORE:
                scored.append((score, inp))
        scored.sort(reverse=True, key=lambda x: x[0])

        for score, inp in scored[:MAX_SYMBOLS]:
            hits.append(inp)

    # Build rows
    out = []
    for inp in hits:
        r = _cache["index"].get(inp)
        if r:
            out.append(r)
    return out


def combine_interpretation(rows: List[Dict[str, Any]]) -> Dict[str, str]:
    """
    Combines multiple matched symbols into the 3-section format.
    IMPORTANT: Does not invent meanings—only uses sheet text.
    """
    spiritual_parts = []
    physical_parts = []
    action_parts = []
    matched = []

    for r in rows:
        inp = (r.get("input") or "").strip()
        matched.append(inp)

        sm = (r.get("Spiritual Meaning") or "").strip()
        pe = (r.get("Physical Effects") or "").strip()
        ac = (r.get("Action") or "").strip()

        if sm:
            spiritual_parts.append(f"• {inp}: {sm}")
        if pe:
            physical_parts.append(f"• {inp}: {pe}")
        if ac:
            action_parts.append(f"• {inp}: {ac}")

    return {
        "spiritual_meaning": "\n".join(spiritual_parts).strip(),
        "effects_in_the_physical_realm": "\n".join(physical_parts).strip(),
        "what_to_do": "\n".join(action_parts).strip(),
        "matched_symbols": matched,
    }


# -------------------------
# Routes
# -------------------------
@app.get("/health")
def health():
    return jsonify({"ok": True, "version": APP_VERSION, "worksheet": WORKSHEET_NAME})

@app.post("/interpret")
def interpret():
    # Accept JSON only
    data = request.get_json(silent=True) or {}

    # Backward/Shopify-safe: accept text OR dream OR input
    text = (data.get("text") or data.get("dream") or data.get("input") or "").strip()
    email = (data.get("email") or "").strip()

    if not text:
        return jsonify({"error": "Missing 'text'"}), 400

    try:
        rows = match_symbols(text)
        interpretation = combine_interpretation(rows)

        # If nothing matched, return a clean “no match” result
        if not interpretation.get("matched_symbols"):
            interpretation = {
                "spiritual_meaning": "No matching symbols found in the dictionary for this dream text.",
                "effects_in_the_physical_realm": "Try adding 1–2 clear symbols (example: running, chased, teeth, water, snake).",
                "what_to_do": "Add more detail and try again. (This engine only uses your sheet—no guessing.)",
                "matched_symbols": [],
            }

        return jsonify({
            "access": "ok",
            "interpretation": {
                "spiritual_meaning": interpretation["spiritual_meaning"],
                "effects_in_the_physical_realm": interpretation["effects_in_the_physical_realm"],
                "what_to_do": interpretation["what_to_do"],
            },
            "matched_symbols": interpretation["matched_symbols"],
            "email_received": bool(email),
            "version": APP_VERSION,
        })

    except Exception as e:
        # Surface the real reason to you (without exposing secrets)
        return jsonify({"error": f"Interpret failed: {str(e)}"}), 500


# Local dev only
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=True)
