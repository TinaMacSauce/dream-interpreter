# app.py — JTS Dream Interpreter (Sheet1 / 5-column)
# Reads from Google Sheet tab: Sheet1
# Columns expected (case-insensitive):
#   input | Spiritual Meaning | Physical Effects | Action | Keywords
#
# Fixes:
# - Avoids get_all_records header issues
# - Deterministic mapping so "running" can't pull meanings from a different row
# - Keyword + fuzzy match against "input" + "keywords"

import os
import re
import json
import time
from typing import Dict, List, Tuple, Optional

from flask import Flask, request, jsonify
from flask_cors import CORS

import gspread
from google.oauth2.service_account import Credentials

from rapidfuzz import fuzz

# ----------------------------
# Config (ENV VARS)
# ----------------------------
SHEET_ID = os.getenv("SHEET_ID", "").strip()
GOOGLE_SHEET_KEY = os.getenv("GOOGLE_SHEET_KEY", "").strip()  # allow either name
SPREADSHEET_ID = SHEET_ID or GOOGLE_SHEET_KEY

WORKSHEET_NAME = os.getenv("WORKSHEET_NAME", "Sheet1").strip()  # <— set this to Sheet1
CREDS_PATH = os.getenv("CREDS_PATH", "credentials.json").strip()

# Matching
MIN_FUZZY_SCORE = int(os.getenv("MIN_FUZZY_SCORE", "86"))
MAX_SYMBOLS = int(os.getenv("MAX_SYMBOLS", "6"))

# Caching
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "300"))

# Simple free quota tracking (optional)
FREE_QUOTA = int(os.getenv("FREE_QUOTA", "3"))
COUNTS_FILE = os.getenv("COUNTS_FILE", "counts.json")

# Optional: Restrict CORS to your Shopify domain(s)
# If empty => allow all (not recommended long-term)
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "").strip()
if CORS_ORIGINS:
    ALLOWED_ORIGINS = [o.strip() for o in CORS_ORIGINS.split(",") if o.strip()]
else:
    ALLOWED_ORIGINS = ["*"]

# ----------------------------
# Flask App
# ----------------------------
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ALLOWED_ORIGINS}})

# ----------------------------
# Utilities
# ----------------------------
def now_ts() -> float:
    return time.time()

def norm(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

def tokenize(s: str) -> List[str]:
    s = norm(s)
    # keep letters/numbers/spaces only
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        return []
    return s.split()

def safe_load_json(path: str) -> Dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def safe_save_json(path: str, data: Dict) -> None:
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

# ----------------------------
# Google Sheets Client
# ----------------------------
def make_gspread_client():
    if not os.path.exists(CREDS_PATH):
        raise RuntimeError(f"CREDS_PATH not found: {CREDS_PATH}")

    scopes = [
        "https://www.googleapis.com/auth/spreadsheets.readonly",
        "https://www.googleapis.com/auth/drive.readonly",
    ]
    creds = Credentials.from_service_account_file(CREDS_PATH, scopes=scopes)
    return gspread.authorize(creds)

# ----------------------------
# Sheet Loader (SAFE)
# ----------------------------
_symbols_cache: Dict = {"ts": 0, "rows": []}

# We expect these columns in Sheet1 (any case/spaces)
EXPECTED_COLS = {
    "input": ["input", "symbol", "dream symbol", "dream_symbols", "dreamsymbol"],
    "spiritual": ["spiritual meaning", "spiritual_meaning", "spiritual"],
    "physical": ["physical effects", "physical_effects", "effects", "physical"],
    "action": ["action", "what to do", "what_to_do", "instructions"],
    "keywords": ["keywords", "keyword", "common context", "common_conte", "common_context"],
}

def _find_col_index(headers: List[str], aliases: List[str]) -> Optional[int]:
    # headers are normalized
    alias_set = set([norm(a) for a in aliases])
    for idx, h in enumerate(headers):
        if norm(h) in alias_set:
            return idx
    return None

def load_symbols_sheet1(force: bool = False) -> List[Dict]:
    """Loads Sheet1 rows into normalized records:
       {input, spiritual, physical, action, keywords_list}
    """
    global _symbols_cache
    if not force and _symbols_cache["rows"] and (now_ts() - _symbols_cache["ts"] < CACHE_TTL_SECONDS):
        return _symbols_cache["rows"]

    if not SPREADSHEET_ID:
        raise RuntimeError("Missing SHEET_ID or GOOGLE_SHEET_KEY env var.")

    gc = make_gspread_client()
    sh = gc.open_by_key(SPREADSHEET_ID)
    ws = sh.worksheet(WORKSHEET_NAME)

    values = ws.get_all_values()  # raw table
    if not values or len(values) < 2:
        _symbols_cache = {"ts": now_ts(), "rows": []}
        return []

    raw_headers = values[0]
    headers = [norm(h) for h in raw_headers]

    idx_input = _find_col_index(headers, EXPECTED_COLS["input"])
    idx_spiritual = _find_col_index(headers, EXPECTED_COLS["spiritual"])
    idx_physical = _find_col_index(headers, EXPECTED_COLS["physical"])
    idx_action = _find_col_index(headers, EXPECTED_COLS["action"])
    idx_keywords = _find_col_index(headers, EXPECTED_COLS["keywords"])

    missing = []
    if idx_input is None: missing.append("input")
    if idx_spiritual is None: missing.append("Spiritual Meaning")
    if idx_physical is None: missing.append("Physical Effects")
    if idx_action is None: missing.append("Action")
    if idx_keywords is None: missing.append("Keywords")

    if missing:
        raise RuntimeError(
            "Sheet1 column(s) missing or not found: "
            + ", ".join(missing)
            + ". Check your header row spelling."
        )

    rows: List[Dict] = []
    for r in values[1:]:
        # pad short rows
        while len(r) < len(raw_headers):
            r.append("")

        input_txt = (r[idx_input] or "").strip()
        if not input_txt:
            continue

        spiritual_txt = (r[idx_spiritual] or "").strip()
        physical_txt = (r[idx_physical] or "").strip()
        action_txt = (r[idx_action] or "").strip()
        keywords_txt = (r[idx_keywords] or "").strip()

        # keywords may be comma-separated or space-separated phrases
        kw_list: List[str] = []
        if keywords_txt:
            # split on commas first, fallback to semicolons/newlines
            parts = re.split(r"[,\n;]+", keywords_txt)
            kw_list = [norm(p) for p in parts if norm(p)]

        rows.append({
            "input": input_txt.strip(),
            "input_norm": norm(input_txt),
            "spiritual": spiritual_txt,
            "physical": physical_txt,
            "action": action_txt,
            "keywords": kw_list,
        })

    _symbols_cache = {"ts": now_ts(), "rows": rows}
    return rows

# ----------------------------
# Matching Logic (deterministic)
# ----------------------------
def best_matches(dream_text: str, rows: List[Dict], max_results: int) -> List[Tuple[int, Dict]]:
    dt = norm(dream_text)
    if not dt:
        return []

    # Quick token set for contains checks
    dt_tokens = set(tokenize(dt))

    scored: List[Tuple[int, Dict]] = []

    for row in rows:
        base = row["input_norm"]
        if not base:
            continue

        # 1) Strong direct contains: if input phrase appears in dream text
        if len(base) >= 3 and base in dt:
            score = 100
            scored.append((score, row))
            continue

        # 2) Keyword contains: any keyword phrase appears
        kw_hit = False
        for kw in row.get("keywords", []):
            if kw and kw in dt:
                scored.append((98, row))
                kw_hit = True
                break
        if kw_hit:
            continue

        # 3) Token overlap (cheap)
        base_tokens = set(tokenize(base))
        if base_tokens and (len(base_tokens & dt_tokens) / max(1, len(base_tokens))) >= 0.6:
            scored.append((92, row))
            continue

        # 4) Fuzzy match against dream text (last resort)
        # Use partial_ratio because input phrases are often shorter than the dream
        f = fuzz.partial_ratio(base, dt)
        if f >= MIN_FUZZY_SCORE:
            scored.append((int(f), row))

    # Sort high→low, keep unique "input"
    scored.sort(key=lambda x: x[0], reverse=True)
    seen = set()
    out: List[Tuple[int, Dict]] = []
    for s, r in scored:
        key = r["input_norm"]
        if key in seen:
            continue
        seen.add(key)
        out.append((s, r))
        if len(out) >= max_results:
            break
    return out

def compose_interpretation(primary_row: Dict, matched_rows: List[Dict]) -> Dict:
    # Primary drives the 3-part meaning (this prevents "running" being overwritten)
    spiritual = (primary_row.get("spiritual") or "").strip()
    physical = (primary_row.get("physical") or "").strip()
    action = (primary_row.get("action") or "").strip()

    # Build receipt symbols list
    top_symbols = [r["input"] for r in matched_rows if r.get("input")][:MAX_SYMBOLS]

    return {
        "interpretation": {
            "spiritual_meaning": spiritual,
            "effects_in_the_physical_realm": physical,
            "what_to_do": action
        },
        "receipt": {
            "code_name": "JTS DREAM RECEIPT",
            "top_symbols": top_symbols,
            "share_phrase": f"I ran my dream through Jamaican True Stories. Top symbols: {', '.join(top_symbols[:3])}."
                          if top_symbols else "I ran my dream through Jamaican True Stories.",
            "doctrine_line": ""
        }
    }

# ----------------------------
# Free quota tracking (simple)
# ----------------------------
def get_user_key(email: str, user_id: str) -> str:
    email = (email or "").strip().lower()
    user_id = (user_id or "").strip()
    return email if email else (user_id if user_id else "anon")

def read_counts() -> Dict:
    return safe_load_json(COUNTS_FILE)

def write_counts(data: Dict) -> None:
    safe_save_json(COUNTS_FILE, data)

def consume_free_use(user_key: str) -> Tuple[int, bool]:
    """Returns (free_uses_left, allowed)"""
    data = read_counts()
    u = data.get(user_key, {})
    used = int(u.get("used", 0))
    if used >= FREE_QUOTA:
        return (0, False)
    used += 1
    u["used"] = used
    data[user_key] = u
    write_counts(data)
    left = max(0, FREE_QUOTA - used)
    return (left, True)

# ----------------------------
# Routes
# ----------------------------
@app.get("/health")
def health():
    return jsonify({
        "ok": True,
        "worksheet": WORKSHEET_NAME,
        "sheet_id_present": bool(SPREADSHEET_ID),
        "cache_ttl": CACHE_TTL_SECONDS
    })

@app.post("/interpret")
def interpret():
    payload = request.get_json(silent=True) or {}
    text = (payload.get("text") or "").strip()
    email = (payload.get("email") or "").strip().lower()
    user_id = (payload.get("user_id") or "").strip()

    if not text:
        return jsonify({"error": "Missing 'text'"}), 400

    # Load symbols
    try:
        rows = load_symbols_sheet1()
    except Exception as e:
        return jsonify({"error": f"Sheet load failed: {str(e)}"}), 500

    # Match
    matches = best_matches(text, rows, max_results=MAX_SYMBOLS)
    if not matches:
        # No match fallback
        return jsonify({
            "access": "free",
            "is_paid": False,
            "free_uses_left": FREE_QUOTA,  # unknown; we keep simple here
            "interpretation": {
                "spiritual_meaning": "No clear symbol match found yet. Try adding 1–2 key symbols (e.g., running, water, teeth, snakes).",
                "effects_in_the_physical_realm": "The meaning depends on the strongest symbol in the dream. Add a few more details and re-try.",
                "what_to_do": "Rewrite the dream in 1–3 sentences and include how it ended."
            },
            "receipt": {
                "code_name": "JTS DREAM RECEIPT",
                "top_symbols": [],
                "share_phrase": "I ran my dream through Jamaican True Stories.",
                "doctrine_line": ""
            }
        }), 200

    # Free quota gate (optional)
    user_key = get_user_key(email, user_id)
    free_left, allowed = consume_free_use(user_key)

    if not allowed:
        # paywall response shape to match your Shopify frontend
        return jsonify({
            "paywall": True,
            "message": "Free limit reached. Subscribe to unlock full interpretations.",
            "checkout_weekly": os.getenv("CHECKOUT_WEEKLY", ""),
            "checkout_monthly": os.getenv("CHECKOUT_MONTHLY", ""),
            "free_uses_left": 0,
            "access": "paywall",
            "is_paid": False
        }), 200

    # Primary match = best score
    primary = matches[0][1]
    matched_rows = [m[1] for m in matches]

    out = compose_interpretation(primary, matched_rows)
    out.update({
        "access": "free",
        "is_paid": False,
        "free_uses_left": free_left
    })

    return jsonify(out), 200

# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)
