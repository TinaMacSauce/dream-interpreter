# app.py — Jamaican True Stories Dream Interpreter (Level 9 Stable)
# ✅ Accepts BOTH payloads: {dream: "..."} OR {text: "..."} (fixes Missing dream text)
# ✅ Modern Google auth (google.oauth2.service_account) (no oauth2client)
# ✅ Strong env-var validation + clear errors
# ✅ Safe sheet loading WITHOUT get_all_records (handles non-unique headers)
# ✅ TTL cache
# ✅ Phase 2 upgrades:
#    - exclude_keywords guard
#    - priority + category boosts
#    - difflib fallback if rapidfuzz isn’t available
#    - ending-weighted seal selection (seal reflects “final message”)
# ✅ CORS allowlist for Shopify domain + local dev

import os
import re
import time
import json
import difflib
from typing import Dict, List, Tuple, Optional

from flask import Flask, request, jsonify, make_response

import gspread
from google.oauth2.service_account import Credentials

try:
    from rapidfuzz import fuzz
except Exception:
    fuzz = None


# ----------------------------
# CONFIG (ENV VARS)
# ----------------------------

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")  # kept for parity; not used here

GOOGLE_SHEET_KEY = os.environ.get("GOOGLE_SHEET_KEY")  # preferred
SHEET_ID = os.environ.get("SHEET_ID")                  # alias supported

TAB_SYMBOLS = os.environ.get("TAB_SYMBOLS", "DOCTRINE_V1_SYMBOLS")  # safer default than Sheet1
TAB_RULES = os.environ.get("TAB_RULES", "DOCTRINE_V1_RULES")
TAB_SEALS = os.environ.get("TAB_SEALS", "SEALS")
TAB_CHANGELOG = os.environ.get("TAB_CHANGELOG", "DOCTRINE_V1_CHANGELOG")

MAX_SYMBOLS = int(os.environ.get("MAX_SYMBOLS", 5))
MIN_FUZZY_SCORE = int(os.environ.get("MIN_FUZZY_SCORE", 86))
CACHE_TTL_SECONDS = int(os.environ.get("CACHE_TTL_SECONDS", 300))

CREDS_PATH = os.environ.get("CREDS_PATH", "credentials.json")

# CORS allowlist:
# Example value:
#   https://jamaicantruestories.com,https://www.jamaicantruestories.com,http://localhost:3000
ALLOWED_ORIGINS = os.environ.get(
    "ALLOWED_ORIGINS",
    "https://jamaicantruestories.com,https://www.jamaicantruestories.com,http://localhost:3000,http://127.0.0.1:3000"
)

DISCLAIMER = (
    "Spiritual disclaimer: This interpretation follows Jamaican True Stories doctrine "
    "and is not medical, legal, or psychological advice."
)

CATEGORY_BOOSTS = {
    "witchcraft": int(os.environ.get("BOOST_WITCHCRAFT", 12)),
    "warfare": int(os.environ.get("BOOST_WARFARE", 10)),
    "death": int(os.environ.get("BOOST_DEATH", 8)),
    "warning": int(os.environ.get("BOOST_WARNING", 7)),
    "cleansing": int(os.environ.get("BOOST_CLEANSING", 6)),
    "blessing": int(os.environ.get("BOOST_BLESSING", 5)),
}


# ----------------------------
# APP
# ----------------------------

app = Flask(__name__)

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets.readonly",
    "https://www.googleapis.com/auth/drive.readonly",
]

_cache: Dict[str, List[Dict]] = {}
_cache_time: Dict[str, float] = {}
_gclient = None  # lazy singleton


# ----------------------------
# UTIL: ENV + CORS
# ----------------------------

def _require_env(name: str, value: Optional[str]) -> str:
    if not value or not str(value).strip():
        raise RuntimeError(f"Missing required env var: {name}")
    return str(value).strip()


def _sheet_key() -> str:
    if GOOGLE_SHEET_KEY and str(GOOGLE_SHEET_KEY).strip():
        return str(GOOGLE_SHEET_KEY).strip()
    if SHEET_ID and str(SHEET_ID).strip():
        return str(SHEET_ID).strip()
    raise RuntimeError("Missing required env var: GOOGLE_SHEET_KEY (or SHEET_ID)")


def _allowed_origins_set() -> set:
    parts = [p.strip() for p in (ALLOWED_ORIGINS or "").split(",") if p.strip()]
    return set(parts)


def _apply_cors(resp):
    origin = request.headers.get("Origin", "")
    allowed = _allowed_origins_set()

    # If request origin is in allowlist, allow it
    if origin and origin in allowed:
        resp.headers["Access-Control-Allow-Origin"] = origin
        resp.headers["Vary"] = "Origin"
        resp.headers["Access-Control-Allow-Credentials"] = "true"
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
        resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return resp


@app.after_request
def after_request(resp):
    return _apply_cors(resp)


@app.route("/interpret", methods=["OPTIONS"])
def interpret_options():
    # Preflight handler
    resp = make_response("", 204)
    return _apply_cors(resp)


# ----------------------------
# GOOGLE SHEETS (modern auth)
# ----------------------------

def gclient():
    global _gclient
    if _gclient is not None:
        return _gclient

    creds_path = _require_env("CREDS_PATH", CREDS_PATH)
    creds = Credentials.from_service_account_file(creds_path, scopes=SCOPES)
    _gclient = gspread.authorize(creds)
    return _gclient


def _normalize_header(h: str) -> str:
    h = (h or "").strip().lower()
    h = re.sub(r"\s+", "_", h)
    h = re.sub(r"[^a-z0-9_]", "", h)
    return h


def _make_headers_unique(headers: List[str]) -> List[str]:
    """
    Ensures headers are unique by suffixing duplicates: name, name__2, name__3...
    This avoids gspread.get_all_records() crashing when headers repeat.
    """
    seen: Dict[str, int] = {}
    out: List[str] = []
    for raw in headers:
        base = _normalize_header(raw)
        if not base:
            base = "col"
        if base not in seen:
            seen[base] = 1
            out.append(base)
        else:
            seen[base] += 1
            out.append(f"{base}__{seen[base]}")
    return out


def load_tab(tab_name: str) -> List[Dict]:
    """
    Loads a worksheet using get_all_values() (safe even if headers repeat).
    Returns list[dict] with normalized UNIQUE keys.
    Uses TTL cache for speed.
    """
    now = time.time()
    if tab_name in _cache and (now - _cache_time.get(tab_name, 0)) < CACHE_TTL_SECONDS:
        return _cache[tab_name]

    sh = gclient().open_by_key(_sheet_key())
    ws = sh.worksheet(tab_name)

    values = ws.get_all_values()  # <-- SAFE
    if not values or len(values) < 2:
        _cache[tab_name] = []
        _cache_time[tab_name] = now
        return []

    raw_headers = values[0]
    headers = _make_headers_unique(raw_headers)

    rows_out: List[Dict] = []
    for row in values[1:]:
        # pad row to header length
        padded = row + [""] * (len(headers) - len(row))
        record = {headers[i]: (padded[i] or "").strip() for i in range(len(headers))}
        # skip fully empty rows
        if not any(v.strip() for v in record.values()):
            continue
        rows_out.append(record)

    _cache[tab_name] = rows_out
    _cache_time[tab_name] = now
    return rows_out


# ----------------------------
# TEXT HELPERS
# ----------------------------

def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9\s']", " ", (s or "").lower()).strip()


def keywords(raw: str) -> List[str]:
    return [norm(k) for k in re.split(r"[;,|]", raw or "") if norm(k)]


def contains_any(text: str, terms: List[str]) -> bool:
    return any(t and t in text for t in terms)


def int_or_zero(x) -> int:
    try:
        return int(str(x).strip())
    except Exception:
        return 0


def score(a: str, b: str) -> int:
    """
    a = dream text (normalized)
    b = candidate keyword/symbol (normalized)
    """
    if not a or not b:
        return 0
    if b in a:
        return 100

    if fuzz:
        try:
            return int(fuzz.partial_ratio(b, a))
        except Exception:
            pass

    return int(difflib.SequenceMatcher(None, b, a).ratio() * 100)


# ----------------------------
# ROW FIELD MAPPING (supports your new Symbols sheet)
# ----------------------------

def pick_first(row: Dict, keys: List[str]) -> str:
    for k in keys:
        v = row.get(k)
        if v is not None and str(v).strip():
            return str(v).strip()
    return ""


def get_symbol_input(row: Dict) -> str:
    # Supports both old schema (input/keywords) and your new schema (symbol, symbol_text, etc.)
    return pick_first(row, [
        "input",
        "symbol",
        "symbol_text",
        "symbolname",
        "symbol_name",
    ])


def get_keywords(row: Dict) -> str:
    # Your sheet has a "common_context..." column that often contains keyword-like phrases.
    # We treat these as optional keywords.
    return pick_first(row, [
        "keywords",
        "common_context_do_not_use_wwrule_links",
        "common_context",
        "context",
        "common_phrases",
        "phrases",
    ])


def get_exclude_keywords(row: Dict) -> str:
    return pick_first(row, [
        "exclude_keywords",
        "exclude",
        "do_not_match",
    ])


def get_category(row: Dict) -> str:
    return pick_first(row, ["category", "seal_category", "type"])


def get_priority(row: Dict) -> str:
    return pick_first(row, ["priority", "rank", "weight"])


def get_spiritual_meaning(row: Dict) -> str:
    # Your Symbols sheet uses Primary_Meaning / Secondary_Meaning
    return pick_first(row, [
        "spiritual_meaning",
        "primary_meaning",
        "meaning",
        "interpretation",
    ])


def get_physical_effects(row: Dict) -> str:
    return pick_first(row, [
        "physical_effects",
        "effects_in_physical_realm",
        "secondary_meaning",
        "secondary_mean",
    ])


def get_action(row: Dict) -> str:
    return pick_first(row, [
        "action",
        "what_to_do",
        "advice",
        "steps",
    ])


# ----------------------------
# MATCHING ENGINE
# ----------------------------

def match_symbols(dream: str, symbols: List[Dict]) -> List[Tuple[int, Dict]]:
    dream_n = norm(dream)
    hits: List[Tuple[int, Dict]] = []

    for row in symbols:
        excludes = keywords(get_exclude_keywords(row))
        if excludes and contains_any(dream_n, excludes):
            continue

        base_input = norm(get_symbol_input(row))
        kw_blob = get_keywords(row)
        inputs = [base_input] + keywords(kw_blob)

        base = max((score(dream_n, i) for i in inputs if i), default=0)
        if base < MIN_FUZZY_SCORE:
            continue

        priority = int_or_zero(get_priority(row))
        priority_boost = min(priority, 30)

        cat_raw = norm(get_category(row)).replace(" ", "")
        category_boost = CATEGORY_BOOSTS.get(cat_raw, 0)

        rank_score = base + priority_boost + category_boost
        hits.append((rank_score, row))

    hits.sort(reverse=True, key=lambda x: x[0])
    return hits[:MAX_SYMBOLS]


def pick_seal(dream: str, seals: List[Dict], symbol_hits: List[Tuple[int, Dict]]) -> Optional[Dict]:
    """
    Ending-weighted seal selection:
    - First pass scores against the END of the dream.
    - Fallback uses full dream if ending score is too low.
    """
    dream_n = norm(dream)

    cut = max(1, int(len(dream_n) * 0.65))
    ending = dream_n[cut:].strip()
    if len(ending) < 30:
        ending = dream_n

    symbol_terms = set()
    for _, r in symbol_hits:
        symbol_terms.update(keywords(get_keywords(r)))
        symbol_terms.add(norm(get_symbol_input(r)))

    def seal_score(text: str, srow: Dict) -> int:
        kws = keywords(pick_first(srow, ["keywords", "keyword", "tags"])) + [norm(pick_first(srow, ["seal_name", "name", "seal"]))]
        base = max((score(text, k) for k in kws if k), default=0)
        boost = 10 if any(k in symbol_terms for k in kws) else 0
        return base + boost

    best = (0, None)

    for s in seals:
        total = seal_score(ending, s)
        if total > best[0]:
            best = (total, s)

    if best[0] < 85:
        for s in seals:
            total = seal_score(dream_n, s)
            if total > best[0]:
                best = (total, s)

    return best[1] if best[0] >= 85 else None


# ----------------------------
# ROUTES
# ----------------------------

@app.post("/interpret")
def interpret():
    data = request.get_json(silent=True) or {}

    # ✅ Accept both keys (fixes your 400)
    dream = (data.get("dream") or data.get("text") or "").strip()

    if not dream:
        return jsonify({"error": "Missing dream text"}), 400

    try:
        symbols = load_tab(TAB_SYMBOLS)
        _ = load_tab(TAB_RULES)   # loaded for parity; not used yet
        seals = load_tab(TAB_SEALS)
    except Exception as e:
        return jsonify({"error": f"Sheet load failed: {str(e)}"}), 500

    symbol_hits = match_symbols(dream, symbols)
    seal = pick_seal(dream, seals, symbol_hits)

    spiritual_parts: List[str] = []
    physical_parts: List[str] = []
    action_parts: List[str] = []

    for _, r in symbol_hits:
        sm = get_spiritual_meaning(r)
        pe = get_physical_effects(r)
        ac = get_action(r)

        if sm: spiritual_parts.append(sm)
        if pe: physical_parts.append(pe)
        if ac: action_parts.append(ac)

    # If your sheet doesn't have action yet, we still return a clean response.
    return jsonify(
        {
            "interpretation": {
                "spiritual_meaning": " ".join(spiritual_parts).strip(),
                "effects_in_the_physical_realm": " ".join(physical_parts).strip(),
                "what_to_do": " ".join(action_parts).strip(),
            },
            "seal": seal,
            "disclaimer": DISCLAIMER,
            "meta": {
                "matched_symbols": len(symbol_hits),
                "rapidfuzz_enabled": bool(fuzz),
                "tab_symbols": TAB_SYMBOLS,
            },
        }
    )


@app.get("/health")
def health():
    return jsonify({"ok": True})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
