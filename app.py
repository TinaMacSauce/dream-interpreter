# app.py — Jamaican True Stories Dream Interpreter (Level 9 Stable)
# ✅ Modern Google auth (no oauth2client)
# ✅ Strong env-var validation + clearer errors
# ✅ Safer sheet loading (NO get_all_records) + TTL cache
# ✅ Phase 2 upgrades:
#    - exclude_keywords guard (prevents false matches)
#    - priority + category boosts (better ranking)
#    - difflib fallback if rapidfuzz isn’t available
#    - ending-weighted seal selection (seal reflects “final message”)
# ✅ Accepts BOTH payload styles: {dream: "..."} OR {text: "..."}

import os
import re
import time
import difflib
from typing import Dict, List, Tuple, Optional

from flask import Flask, request, jsonify

import gspread
from google.oauth2.service_account import Credentials

try:
    from rapidfuzz import fuzz
except Exception:
    fuzz = None


# ----------------------------
# CONFIG (ENV VARS)
# ----------------------------

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")  # kept for parity (unused here)

GOOGLE_SHEET_KEY = os.environ.get("GOOGLE_SHEET_KEY")  # preferred name
SHEET_ID = os.environ.get("SHEET_ID")                  # alias supported

TAB_SYMBOLS = os.environ.get("TAB_SYMBOLS", "Sheet1")
TAB_RULES = os.environ.get("TAB_RULES", "DOCTRINE_V1_RULES")
TAB_SEALS = os.environ.get("TAB_SEALS", "SEALS")
TAB_CHANGELOG = os.environ.get("TAB_CHANGELOG", "DOCTRINE_V1_CHANGELOG")

MAX_SYMBOLS = int(os.environ.get("MAX_SYMBOLS", 5))
MIN_FUZZY_SCORE = int(os.environ.get("MIN_FUZZY_SCORE", 86))
CACHE_TTL_SECONDS = int(os.environ.get("CACHE_TTL_SECONDS", 300))

CREDS_PATH = os.environ.get("CREDS_PATH", "credentials.json")

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


# ----------------------------
# GOOGLE SHEETS (modern auth)
# ----------------------------

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive.readonly",
]

_cache: Dict[str, List[Dict]] = {}
_cache_time: Dict[str, float] = {}
_gclient = None  # lazy singleton


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


def gclient():
    global _gclient
    if _gclient is not None:
        return _gclient

    creds_path = _require_env("CREDS_PATH", CREDS_PATH)
    creds = Credentials.from_service_account_file(creds_path, scopes=SCOPES)
    _gclient = gspread.authorize(creds)
    return _gclient


def _normalize_key(k: str) -> str:
    # normalize header keys to match your engine expectations
    k = (k or "").strip().lower()
    k = re.sub(r"\s+", " ", k)
    k = k.replace(" ", "_")
    return k


def _dedupe_headers(headers: List[str]) -> List[str]:
    """
    Makes headers unique and non-empty.
    Example: ["locked", "locked"] -> ["locked", "locked__2"]
    Blank headers become: "col__7"
    """
    seen: Dict[str, int] = {}
    out: List[str] = []

    for idx, raw in enumerate(headers):
        base = _normalize_key(raw)
        if not base:
            base = f"col__{idx+1}"

        if base in seen:
            seen[base] += 1
            out.append(f"{base}__{seen[base]}")
        else:
            seen[base] = 1
            out.append(base)

    return out


def load_tab(tab_name: str) -> List[Dict]:
    """
    Safer sheet loading:
    - uses get_all_values() (works even with duplicate headers)
    - dedupes headers
    - returns list[dict] rows
    - TTL cache
    """
    now = time.time()
    if tab_name in _cache and (now - _cache_time.get(tab_name, 0)) < CACHE_TTL_SECONDS:
        return _cache[tab_name]

    sh = gclient().open_by_key(_sheet_key())
    ws = sh.worksheet(tab_name)

    values = ws.get_all_values()
    if not values or len(values) < 2:
        _cache[tab_name] = []
        _cache_time[tab_name] = now
        return []

    raw_headers = values[0]
    headers = _dedupe_headers(raw_headers)

    clean: List[Dict] = []
    for row in values[1:]:
        # pad row to header length
        if len(row) < len(headers):
            row = row + ([""] * (len(headers) - len(row)))

        record = {headers[i]: str(row[i]).strip() for i in range(len(headers))}

        # skip fully empty rows
        if not any(v.strip() for v in record.values()):
            continue

        clean.append(record)

    _cache[tab_name] = clean
    _cache_time[tab_name] = now
    return clean


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
# MATCHING ENGINE
# ----------------------------

def match_symbols(dream: str, symbols: List[Dict]) -> List[Tuple[int, Dict]]:
    dream_n = norm(dream)
    hits: List[Tuple[int, Dict]] = []

    for row in symbols:
        excludes = keywords(row.get("exclude_keywords", ""))
        if excludes and contains_any(dream_n, excludes):
            continue

        inputs = [norm(row.get("input", ""))] + keywords(row.get("keywords", ""))
        base = max((score(dream_n, i) for i in inputs if i), default=0)

        if base < MIN_FUZZY_SCORE:
            continue

        priority = int_or_zero(row.get("priority", 0))
        priority_boost = min(priority, 30)

        cat_raw = norm(row.get("category", "")).replace(" ", "")
        category_boost = CATEGORY_BOOSTS.get(cat_raw, 0)

        rank_score = base + priority_boost + category_boost
        hits.append((rank_score, row))

    hits.sort(reverse=True, key=lambda x: x[0])
    return hits[:MAX_SYMBOLS]


def pick_seal(dream: str, seals: List[Dict], symbol_hits: List[Tuple[int, Dict]]) -> Optional[Dict]:
    dream_n = norm(dream)

    cut = max(1, int(len(dream_n) * 0.65))
    ending = dream_n[cut:].strip()
    if len(ending) < 30:
        ending = dream_n

    symbol_terms = set()
    for _, r in symbol_hits:
        symbol_terms.update(keywords(r.get("keywords", "")))
        symbol_terms.add(norm(r.get("input", "")))

    def seal_score(text: str, srow: Dict) -> int:
        kws = keywords(srow.get("keywords", "")) + [norm(srow.get("seal_name", ""))]
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

    # Accept both payload styles:
    # Shopify JS often sends { text: "..."}; some clients send { dream: "..."}
    dream = (data.get("dream") or data.get("text") or "").strip()

    if not dream:
        return jsonify({"error": "Missing dream text"}), 400

    try:
        symbols = load_tab(TAB_SYMBOLS)
        _ = load_tab(TAB_RULES)   # loaded for parity; not used in output yet
        seals = load_tab(TAB_SEALS)
    except Exception as e:
        return jsonify({"error": f"Sheet load failed: {str(e)}"}), 500

    symbol_hits = match_symbols(dream, symbols)
    seal = pick_seal(dream, seals, symbol_hits)

    spiritual_parts: List[str] = []
    physical_parts: List[str] = []
    action_parts: List[str] = []

    for _, r in symbol_hits:
        spiritual_parts.append(r.get("spiritual_meaning", ""))
        physical_parts.append(r.get("physical_effects", ""))
        action_parts.append(r.get("action", ""))

    return jsonify(
        {
            "interpretation": {
                "spiritual_meaning": " ".join([p for p in spiritual_parts if p]).strip(),
                "effects_in_physical_realm": " ".join([p for p in physical_parts if p]).strip(),
                "what_to_do": " ".join([p for p in action_parts if p]).strip(),
            },
            "seal": seal,
            "disclaimer": DISCLAIMER,
            "meta": {
                "matched_symbols": len(symbol_hits),
                "rapidfuzz_enabled": bool(fuzz),
                "tab_symbols": TAB_SYMBOLS,
                "tab_seals": TAB_SEALS,
            },
        }
    )


@app.get("/health")
def health():
    return jsonify({"ok": True})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
