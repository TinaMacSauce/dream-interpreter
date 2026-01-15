# app.py — Jamaican True Stories Dream Interpreter (Level 9 Stable)
# ✅ Updated: removed deprecated oauth2client
# ✅ Uses: google.oauth2.service_account.Credentials (works with your current requirements.txt)
# ✅ Adds: clearer env-var checks + safer sheet loading

import os
import re
import time
from typing import Dict, List, Tuple, Optional

from flask import Flask, request, jsonify

import gspread
from google.oauth2.service_account import Credentials  # ✅ modern auth

try:
    from rapidfuzz import fuzz
except Exception:
    fuzz = None


# ----------------------------
# CONFIG (ENV VARS)
# ----------------------------

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")  # (not used in this file yet, but kept)
GOOGLE_SHEET_KEY = os.environ.get("GOOGLE_SHEET_KEY")

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


def gclient():
    """
    Returns an authorized gspread client.
    Uses service account creds from CREDS_PATH.
    """
    global _gclient
    if _gclient is not None:
        return _gclient

    creds_path = _require_env("CREDS_PATH", CREDS_PATH)

    # If you store the JSON file in the repo, CREDS_PATH can be "credentials.json".
    # If you store it elsewhere on Render, set CREDS_PATH to the absolute path.
    creds = Credentials.from_service_account_file(creds_path, scopes=SCOPES)
    _gclient = gspread.authorize(creds)
    return _gclient


def load_tab(tab_name: str) -> List[Dict]:
    """
    Loads a worksheet tab and returns cleaned rows as dicts with normalized keys.
    Uses an in-memory TTL cache for speed.
    """
    now = time.time()
    if tab_name in _cache and (now - _cache_time.get(tab_name, 0)) < CACHE_TTL_SECONDS:
        return _cache[tab_name]

    sheet_key = _require_env("GOOGLE_SHEET_KEY", GOOGLE_SHEET_KEY)

    sh = gclient().open_by_key(sheet_key)
    ws = sh.worksheet(tab_name)
    rows = ws.get_all_records()  # list[dict] with header row keys

    clean: List[Dict] = []
    for r in rows:
        # skip blank rows
        if not any(str(v).strip() for v in r.values()):
            continue

        clean.append(
            {
                str(k).lower().strip().replace(" ", "_"): str(v).strip()
                for k, v in r.items()
            }
        )

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


def score(a: str, b: str) -> int:
    # a = dream text (normalized), b = candidate keyword/symbol (normalized)
    if not a or not b:
        return 0
    if b in a:
        return 100
    if fuzz:
        return int(fuzz.partial_ratio(b, a))
    return 0


# ----------------------------
# MATCHING ENGINE
# ----------------------------

def match_symbols(dream: str, symbols: List[Dict]) -> List[Tuple[int, Dict]]:
    dream_n = norm(dream)
    hits: List[Tuple[int, Dict]] = []

    for row in symbols:
        # expects columns like: input, keywords, spiritual_meaning, physical_effects, action
        inputs = [norm(row.get("input", ""))] + keywords(row.get("keywords", ""))
        best = max((score(dream_n, i) for i in inputs if i), default=0)

        if best >= MIN_FUZZY_SCORE:
            hits.append((best, row))

    hits.sort(reverse=True, key=lambda x: x[0])
    return hits[:MAX_SYMBOLS]


def pick_seal(dream: str, seals: List[Dict], symbols: List[Tuple[int, Dict]]) -> Optional[Dict]:
    dream_n = norm(dream)
    symbol_terms = set()

    for _, r in symbols:
        symbol_terms.update(keywords(r.get("keywords", "")))
        symbol_terms.add(norm(r.get("input", "")))

    best_score = 0
    best_seal = None

    for s in seals:
        kws = keywords(s.get("keywords", "")) + [norm(s.get("seal_name", ""))]
        base = max((score(dream_n, k) for k in kws if k), default=0)
        boost = 10 if any(k in symbol_terms for k in kws) else 0
        total = base + boost

        if total > best_score:
            best_score = total
            best_seal = s

    return best_seal if best_score >= 85 else None


# ----------------------------
# ROUTES
# ----------------------------

@app.post("/interpret")
def interpret():
    data = request.get_json(silent=True) or {}
    dream = (data.get("dream") or "").strip()

    if not dream:
        return jsonify({"error": "Missing dream text"}), 400

    try:
        symbols = load_tab(TAB_SYMBOLS)
        _ = load_tab(TAB_RULES)  # loaded (kept for parity), not used in current output
        seals = load_tab(TAB_SEALS)
    except Exception as e:
        # Make sheet/auth errors obvious in Render logs and in response
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

    return jsonify({
        "interpretation": {
            "spiritual_meaning": " ".join([p for p in spiritual_parts if p]).strip(),
            "effects_in_physical_realm": " ".join([p for p in physical_parts if p]).strip(),
            "what_to_do": " ".join([p for p in action_parts if p]).strip(),
        },
        "seal": seal,
        "disclaimer": DISCLAIMER
    })


@app.get("/health")
def health():
    return jsonify({"ok": True})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
