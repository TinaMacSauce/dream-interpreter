# app.py — Jamaican True Stories Dream Interpreter (updated)
# Key changes reflected:
# 1) NO hard-coded API keys in code.
# 2) OpenAI key is read from environment variable: OPENAI_API_KEY
# 3) Safer OpenAI call: never sends null/blank content; clearer errors.
# 4) Works even if you choose to disable OpenAI (USE_OPENAI=false).

import os
import re
import json
import time
from difflib import SequenceMatcher
from typing import List, Dict, Tuple, Optional

from flask import Flask, request, jsonify, render_template

# --- Google Sheets (gspread) ---
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# --- Optional OpenAI (only if you want it) ---
try:
    from openai import OpenAI
except Exception:
    OpenAI = None


# =========================
# CONFIG
# =========================
APP_TITLE = "Jamaican True Stories Dream Interpreter"

GOOGLE_SHEET_NAME = os.getenv("GOOGLE_SHEET_NAME", "DOCTRINE_V1_SYMBOLS")  # or your workbook name
SYMBOLS_TAB = os.getenv("SYMBOLS_TAB", "SYMBOLS")
SEALS_TAB = os.getenv("SEALS_TAB", "SEALS")

# If you’re running on Render, set these in Render -> Environment:
# - OPENAI_API_KEY
# - USE_OPENAI (true/false)
# - OPENAI_MODEL (ex: gpt-4.1-mini)
USE_OPENAI = os.getenv("USE_OPENAI", "false").strip().lower() in ("1", "true", "yes", "y")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

# Fuzzy matching threshold
FUZZY_THRESHOLD = float(os.getenv("FUZZY_THRESHOLD", "0.78"))

# Top matches to return
TOP_K = int(os.getenv("TOP_K", "5"))


# =========================
# FLASK
# =========================
app = Flask(__name__)


# =========================
# HELPERS
# =========================
def _clean_text(s: str) -> str:
    if not s:
        return ""
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s


def _ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _require_env(name: str) -> str:
    val = os.getenv(name, "")
    if not val:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return val


def get_gspread_client() -> gspread.Client:
    """
    Expects a service account JSON file at ./credentials.json
    OR a JSON string in env var GOOGLE_CREDENTIALS_JSON.
    """
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]

    # Option A: credentials.json file
    if os.path.exists("credentials.json"):
        creds = ServiceAccountCredentials.from_json_keyfile_name("credentials.json", scope)
        return gspread.authorize(creds)

    # Option B: JSON in env var
    raw = os.getenv("GOOGLE_CREDENTIALS_JSON", "")
    if raw:
        data = json.loads(raw)
        creds = ServiceAccountCredentials.from_json_keyfile_dict(data, scope)
        return gspread.authorize(creds)

    raise RuntimeError("Google credentials not found. Add credentials.json or GOOGLE_CREDENTIALS_JSON.")


def read_sheet_records(sheet_name: str, tab_name: str) -> List[Dict[str, str]]:
    gc = get_gspread_client()
    ss = gc.open(sheet_name)
    ws = ss.worksheet(tab_name)
    rows = ws.get_all_records()  # uses first row as headers
    # Normalize keys + values
    norm = []
    for r in rows:
        norm.append({str(k).strip(): ("" if r[k] is None else str(r[k]).strip()) for k in r})
    return norm


def build_symbol_index(records: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Expected columns (flexible but recommended):
      - Symbol
      - Spiritual Meaning
      - Effects in the Physical Realm
      - What to Do
      - Keywords (optional)
      - Seal (optional)
    """
    out = []
    for r in records:
        sym = _clean_text(r.get("Symbol", ""))
        if not sym:
            continue
        out.append({
            "symbol": sym,
            "spiritual_meaning": _clean_text(r.get("Spiritual Meaning", "")),
            "effects": _clean_text(r.get("Effects in the Physical Realm", "")),
            "what_to_do": _clean_text(r.get("What to Do", "")),
            "keywords": _clean_text(r.get("Keywords", "")),
            "seal": _clean_text(r.get("Seal", "")),
        })
    return out


def match_symbols(user_text: str, symbols: List[Dict[str, str]], k: int = TOP_K) -> List[Tuple[float, Dict[str, str]]]:
    """
    Match by:
      1) direct substring hit on symbol or keywords
      2) fuzzy similarity to symbol string
    """
    t = _clean_text(user_text).lower()
    if not t:
        return []

    scored: List[Tuple[float, Dict[str, str]]] = []

    for item in symbols:
        sym = item["symbol"].lower()
        kws = item.get("keywords", "").lower()

        # Direct trigger
        direct = 0.0
        if sym and sym in t:
            direct = 1.0
        elif kws:
            # any keyword phrase found
            for kw in [k.strip() for k in kws.split(",") if k.strip()]:
                if kw and kw in t:
                    direct = max(direct, 0.92)
                    break

        # Fuzzy
        fuzz = _ratio(t, sym) if sym else 0.0

        score = max(direct, fuzz)
        if score >= FUZZY_THRESHOLD or direct >= 0.9:
            scored.append((score, item))

    scored.sort(key=lambda x: x[0], reverse=True)

    # Deduplicate by symbol
    seen = set()
    uniq = []
    for sc, it in scored:
        key = it["symbol"].lower()
        if key in seen:
            continue
        seen.add(key)
        uniq.append((sc, it))
        if len(uniq) >= k:
            break

    return uniq


def format_interpretation(matches: List[Tuple[float, Dict[str, str]]]) -> str:
    if not matches:
        return (
            "Spiritual Meaning: I couldn’t match any symbols from your doctrine yet.\n"
            "Effects in the Physical Realm: This usually means your input needs more detail or the symbol isn’t in the sheet.\n"
            "What to Do: Add a few more specifics (who/what/where/how you felt), or add the missing symbol to your doctrine sheet."
        )

    # Build a clean 3-part format using ONLY the sheet content (no invention)
    spiritual_bits = []
    effects_bits = []
    todo_bits = []

    for score, it in matches:
        label = it["symbol"]
        if it["spiritual_meaning"]:
            spiritual_bits.append(f"{label}: {it['spiritual_meaning']}")
        if it["effects"]:
            effects_bits.append(f"{label}: {it['effects']}")
        if it["what_to_do"]:
            todo_bits.append(f"{label}: {it['what_to_do']}")

    spiritual = " | ".join(spiritual_bits) if spiritual_bits else "Matched symbols found, but spiritual meaning is blank in your sheet."
    effects = " | ".join(effects_bits) if effects_bits else "Matched symbols found, but effects are blank in your sheet."
    todo = " | ".join(todo_bits) if todo_bits else "Matched symbols found, but what-to-do is blank in your sheet."

    return f"Spiritual Meaning: {spiritual}\nEffects in the Physical Realm: {effects}\nWhat to Do: {todo}"


def openai_polish_only(sheet_based_text: str) -> str:
    """
    OPTIONAL: Polishes wording ONLY. It must not add meanings.
    This is the “changed” safety: never send blank content.
    """
    if not USE_OPENAI:
        return sheet_based_text

    if OpenAI is None:
        return sheet_based_text

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        # If you want OpenAI, you must set OPENAI_API_KEY in env.
        return sheet_based_text

    cleaned = _clean_text(sheet_based_text)
    if not cleaned:
        return sheet_based_text  # never send empty/None

    client = OpenAI(api_key=api_key)

    system = (
        "You are a formatter. You may ONLY rephrase for clarity and warmth. "
        "You must NOT add any new meanings, symbols, spiritual claims, or advice. "
        "Keep the exact 3 headings and keep content faithful to the input."
    )

    user = (
        "Polish this text for readability while preserving meaning exactly:\n\n"
        f"{cleaned}"
    )

    # Safe call: always a string, never null
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
    )

    out = resp.choices[0].message.content
    return out.strip() if out else sheet_based_text


# =========================
# ROUTES
# =========================
@app.get("/")
def home():
    return f"{APP_TITLE} is running."


@app.post("/interpret")
def interpret():
    payload = request.get_json(silent=True) or {}
    dream_text = _clean_text(payload.get("dream", ""))

    if not dream_text:
        return jsonify({"error": "dream is required"}), 400

    # Read doctrine from Google Sheets
    try:
        symbol_records = read_sheet_records(GOOGLE_SHEET_NAME, SYMBOLS_TAB)
        symbols = build_symbol_index(symbol_records)
    except Exception as e:
        return jsonify({"error": f"Failed to read Google Sheets: {str(e)}"}), 500

    matches = match_symbols(dream_text, symbols, k=TOP_K)
    base = format_interpretation(matches)

    # Optional polish only (no invention)
    final = openai_polish_only(base)

    return jsonify({
        "dream": dream_text,
        "matches": [{"score": round(sc, 3), "symbol": it["symbol"]} for sc, it in matches],
        "result": final
    })


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    # Render uses PORT
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)
