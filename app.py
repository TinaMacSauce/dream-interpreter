# app.py
# Jamaican True Stories Dream Interpreter
# Structured Doctrine Edition

import os
import re
import json
import logging
import secrets
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from flask import Flask, request, jsonify, session
from flask_cors import CORS

import gspread
from google.oauth2.service_account import Credentials
import stripe

from ai_voice import write_interpretation
from airtable_store import (
    is_paid as airtable_is_paid,
    mark_paid as airtable_mark_paid,
    get_usage,
    set_email_captured,
    increment_free_used,
    save_receipt,
)

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

logging.basicConfig(level=logging.INFO)

# =====================
# ENV CONFIG
# =====================
SHEET_ID = os.environ.get("SHEET_ID", "").strip()
CREDS_PATH = os.environ.get("CREDS_PATH", "credentials.json").strip()

SYMBOLS_WORKSHEET_NAME = os.environ.get("SYMBOLS_WORKSHEET_NAME", "Sheet1").strip()
SEALS_WORKSHEET_NAME = os.environ.get("SEALS_WORKSHEET_NAME", "SEALS").strip()
EVENTS_SHEET_NAME = os.environ.get("EVENTS_SHEET_NAME", "Dream Interpreter Events").strip()

FREE_QUOTA = int(os.environ.get("FREE_QUOTA", "3"))

STRIPE_SECRET_KEY = os.environ.get("STRIPE_SECRET_KEY", "").strip()
STRIPE_WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET", "").strip()
PRICE_WEEKLY = os.environ.get("PRICE_WEEKLY", "").strip()
PRICE_MONTHLY = os.environ.get("PRICE_MONTHLY", "").strip()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4.1-mini").strip()
USE_AI_VOICE = os.environ.get("USE_AI_VOICE", "1") == "1"

DISCLAIMER = (
    "Reflective and educational guidance based on cultural symbolism. "
    "No divination. No predictions. No medical or legal advice."
)

NOTE_TEXT = "Based on traditional symbolism, here is a reflective interpretation."
DOCTRINE_LINE = "The ending of your dream is the receipt. It seals the message."

if STRIPE_SECRET_KEY:
    stripe.api_key = STRIPE_SECRET_KEY

openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY and OpenAI else None

# =====================
# GLOBAL CACHES
# =====================
SYMBOL_ROWS: List[Dict] = []
SEAL_ROWS: List[Dict] = []

# =====================
# GOOGLE SHEETS
# =====================
def gs_client():
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = Credentials.from_service_account_file(CREDS_PATH, scopes=scopes)
    return gspread.authorize(creds)

def load_sheet(name: str) -> List[Dict]:
    gc = gs_client()
    ws = gc.open_by_key(SHEET_ID).worksheet(name)
    return ws.get_all_records()

def refresh_cache():
    global SYMBOL_ROWS, SEAL_ROWS
    SYMBOL_ROWS = load_sheet(SYMBOLS_WORKSHEET_NAME)
    SEAL_ROWS = load_sheet(SEALS_WORKSHEET_NAME)

# =====================
# HELPERS
# =====================
def normalize_dream(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())

def csv_keywords(cell: str) -> List[str]:
    return [k.strip().lower() for k in str(cell).split(",") if k.strip()]

# =====================
# SYMBOL MATCHING
# =====================
def find_best_rows(dream_text: str, max_hits: int = 5) -> List[Dict]:
    dream = normalize_dream(dream_text)
    scored = []

    for row in SYMBOL_ROWS:
        symbol = str(row.get("input", "")).lower().strip()
        if not symbol:
            continue

        score = 0
        if symbol in dream:
            score += 4

        for kw in csv_keywords(row.get("keywords", "")):
            if kw in dream:
                score += 1

        if score > 0:
            scored.append((score, row))

    scored.sort(key=lambda x: x[0], reverse=True)
    seen = set()
    results = []

    for _, row in scored:
        key = row["input"].lower()
        if key not in seen:
            seen.add(key)
            results.append(row)
        if len(results) >= max_hits:
            break

    return results

# =====================
# DOCTRINE COMPOSER
# =====================
def compose_interpretation(rows: List[Dict]) -> Dict[str, object]:
    if not rows:
        return {
            "note": NOTE_TEXT,
            "symbols_matched": [],
            "interpretation": {
                "spiritual_meaning": "No specific symbol matched.",
                "effects_in_the_physical_realm": "Observe patterns and emotions connected to the dream.",
                "what_to_do": "Reflect, pray for clarity, and write the dream down."
            },
            "disclaimer": DISCLAIMER
        }

    combined = {
        "spiritual_meaning": "",
        "effects_in_the_physical_realm": "",
        "what_to_do": ""
    }

    symbols = []

    for row in rows:
        symbols.append(row.get("input", ""))

        combined["spiritual_meaning"] += " " + str(row.get("Spiritual Meaning", "")).strip()
        combined["effects_in_the_physical_realm"] += " " + str(row.get("Physical Effects", "")).strip()
        combined["what_to_do"] += " " + str(row.get("Action", "")).strip()

    for k in combined:
        combined[k] = combined[k].strip()

    return {
        "note": NOTE_TEXT,
        "symbols_matched": symbols,
        "interpretation": combined,
        "disclaimer": DISCLAIMER
    }

# =====================
# AI COMBINER
# =====================
def ai_combine(dream_text: str, rows: List[Dict]) -> Dict[str, str]:
    if not openai_client:
        return compose_interpretation(rows)["interpretation"]

    symbols_payload = []
    for row in rows:
        symbols_payload.append({
            "symbol": row.get("input", ""),
            "spiritual_meaning": row.get("Spiritual Meaning", ""),
            "physical_effects": row.get("Physical Effects", ""),
            "what_to_do": row.get("Action", "")
        })

    system = (
        "You are the Jamaican True Stories dream interpreter. "
        "Use only the meanings provided. "
        "Do not invent meanings. "
        "Return JSON with spiritual_meaning, effects_in_the_physical_realm, what_to_do."
    )

    response = openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(symbols_payload)}
        ],
        temperature=0.3
    )

    return json.loads(response.choices[0].message.content)

# =====================
# FLASK APP
# =====================
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "change-this")
CORS(app)

refresh_cache()

@app.post("/interpret")
def interpret():
    data = request.get_json() or {}
    dream_text = data.get("text", "").strip()
    email = (data.get("email") or "").lower().strip()

    if not dream_text:
        return jsonify({"error": "Dream text required"}), 400

    rows = find_best_rows(dream_text)
    interpretation = compose_interpretation(rows)

    if USE_AI_VOICE:
        interpretation["interpretation"] = write_interpretation(
            dream_text=dream_text,
            doctrine_summary=json.dumps(interpretation["interpretation"]),
            action_guidance="Refine wording only"
        )

    return jsonify(interpretation), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
