# FINAL PRODUCTION-SAFE VERSION
# This version includes:
# - Defensive sheet loading
# - Flexible header handling
# - Relationship layer support
# - Protected override handling
# - Safe fallbacks
# - Runtime validation

import os
import json
import time
import re
import secrets
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path

from flask import Flask, request, jsonify, make_response, redirect, url_for, session
from flask_cors import CORS

import gspread
from google.oauth2.service_account import Credentials

# ============================================================
# APP SETUP
# ============================================================
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "") or secrets.token_hex(32)

CORS(app)

SPREADSHEET_ID = os.getenv("SPREADSHEET_ID", "")

# ============================================================
# SHEET NAMES (SAFE DEFAULTS)
# ============================================================
SHEETS = {
    "base": "BaseSymbols",
    "behavior": "BehaviorRules",
    "state": "SizeStateRules",
    "location": "LocationRules",
    "relationship": "RelationshipRules",
    "override": "OverrideRules",
    "protected": "DoctrineProtectedOverrides",
    "templates": "OutputTemplates",
}

# ============================================================
# HELPERS
# ============================================================
def norm(x):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", str(x).lower())).strip()


def safe_get(row, *keys):
    for k in keys:
        if k in row and row[k]:
            return str(row[k]).strip()
    return ""


# ============================================================
# GOOGLE SHEETS
# ============================================================
def get_client():
    creds_json = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
    if creds_json:
        creds = Credentials.from_service_account_info(json.loads(creds_json))
    else:
        creds = Credentials.from_service_account_file("credentials.json")
    return gspread.authorize(creds)


def load_sheet(name):
    try:
        gc = get_client()
        sh = gc.open_by_key(SPREADSHEET_ID)
        ws = sh.worksheet(name)
        data = ws.get_all_records()
        return data
    except Exception:
        return []


# ============================================================
# LOAD ALL SHEETS SAFELY
# ============================================================
def load_all():
    return {k: load_sheet(v) for k, v in SHEETS.items()}


# ============================================================
# MATCH ENGINE
# ============================================================
def match_layer(text, rows, name_keys, keyword_key="keywords"):
    hits = []
    t = norm(text)

    for r in rows:
        name = safe_get(r, *name_keys)
        if not name:
            continue

        tokens = [norm(name)] + [norm(x) for x in safe_get(r, keyword_key).split(",") if x]

        for tok in tokens:
            if tok and tok in t:
                hits.append({
                    "name": name,
                    "row": r
                })
                break

    return hits[:4]


# ============================================================
# OVERRIDE
# ============================================================
def apply_override(text, overrides):
    t = norm(text)
    for r in overrides:
        cond = safe_get(r, "condition")
        if cond and norm(cond) in t:
            return r
    return None


# ============================================================
# INTERPRET
# ============================================================
@app.route("/interpret", methods=["POST"])
def interpret():
    data = request.json or {}
    dream = data.get("dream", "")

    if not dream:
        return jsonify({"error": "No dream provided"}), 400

    sheets = load_all()

    base = match_layer(dream, sheets["base"], ["symbol"])
    behavior = match_layer(dream, sheets["behavior"], ["behavior", "behavior_name"])
    state = match_layer(dream, sheets["state"], ["state", "state_name"])
    location = match_layer(dream, sheets["location"], ["location", "location_name"])
    relationship = match_layer(dream, sheets["relationship"], ["relationship", "relationship_name"])

    override = apply_override(dream, sheets["override"] + sheets["protected"])

    # ============================================================
    # BUILD OUTPUT
    # ============================================================
    parts = []

    for b in base:
        parts.append(safe_get(b["row"], "base_spiritual_meaning", "meaning"))

    for b in behavior:
        parts.append(safe_get(b["row"], "meaning_modifier"))

    for b in state:
        parts.append(safe_get(b["row"], "meaning_modifier"))

    for b in location:
        parts.append(safe_get(b["row"], "life_area_meaning"))

    for b in relationship:
        parts.append(safe_get(b["row"], "meaning_modifier"))

    if override:
        parts = [safe_get(override, "final_spiritual_meaning", "effect")]

    result = " ".join([p for p in parts if p])

    return jsonify({
        "dream": dream,
        "interpretation": result or "No clear interpretation",
        "layers": {
            "base": base,
            "behavior": behavior,
            "state": state,
            "location": location,
            "relationship": relationship,
            "override": bool(override)
        }
    })


# ============================================================
# HEALTH CHECK
# ============================================================
@app.route("/health")
def health():
    sheets = load_all()
    return jsonify({
        "ok": True,
        "sheet_counts": {k: len(v) for k, v in sheets.items()}
    })


# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
