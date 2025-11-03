# app.py — clean, no pandas, no transformers
import os, re, logging
from typing import Dict, List, Tuple
from flask import Flask, request, jsonify
from flask_cors import CORS
import gspread
from google.oauth2.service_account import Credentials

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ── Env ─────────────────────────────────────────────────────────────
SHEET_ID      = os.environ.get("SHEET_ID", "").strip()           # required
WORKSHEET_NAME= os.environ.get("WORKSHEET_NAME", "Sheet1")       # you wanted Sheet1
CREDENTIALS   = os.environ.get("CREDS_PATH", "credentials.json") # path to your JSON

# ── Globals (in-memory dictionary) ─────────────────────────────────
# LUT maps 'input' -> 'output'
LUT: Dict[str, str] = {}
INPUTS: List[str] = []

# ── Google Sheets helpers ─────────────────────────────────────────
def _gs_client():
    scopes = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
    creds  = Credentials.from_service_account_file(CREDENTIALS, scopes=scopes)
    return gspread.authorize(creds)

def load_sheet() -> Tuple[List[str], Dict[str, str]]:
    """
    Reads 2 columns from Sheet1:
      A: input (lowercased key)
      B: output (full interpretation text)
    Returns (inputs, lut)
    """
    if not SHEET_ID:
        raise RuntimeError("Missing SHEET_ID env var.")
    gc = _gs_client()
    ws = gc.open_by_key(SHEET_ID).worksheet(WORKSHEET_NAME)
    rows = ws.get_all_records()  # uses row1 as headers
    lut: Dict[str, str] = {}
    for r in rows:
        k = (r.get("input") or "").strip().lower()
        v = (r.get("output") or "").strip()
        if k:
            lut[k] = v
    inputs = list(lut.keys())
    logging.info("Loaded %d rows from %s", len(inputs), WORKSHEET_NAME)
    return inputs, lut

def refresh_cache():
    global INPUTS, LUT
    INPUTS, LUT = load_sheet()

# ── Matching (simple & fast) ───────────────────────────────────────
WORD = re.compile(r"[a-z']+")

def best_matches(paragraph: str, max_hits: int = 8) -> List[str]:
    """
    1) Direct phrase match if the sheet's input phrase appears in the text.
    2) If none, fall back to token overlap (very light fuzzy).
    """
    text = paragraph.lower().strip()
    if not text:
        return []
    # 1) direct phrase hits
    direct = [k for k in INPUTS if k and k in text]
    if direct:
        # remove duplicates preserving order
        seen, ordered = set(), []
        for k in direct:
            if k not in seen:
                ordered.append(k); seen.add(k)
        return ordered[:max_hits]

    # 2) token overlap fallback
    toks = set(WORD.findall(text))
    scored = []
    for k in INPUTS:
        ktoks = set(WORD.findall(k))
        score = len(toks & ktoks)
        if score:
            scored.append((score, k))
    scored.sort(reverse=True)
    return [k for _, k in scored[:max_hits]]

def compose_output(keys: List[str]) -> Dict[str, str]:
    DISCLAIMER = ("Reflective and educational guidance based on cultural symbolism—"
                  "no divination, predictions, medical or legal advice.")
    if not keys:
        return {
            "note": "Based on traditional symbolism, here is a reflective interpretation:",
            "symbols_matched": [],
            "interpretation": {
                "spiritual_meaning": "No specific entry matched. Clarity grows as you record and review dreams.",
                "effects_in_the_physical_realm": "Notice patterns and changes around you this week.",
                "what_to_do": "Use simple terms; reflect, repent, and read Psalm 91 before bed."
            },
            "disclaimer": DISCLAIMER
        }
    merged = "\n\n".join([LUT[k] for k in keys if k in LUT])
    return {
        "note": "Based on traditional symbolism, here is a reflective interpretation:",
        "symbols_matched": keys,
        "interpretation": {
            # Your 'output' already contains the 3-part text; we return it as one block.
            "spiritual_meaning": merged,
            "effects_in_the_physical_realm": "",
            "what_to_do": ""
        },
        "disclaimer": DISCLAIMER
    }

# ── Flask app ──────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

@app.get("/health")
def health():
    return {"ok": True, "loaded_rows": len(INPUTS), "worksheet": WORKSHEET_NAME}

@app.post("/refresh")
def refresh():
    refresh_cache()
    return {"reloaded": len(INPUTS)}

@app.post("/interpret")
def interpret():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "Please paste your dream text."}), 400
    keys = best_matches(text)
    return jsonify(compose_output(keys)), 200

if __name__ == "__main__":
    refresh_cache()
    app.run(host="0.0.0.0", port=5000)
