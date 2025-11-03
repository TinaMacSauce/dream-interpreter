# app.py — paragraph-ready, no pandas, safe wording, simple free-usage counter
import os, re, json, logging, time
from typing import Dict, List, Tuple
from flask import Flask, request, jsonify
from flask_cors import CORS
import gspread
from google.oauth2.service_account import Credentials

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ── Env ─────────────────────────────────────────────────────────────
SHEET_ID       = os.environ.get("SHEET_ID", "").strip()              # required
WORKSHEET_NAME = os.environ.get("WORKSHEET_NAME", "Sheet1").strip()  # your tab name
CREDS_PATH     = os.environ.get("CREDS_PATH", "credentials.json").strip()

# Free tier (simple local counter; for production use a DB)
FREE_QUOTA     = int(os.environ.get("FREE_QUOTA", "20"))
COUNTS_FILE    = os.environ.get("COUNTS_FILE", "usage_counts.json")

DISCLAIMER = ("Reflective and educational guidance based on cultural symbolism—"
              "no divination, no predictions, and no medical or legal advice.")

NOTE_TEXT = "Based on traditional symbolism, here is a reflective interpretation:"

# ── Globals (in-memory cache) ───────────────────────────────────────
LUT: Dict[str, str] = {}   # 'input' -> 'output'
INPUTS: List[str] = []     # list of input phrases (lowercased)

# ── Google Sheets helpers ───────────────────────────────────────────
def _gs_client():
    scopes = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
    creds  = Credentials.from_service_account_file(CREDS_PATH, scopes=scopes)
    return gspread.authorize(creds)

def load_sheet() -> Tuple[List[str], Dict[str, str]]:
    """Read columns A: input, B: output from WORKSHEET_NAME."""
    if not SHEET_ID:
        raise RuntimeError("Missing SHEET_ID environment variable.")
    gc = _gs_client()
    ws = gc.open_by_key(SHEET_ID).worksheet(WORKSHEET_NAME)
    rows = ws.get_all_records()  # header row used automatically
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

# ── Simple usage counter (local file; replace with DB later) ────────
def _load_counts() -> Dict[str, int]:
    try:
        with open(COUNTS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def _save_counts(counts: Dict[str, int]) -> None:
    try:
        with open(COUNTS_FILE, "w", encoding="utf-8") as f:
            json.dump(counts, f)
    except Exception:
        pass

def get_identifier(email: str | None) -> str:
    if email:
        return email.strip().lower()
    # fallback to IP for anonymous users
    return f"ip:{request.remote_addr}"

def get_count(identifier: str) -> int:
    return _load_counts().get(identifier, 0)

def bump_count(identifier: str) -> int:
    counts = _load_counts()
    counts[identifier] = counts.get(identifier, 0) + 1
    _save_counts(counts)
    return counts[identifier]

# ── Matching (paragraph → symbols) ──────────────────────────────────
WORD = re.compile(r"[a-z']+")

def best_matches(paragraph: str, max_hits: int = 8) -> List[str]:
    """
    Paragraph decoding:
      1) Direct phrase match (fast): keep any sheet inputs that appear in the text.
      2) If none, token-overlap fallback for rough matching.
    """
    text = paragraph.lower().strip()
    if not text:
        return []

    # 1) direct phrase hits
    found = [k for k in INPUTS if k and k in text]

    # de-duplicate while keeping order
    seen, ordered = set(), []
    for k in found:
        if k not in seen:
            ordered.append(k); seen.add(k)

    if ordered:
        return ordered[:max_hits]

    # 2) token overlap fallback (very light)
    toks = set(WORD.findall(text))
    scored = []
    for k in INPUTS:
        ktoks = set(WORD.findall(k))
        score = len(toks & ktoks)
        if score:
            scored.append((score, k))
    scored.sort(reverse=True)
    return [k for _, k in scored[:max_hits]]

def compose_output(keys: List[str]) -> Dict[str, object]:
    if not keys:
        return {
            "note": NOTE_TEXT,
            "symbols_matched": [],
            "interpretation": {
                "spiritual_meaning": "No specific entry matched. Clarity often grows as you record and review your dreams.",
                "effects_in_the_physical_realm": "Notice patterns and changes around you this week.",
                "what_to_do": "Use simple terms; reflect, repent, and read Psalm 91 before bed."
            },
            "disclaimer": DISCLAIMER
        }

    # Your sheet's 'output' already includes Spiritual Meaning / Effects / What to Do.
    merged = "\n\n".join([LUT[k] for k in keys if k in LUT])
    return {
        "note": NOTE_TEXT,
        "symbols_matched": keys,
        "interpretation": {
            "spiritual_meaning": merged,
            "effects_in_the_physical_realm": "",
            "what_to_do": ""
        },
        "disclaimer": DISCLAIMER
    }

# ── Flask app ───────────────────────────────────────────────────────
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
    text  = (data.get("text")  or "").strip()
    email = (data.get("email") or "").strip().lower() or None

    if not text:
        return jsonify({"error": "Please paste your dream text."}), 400

    # simple free-usage gating (local)
    ident = get_identifier(email)
    used  = get_count(ident)
    if used >= FREE_QUOTA:
        return jsonify({
            "paywall": True,
            "message": f"Free limit reached ({FREE_QUOTA}). Please subscribe to continue.",
            "free_uses_left": 0
        }), 402

    keys = best_matches(text)
    result = compose_output(keys)

    # bump usage after a successful interpretation
    used_now = bump_count(ident)
    result["free_uses_left"] = max(0, FREE_QUOTA - used_now)

    return jsonify(result), 200

# ── Entrypoint ──────────────────────────────────────────────────────
if __name__ == "__main__":
    refresh_cache()
    # Dev server; for production use: gunicorn app:app
    app.run(host="0.0.0.0", port=5000)
