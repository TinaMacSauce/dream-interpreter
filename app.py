# app.py
import os
import re
import time
import difflib
from typing import List, Tuple, Optional

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import gspread
import pandas as pd

app = Flask(__name__)

# --- CORS (use exact origins from ALLOWED_ORIGINS) ---
_allowed = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()]
if _allowed:
    CORS(app, resources={r"/*": {"origins": _allowed}})
else:
    # fallback to allow same-origin/local testing
    CORS(app)

# --- ENV CONFIG ---
SHEET_KEY = os.getenv("SHEET_KEY")                         # REQUIRED
WORKSHEET_NAME = os.getenv("WORKSHEET_NAME", "Sheet1")
GCRED_PATH = os.getenv("GSPREAD_CREDENTIALS", "/etc/secrets/credentials.json")

# --- GLOBALS (simple in-memory cache) ---
DF: Optional[pd.DataFrame] = None
LAST_LOAD = 0.0
RELOAD_SEC = 60 * 5  # reload sheet at most every 5 minutes on demand


def normalize_text(s: str) -> str:
    """Lowercase and collapse whitespace/punctuation to help matching."""
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s,/-]+", " ", s)   # keep basic word chars
    s = re.sub(r"\s+", " ", s).strip()
    return s


def load_sheet(force: bool = False) -> Tuple[bool, str]:
    """Load Google Sheet into global DF with normalized helper columns."""
    global DF, LAST_LOAD

    # throttle reloads unless forced
    if not force and DF is not None and (time.time() - LAST_LOAD) < RELOAD_SEC:
        return True, "cached"

    if not SHEET_KEY:
        DF = None
        return False, "Missing SHEET_KEY"

    if not os.path.exists(GCRED_PATH):
        DF = None
        return False, f"Credentials file not found at {GCRED_PATH}"

    try:
        sa = gspread.service_account(filename=GCRED_PATH)
        sh = sa.open_by_key(SHEET_KEY)
        ws = sh.worksheet(WORKSHEET_NAME)
        records = ws.get_all_records()  # list[dict]
        if not records:
            DF = None
            return False, "Sheet is empty"

        df = pd.DataFrame(records)
        # Expect columns: input, output, (optional) keywords
        # Make sure required columns exist
        if "input" not in df.columns or "output" not in df.columns:
            DF = None
            return False, "Sheet must have columns: input, output (and optional keywords)"

        # Normalize and add helper columns
        df["_input_norm"] = df["input"].astype(str).map(normalize_text)

        if "keywords" in df.columns:
            # normalize keywords to list[str]
            def split_kw(val: str) -> List[str]:
                parts = re.split(r"[,\|;/]", str(val))
                return [normalize_text(p) for p in parts if normalize_text(p)]
            df["_kw_list"] = df["keywords"].apply(split_kw)
        else:
            df["_kw_list"] = [[] for _ in range(len(df))]

        DF = df
        LAST_LOAD = time.time()
        return True, "loaded"
    except Exception as e:
        DF = None
        return False, f"Error loading sheet: {e}"


def score_by_keywords(text_norm: str, kw_list: List[str]) -> int:
    """Simple keyword count score."""
    score = 0
    for kw in kw_list:
        # match as substring of normalized text
        if kw and kw in text_norm:
            score += 1
    return score


def find_best_match(user_input: str) -> Tuple[Optional[str], dict]:
    """
    Try: exact match -> fuzzy match -> keyword score.
    Returns (output_text, debug_info)
    """
    ok, msg = load_sheet(force=False)
    if not ok or DF is None:
        return None, {"status": "sheet_unavailable", "detail": msg}

    t = normalize_text(user_input)
    inputs = DF["_input_norm"].tolist()

    # 1) Exact match
    if t in inputs:
        idx = inputs.index(t)
        return DF.iloc[idx]["output"], {"method": "exact", "matches": 1}

    # 2) Fuzzy match on 'input' column
    close = difflib.get_close_matches(t, inputs, n=1, cutoff=0.82)  # fairly strict
    if close:
        idx = inputs.index(close[0])
        return DF.iloc[idx]["output"], {"method": "fuzzy", "similar_to": close[0]}

    # 3) Keyword scoring (from 'keywords' column)
    scores = []
    for i, row in DF.iterrows():
        kw_score = score_by_keywords(t, row["_kw_list"])
        if kw_score > 0:
            scores.append((kw_score, i))
    if scores:
        scores.sort(reverse=True)  # highest score first
        _, best_idx = scores[0]
        return DF.iloc[best_idx]["output"], {
            "method": "keywords",
            "score": scores[0][0],
            "candidates": len(scores),
        }

    return None, {"method": "none", "matches": 0}


# ---------- ROUTES ----------
@app.route("/")
def index():
    # If you have a template, this will serve it. Otherwise you can remove this.
    try:
        return render_template("index.html")
    except Exception:
        return "Dream Interpreter API", 200


@app.route("/interpret", methods=["POST"])
def interpret():
    """
    JSON body: { "dream": "text the user typed" }
    Returns: { interpretation: str, meta: {...} }
    """
    data = request.get_json(silent=True) or {}
    dream_text = (data.get("dream") or "").strip()
    if not dream_text:
        return jsonify({"interpretation": "Please type a dream.", "meta": {"error": "empty"}}), 400

    result, meta = find_best_match(dream_text)
    if result:
        return jsonify({"interpretation": result, "meta": meta}), 200
    else:
        return jsonify({"interpretation": "No response", "meta": meta}), 200


@app.route("/reload", methods=["POST", "GET"])
def reload_sheet():
    """Manual reload endpoint in case you update the sheet and want instant refresh."""
    ok, detail = load_sheet(force=True)
    code = 200 if ok else 500
    return jsonify({"ok": ok, "detail": detail}), code


@app.route("/healthz")
def healthz():
    missing = []
    if not SHEET_KEY:
        missing.append("SHEET_KEY")
    if not os.path.exists(GCRED_PATH):
        missing.append(f"credentials@{GCRED_PATH}")

    status = "OK" if not missing else "Missing: " + ", ".join(missing)
    info = {"status": status, "loaded": DF is not None}
    return jsonify(info), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Render assigns this
    app.run(host="0.0.0.0", port=port)