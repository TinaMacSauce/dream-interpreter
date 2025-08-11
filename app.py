# app.py
from flask import Flask, request, jsonify
import os
import gspread
import pandas as pd
import difflib

app = Flask(__name__)

# ---- Google Sheets Auth (Render Secret File) -----------------
SHEET_KEY = "1ToWPpJ_u-Z14eqL9U1oJppXX6f63h_eZcjUVglKr4Zk"
GCRED_PATH = os.getenv("GSPREAD_CREDENTIALS", "/etc/secrets/credentials.json")

# Lazy-init globals so app can start even if Sheets is briefly unavailable
_gc = None
_ws = None
_df = None

def get_ws():
    """Return an authed worksheet object; re-auth if needed."""
    global _gc, _ws
    if _gc is None:
        _gc = gspread.service_account(filename=GCRED_PATH)
    if _ws is None:
        _ws = _gc.open_by_key(SHEET_KEY).sheet1
    return _ws

def get_df(force_refresh=False):
    """Return a pandas DataFrame of the sheet (cached)."""
    global _df
    if force_refresh or _df is None:
        ws = get_ws()
        data = ws.get_all_records()
        df = pd.DataFrame(data)
        # Normalize headers
        df.columns = [str(c).strip().lower() for c in df.columns]
        _df = df
    return _df

# ---- Routes --------------------------------------------------

@app.route("/")
def index():
    return (
        "<h1>Dream Interpreter</h1>"
        "<p>POST a JSON body to <code>/interpret</code> with {'text': 'your dream...'}</p>"
        "<p>Health: <a href='/healthz'>/healthz</a></p>"
    )

@app.route("/healthz")
def healthz():
    """
    Light, dependency-aware health check used by Render.
    - Confirms Google auth
    - Confirms the sheet is reachable
    - Confirms at least A1 is readable
    """
    try:
        ws = get_ws()
        _ = ws.get("A1")
        return "ok", 200
    except Exception as e:
        # 200 so Render doesn't keep restarting; body shows degradation reason.
        return f"degraded: {type(e).__name__}: {e}", 200

@app.route("/interpret", methods=["POST"])
def interpret():
    """
    Minimal interpreter:
    - Accepts JSON: {"text": "..."} or form "text"
    - Fuzzy-matches against columns named 'input' or 'symbol'
    - Returns matched row's 'output' if present, else a gentle fallback
    """
    user_text = (request.json or {}).get("text") if request.is_json else request.form.get("text")
    if not user_text:
        return jsonify({"error": "Provide dream text in 'text'"}), 400

    try:
        df = get_df()
    except Exception as e:
        return jsonify({"error": f"Sheet unavailable: {e}"}), 503

    # Identify columns
    cols = list(df.columns)
    candidates_col = "input" if "input" in cols else ("symbol" if "symbol" in cols else None)
    output_col = "output" if "output" in cols else None

    if not candidates_col or not output_col:
        return jsonify({
            "error": "Sheet must contain columns 'input' (or 'symbol') and 'output'.",
            "columns_found": cols
        }), 500

    # Fuzzy match
    candidate_values = df[candidates_col].astype(str).tolist()
    best = difflib.get_close_matches(user_text, candidate_values, n=1, cutoff=0.3)
    if not best:
        return jsonify({
            "match": None,
            "interpretation": (
                "I couldn't find a close match yet. Try a simpler phrase (e.g., 'teeth falling out', "
                "'running late', 'snake bite')."
            )
        }), 200

    best_val = best[0]
    row = df.loc[df[candidates_col].astype(str) == best_val].iloc[0]
    interpretation = str(row.get(output_col, "")).strip() or "No interpretation found yet."

    return jsonify({
        "match": best_val,
        "interpretation": interpretation
    }), 200

# Gunicorn entrypoint: app = Flask(__name__)
# (No __main__ needed for Render, but harmless if kept.)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")))
