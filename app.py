# app.py
from flask import Flask, request, jsonify, render_template
import os, re, difflib
import gspread
import pandas as pd

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024  # 2MB

# === CONFIG ===
SHEET_KEY = "1ToWPpJ_u-Z14eqL9U1oJppXX6f63h_eZcjUVglKr4Zk"
WORKSHEET_NAME = os.getenv("WORKSHEET_NAME", "Sheet11")  # <-- set to your tab name
GCRED_PATH = os.getenv("GSPREAD_CREDENTIALS", "/etc/secrets/credentials.json")

# === CACHED OBJECTS ===
_gc = None
_ws = None
_df = None

# --- helpers ---
def _norm(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip().lower()
    # collapse whitespace + remove extra punctuation spacing
    s = re.sub(r"\s+", " ", s)
    return s

def get_ws():
    global _gc, _ws
    if _gc is None:
        _gc = gspread.service_account(filename=GCRED_PATH)
    if _ws is None:
        # Use explicit worksheet by name to avoid reading the wrong tab
        _ws = _gc.open_by_key(SHEET_KEY).worksheet(WORKSHEET_NAME)
    return _ws

def get_df(force_refresh: bool = False):
    """Read the sheet, normalize headers & create helper columns."""
    global _df
    if force_refresh or _df is None:
        ws = get_ws()
        rows = ws.get_all_records()  # list[dict]
        df = pd.DataFrame(rows)

        if df.empty:
            # keep a tiny df so header checks don't crash
            df = pd.DataFrame(columns=["input", "output"])

        # normalize headers
        df.columns = [str(c).strip().lower() for c in df.columns]

        # required columns check
        cols = set(df.columns)
        if "input" not in cols and "symbol" not in cols:
            # leave df as-is; /interpret will return a clear error
            _df = df
            return _df
        if "output" not in cols:
            _df = df
            return _df

        # create working columns
        cand_col = "input" if "input" in cols else "symbol"
        df["_cand"] = df[cand_col].astype(str).map(_norm)

        if "keywords" in cols:
            # explode keywords into list of normalized strings
            def parse_kw(v):
                if pd.isna(v) or v is None:
                    return []
                return [_norm(p) for p in str(v).split(";") if _norm(p)]
            df["_kw"] = df["keywords"].map(parse_kw)
        else:
            df["_kw"] = [[] for _ in range(len(df))]

        df["_output_txt"] = df["output"].astype(str)

        _df = df
    return _df

def similarity(query: str, candidate: str) -> float:
    """Blend token overlap with difflib for robust partial matches."""
    q = set(_norm(query).split())
    c = set(_norm(candidate).split())
    if not q or not c:
        base = 0.0
    else:
        jaccard = len(q & c) / len(q | c)
        base = jaccard
    fuzzy = difflib.SequenceMatcher(None, _norm(query), _norm(candidate)).ratio()
    # weight token overlap a bit higher for phrase searches
    return 0.6 * base + 0.4 * fuzzy

def best_match(user_text: str, df: pd.DataFrame):
    """Return (row, score, matched_string) or (None, 0, None)."""
    if df.empty:
        return None, 0.0, None
    # choose candidate pool = input/symbol + keywords
    pool = []
    for idx, row in df.iterrows():
        pool.append((idx, row["_cand"]))
        for kw in row["_kw"]:
            pool.append((idx, kw))

    # score each unique candidate
    q = _norm(user_text)
    best = (-1, 0.0, None)
    seen = {}
    for idx, cand in pool:
        if not cand:
            continue
        if cand in seen:
            score = seen[cand]
        else:
            score = similarity(q, cand)
            seen[cand] = score
        if score > best[1]:
            best = (idx, score, cand)

    if best[0] == -1:
        return None, 0.0, None
    return df.iloc[best[0]], best[1], best[2]

# --- routes ---
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/healthz")
def healthz():
    try:
        ws = get_ws()
        _ = ws.get("A1")
        return "ok", 200
    except Exception as e:
        return f"degraded: {type(e).__name__}: {e}", 200

@app.route("/refresh", methods=["POST", "GET"])
def refresh():
    try:
        get_df(force_refresh=True)
        return jsonify({"status": "refreshed"}), 200
    except Exception as e:
        return jsonify({"error": f"Refresh failed: {e}"}), 500

@app.route("/interpret", methods=["POST"])
def interpret():
    payload = request.get_json(silent=True) or {}
    user_text = payload.get("text") if request.is_json else request.form.get("text")
    if not user_text or not _norm(user_text):
        return jsonify({"error": "Provide dream text in 'text'"}), 400

    df = get_df()
    cols = list(df.columns)

    # check required columns exist in the *dataframe*, not just empty df
    if ("input" not in cols and "symbol" not in cols) or ("output" not in cols):
        return jsonify({
            "error": "Sheet must contain columns 'input' (or 'symbol') and 'output'.",
            "columns_found": cols
        }), 500

    row, score, matched = best_match(user_text, df)
    if row is None or score < 0.28:  # tolerant threshold; tweak as you like
        return jsonify({
            "match": None,
            "score": round(float(score), 3),
            "interpretation": (
                "I couldn't find a close match yet. Try a simpler phrase "
                "(e.g., 'teeth falling out', 'running late', 'snake bite')."
            )
        }), 200

    interpretation = str(row.get("_output_txt", "")).strip() or "No interpretation found yet."
    # also report which input row we matched
    canonical = str(row.get("_cand", "")).strip()

    return jsonify({
        "match": canonical,
        "score": round(float(score), 3),
        "interpretation": interpretation
    }), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")))
