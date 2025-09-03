# app.py
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os, re, unicodedata
import gspread
import pandas as pd

app = Flask(__name__)

# ===== CORS =====
ALLOWED = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",") if o.strip()]
CORS(app, resources={r"/*": {"origins": ALLOWED}}, supports_credentials=False)

# ===== ENV =====
SHEET_KEY = os.getenv("SHEET_KEY", "").strip()
WORKSHEET_NAME = os.getenv("WORKSHEET_NAME", "Sheet1").strip()
GCRED_PATH = os.getenv("GSPREAD_CREDENTIALS", "/etc/secrets/credentials.json").strip()

# ---------- helpers ----------
def norm(txt: str) -> str:
    if not isinstance(txt, str): return ""
    t = unicodedata.normalize("NFKD", txt).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", t).strip().lower()

def tokenize_words(txt: str):
    return re.findall(r"\b[a-z0-9][a-z0-9\-']+\b", norm(txt))

def split_keywords(cell: str):
    raw = [k.strip() for k in (cell or "").split(",") if k.strip()]
    phrases, words = [], []
    for k in raw:
        if " " in k:
            phrases.append(norm(k))
        else:
            words.append(norm(k))
    return phrases, words

def score_row(dream_norm: str, dream_words: set, phrases, words):
    score = 0
    for ph in phrases:  # phrase hits = 2 pts
        if re.search(rf"(?<!\w){re.escape(ph)}(?!\w)", dream_norm):
            score += 2
    for w in words:     # word hits = 1 pt
        if w in dream_words:
            score += 1
    return score

def find_col(df: pd.DataFrame, candidates):
    cols = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in cols:
            return cols[c.lower()]
    return None

# ---------- data ----------
def load_sheet():
    if not SHEET_KEY:
        raise RuntimeError("Missing SHEET_KEY env var.")
    if not os.path.exists(GCRED_PATH):
        raise RuntimeError(f"Google credentials not found at {GCRED_PATH}")

    gc = gspread.service_account(filename=GCRED_PATH)
    sh = gc.open_by_key(SHEET_KEY)
    ws = sh.worksheet(WORKSHEET_NAME)
    data = ws.get_all_records()
    df = pd.DataFrame(data)

    # map your exact headers (case-insensitive)
    col_input = find_col(df, ["Input"])
    col_output = find_col(df, ["output"])
    col_keywords = find_col(df, ["keywords"])
    if not all([col_input, col_output, col_keywords]):
        raise ValueError("Sheet must include columns: Input, output, keywords")

    # optional Priority support
    col_priority = find_col(df, ["Priority"]) or None
    df["_input_col"] = col_input
    df["_output_col"] = col_output
    df["_keywords_col"] = col_keywords
    df["_priority_col"] = col_priority
    if col_priority is None:
        df["Priority"] = 0
        df["_priority_col"] = "Priority"
    return df

DF_CACHE = {"df": None}
def get_df():
    if DF_CACHE["df"] is None:
        DF_CACHE["df"] = load_sheet()
    return DF_CACHE["df"]

# ---------- health / utility ----------
@app.route("/healthz", methods=["GET"])
def healthz():
    # Do NOT touch Google APIs here; keep it instant for Render.
    return "OK", 200

@app.route("/warmup", methods=["POST", "GET"])
def warmup():
    """Optionally preload the sheet cache after deploy."""
    try:
        DF_CACHE["df"] = load_sheet()
        return jsonify({"ok": True, "message": "Cache warmed."})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/refresh", methods=["POST"])
def refresh():
    try:
        DF_CACHE["df"] = load_sheet()
        return jsonify({"ok": True, "message": "Dictionary reloaded."})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

# ---------- main API ----------
@app.route("/interpret", methods=["POST"])
def interpret():
    payload = request.get_json(force=True) or {}
    dream_text = payload.get("text", "")
    top_k = int(payload.get("top_k", 25))
    if not dream_text.strip():
        return jsonify({"ok": False, "error": "Missing 'text'"}), 400

    dream_norm = norm(dream_text)
    dream_words = set(tokenize_words(dream_text))

    try:
        df = get_df()
    except Exception as e:
        return jsonify({"ok": False, "error": f"Dictionary not loaded: {e}"}), 500

    inp_col = df["_input_col"].iat[0]
    out_col = df["_output_col"].iat[0]
    kw_col = df["_keywords_col"].iat[0]
    pr_col = df["_priority_col"].iat[0]

    scored = []
    for _, row in df.iterrows():
        phrases, words = split_keywords(str(row.get(kw_col, "")))
        s = score_row(dream_norm, dream_words, phrases, words)
        if s > 0:
            scored.append({
                "score": s,
                "priority": int(row.get(pr_col, 0) or 0),
                "input": str(row.get(inp_col, "")).strip(),
                "output": str(row.get(out_col, "")).strip()
            })

    scored.sort(key=lambda r: (r["score"], r["priority"], r["input"].lower()), reverse=True)

    seen, final = set(), []
    for r in scored:
        if r["input"] not in seen:
            seen.add(r["input"])
            final.append({"Input (symbol)": r["input"], "Output (interpretations)": r["output"]})
        if len(final) >= top_k:
            break

    return jsonify({"ok": True, "query": dream_text, "results": final})

# ---------- site ----------
@app.route("/", methods=["GET"])
def home():
    # Optional template; health check never hits this.
    try:
        return render_template("index.html")
    except Exception:
        return "Dream Interpreter API is running.", 200

# ---------- run ----------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)