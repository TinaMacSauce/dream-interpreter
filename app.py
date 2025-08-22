# app.py
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os, re, time, logging
import gspread
import pandas as pd

# ===== Flask setup =====
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024  # 2MB
logging.basicConfig(level=logging.INFO)

# ===== Config (env overrides allowed) =====
SHEET_KEY = os.getenv("SHEET_KEY", "1ToWPpJ_u-Z14eqL9U1oJppXX6f63h_eZcjUVglKr4Zk")
WORKSHEET_NAME = os.getenv("WORKSHEET_NAME", "Sheet1")
GCRED_PATH = os.getenv("GSPREAD_CREDENTIALS", "/etc/secrets/credentials.json")

ALLOWED_ORIGINS = [
    o.strip() for o in os.getenv(
        "ALLOWED_ORIGINS",
        "https://jamaicantruestories.com,https://www.jamaicantruestories.com"
    ).split(",") if o.strip()
]

CORS(app, resources={r"/*": {"origins": ALLOWED_ORIGINS}}, supports_credentials=False)

# ===== Text utils =====
WORD_RE = re.compile(r"[a-z0-9']+")

def normalize(text: str) -> str:
    """Lowercase text and strip punctuation; keep words/numbers only."""
    if not isinstance(text, str):
        return ""
    return " ".join(WORD_RE.findall(text.lower()))

def split_keywords(s: str):
    """Split comma/semicolon-separated keywords into a clean list."""
    if not isinstance(s, str):
        return []
    parts = re.split(r"[,;]+", s)
    return [p.strip().lower() for p in parts if p.strip()]

# ===== Google Sheet loading with cache =====
_df_cache = None
_df_cache_time = 0
CACHE_TTL = int(os.getenv("SHEET_CACHE_TTL_SEC", "300"))  # seconds

def load_sheet(force: bool = False) -> pd.DataFrame:
    """Load sheet into a DataFrame; normalize headers; prepare helpers."""
    global _df_cache, _df_cache_time

    if _df_cache is not None and not force and (time.time() - _df_cache_time) < CACHE_TTL:
        return _df_cache

    gc = gspread.service_account(filename=GCRED_PATH)
    ws = gc.open_by_key(SHEET_KEY).worksheet(WORKSHEET_NAME)
    rows = ws.get_all_records()  # requires header row
    df = pd.DataFrame(rows)

    # Normalize column names: lowercase + strip spaces
    df.rename(columns={c: c.strip().lower() for c in df.columns}, inplace=True)

    # Ensure required columns exist
    for col in ("input", "output", "keywords"):
        if col not in df.columns:
            df[col] = ""

    # Precompute normalized fields
    df["__input_norm"] = df["input"].apply(normalize)
    df["__kw_list"] = df["keywords"].apply(split_keywords)

    # Fallback keywords from input if keywords column is empty for a row
    df["__kw_list"] = df.apply(
        lambda r: r["__kw_list"] if r["__kw_list"] else
        [w for w in r["__input_norm"].split() if len(w) > 2],
        axis=1
    )

    _df_cache, _df_cache_time = df, time.time()
    app.logger.info("Loaded sheet: %d rows; columns=%s", len(df), list(df.columns))
    return df

# ===== Matching logic =====
def find_all_matches(user_text: str, limit: int = 50):
    df = load_sheet()
    t = normalize(user_text)
    t_words = set(t.split())

    matches = []
    for _, r in df.iterrows():
        hit_words = []

        for k in r["__kw_list"]:
            # Allow multi-word phrases (e.g., "high place") and single words (fast set lookup)
            if " " in k:
                if re.search(rf"\b{re.escape(k)}\b", t):
                    hit_words.append(k)
            else:
                if k in t_words:
                    hit_words.append(k)

        if hit_words:
            score = sum(len(hw.split()) for hw in hit_words) + 0.1 * len(hit_words)
            matches.append({
                "symbol": (r["input"] or "").strip(),
                "output": (r["output"] or "").strip(),
                "why": sorted(set(hit_words)),
                "score": score
            })

    matches.sort(key=lambda m: m["score"], reverse=True)
    return matches[:limit]

# ===== Routes =====
@app.route("/")
def home():
    # Optional: serve your index.html if you have it in /templates
    try:
        return render_template("index.html")
    except Exception:
        return "Dream Interpreter API is running.", 200

@app.route("/interpret", methods=["POST"])
def interpret():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"ok": False, "error": "No text provided"}), 400

    results = find_all_matches(text)

    if results:
        blocks = []
        for m in results:
            because = f"(Matched on: {', '.join(m['why'])})" if m["why"] else ""
            block = f"{m['output']}\n\n{because}".strip()
            blocks.append(block)
        combined_output = "\n\n— — —\n\n".join(blocks)
        match_label = ", ".join([m["symbol"] for m in results if m["symbol"]])
    else:
        combined_output = "No direct matches yet. Try simpler words, or add new keywords to the Sheet."
        match_label = "No match"

    return jsonify({
        "ok": True,
        "text": text,
        "matches": results,
        "combined_output": combined_output,
        "match_label": match_label
    })

@app.route("/admin/reload", methods=["POST"])
def reload_sheet():
    load_sheet(force=True)
    return jsonify({"ok": True, "reloaded": True})

@app.route("/debug/keywords")
def debug_keywords():
    # Force reload so you always see fresh data
    df = load_sheet(force=True)
    try:
        preview = df[["input", "keywords"]].head(50).to_dict(orient="records")
    except Exception:
        # if 'keywords' column has a funky header, show all columns to diagnose
        preview = df.head(10).to_dict(orient="records")
    return jsonify(preview)

@app.route("/healthz")
def healthz():
    return "ok", 200

# ===== Entrypoint =====
if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))  # Render sets PORT
    app.run(host="0.0.0.0", port=port)