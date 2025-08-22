# app.py
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os, re, time
import gspread
import pandas as pd

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024  # 2MB

# === CONFIG (same as your current) ===
SHEET_KEY = "1ToWPpJ_u-Z14eqL9U1oJppXX6f63h_eZcjUVglKr4Zk"
WORKSHEET_NAME = "Sheet1"
GCRED_PATH = os.getenv("GSPREAD_CREDENTIALS", "/etc/secrets/credentials.json")

ALLOWED_ORIGINS = [
    o.strip() for o in os.getenv(
        "ALLOWED_ORIGINS",
        (
            "https://jamaicantruestories.com,"
            "https://www.jamaicantruestories.com"
        )
    ).split(",")
    if o.strip()
]
CORS(app, resources={r"/*": {"origins": ALLOWED_ORIGINS}}, supports_credentials=False)

# ===== Utilities =====
WORD_RE = re.compile(r"[a-z0-9']+")

def normalize(text: str) -> str:
    """lowercase + strip punctuation; keep words & numbers only"""
    if not isinstance(text, str):
        return ""
    return " ".join(WORD_RE.findall(text.lower()))

def split_keywords(s: str):
    """split comma/semicolon separated keywords -> clean list"""
    if not isinstance(s, str):
        return []
    parts = re.split(r"[,;]+", s)
    return [p.strip().lower() for p in parts if p.strip()]

# ===== Sheet loading / caching =====
_df_cache = None
_df_cache_time = 0
CACHE_TTL = int(os.getenv("SHEET_CACHE_TTL_SEC", "300"))  # 5 min

def load_sheet(force=False) -> pd.DataFrame:
    global _df_cache, _df_cache_time
    if _df_cache is not None and not force and (time.time() - _df_cache_time) < CACHE_TTL:
        return _df_cache

    gc = gspread.service_account(filename=GCRED_PATH)
    ws = gc.open_by_key(SHEET_KEY).worksheet(WORKSHEET_NAME)
    rows = ws.get_all_records()  # requires headers in row 1
    df = pd.DataFrame(rows)

    # Ensure required columns exist
    for col in ["input", "output", "keywords"]:
        if col not in df.columns:
            df[col] = ""

    # Precompute normalized helpers
    df["__input_norm"] = df["input"].apply(normalize)
    df["__kw_list"] = df["keywords"].apply(split_keywords)

    # Fallback: if no keywords provided, derive some from input phrase
    df["__kw_list"] = df.apply(
        lambda r: r["__kw_list"] if len(r["__kw_list"]) > 0
        else [w for w in r["__input_norm"].split() if len(w) > 2],
        axis=1
    )

    _df_cache, _df_cache_time = df, time.time()
    return df

# ===== Matching =====
def find_all_matches(user_text: str, limit:int=50):
    df = load_sheet()
    t = normalize(user_text)

    matches = []
    t_words = set(t.split())

    for _, r in df.iterrows():
        hit_words = []
        for k in r["__kw_list"]:
            # word-boundary match (so 'naked.' and 'naked' both hit via normalization)
            # also allow 2-word phrases like "high place"
            if " " in k:
                if re.search(rf"\b{re.escape(k)}\b", t):
                    hit_words.append(k)
            else:
                if k in t_words:
                    hit_words.append(k)

        if hit_words:
            # Simple score: longer phrases weigh more; more hits -> higher score
            score = sum(len(hw.split()) for hw in hit_words) + 0.1 * len(hit_words)
            matches.append({
                "symbol": (r["input"] or "").strip(),
                "why": sorted(list(set(hit_words))),
                "output": (r["output"] or "").strip(),
                "score": score
            })

    # Sort strongest first but return all (up to limit)
    matches.sort(key=lambda m: m["score"], reverse=True)
    return matches[:limit]

# ===== Routes =====
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/interpret", methods=["POST"])
def interpret():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"ok": False, "error": "No text provided"}), 400

    results = find_all_matches(text)

    # Combine outputs (your rows are already formatted with Spiritual Meaning / Effects / What to Do)
    combined_output = ""
    if results:
        blocks = []
        for m in results:
            header = m["symbol"] if m["symbol"] else "Match"
            why = ", ".join(m["why"]) if m["why"] else ""
            block = f"{m['output']}\n\n(Matched on: {why})"
            blocks.append(block)
        combined_output = "\n\n— — —\n\n".join(blocks)
    else:
        combined_output = (
            "No direct matches yet. Try simpler words, or add new keywords to the Sheet."
        )

    return jsonify({
        "ok": True,
        "text": text,
        "matches": results,        # list of {symbol, why, output, score}
        "combined_output": combined_output
    })

@app.route("/admin/reload", methods=["POST"])
def reload_sheet():
    """Optional: call this after you edit the Google Sheet to bust cache immediately."""
    load_sheet(force=True)
    return jsonify({"ok": True, "reloaded": True})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")))
