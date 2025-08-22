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

# ===== Config =====
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
    if not isinstance(text, str):
        return ""
    return " ".join(WORD_RE.findall(text.lower()))

def split_keywords(s: str):
    if not isinstance(s, str):
        return []
    parts = re.split(r"[,;]+", s)
    return [p.strip().lower() for p in parts if p.strip()]

# ===== Google Sheet loading with cache =====
_df_cache = None
_df_cache_time = 0
CACHE_TTL = int(os.getenv("SHEET_CACHE_TTL_SEC", "300"))

def load_sheet(force: bool = False) -> pd.DataFrame:
    global _df_cache, _df_cache_time
    if _df_cache is not None and not force and (time.time() - _df_cache_time) < CACHE_TTL:
        return _df_cache

    gc = gspread.service_account(filename=GCRED_PATH)
    ws = gc.open_by_key(SHEET_KEY).worksheet(WORKSHEET_NAME)
    rows = ws.get_all_records()
    df = pd.DataFrame(rows)

    # Normalize headers
    df.rename(columns={c: c.strip().lower() for c in df.columns}, inplace=True)
    for col in ("input", "output", "keywords"):
        if col not in df.columns:
            df[col] = ""

    # Precompute helpers
    df["__input_norm"] = df["input"].apply(normalize)
    df["__kw_list"] = df["keywords"].apply(split_keywords)
    df["__kw_list"] = df.apply(
        lambda r: r["__kw_list"] if r["__kw_list"] else
        [w for w in r["__input_norm"].split() if len(w) > 2],
        axis=1
    )

    _df_cache, _df_cache_time = df, time.time()
    app.logger.info("Loaded sheet: %d rows; columns=%s", len(df), list(df.columns))
    return df

# ===== Matching (phrase-first, filter generic singles) =====
def find_all_matches(user_text: str, limit: int = 50):
    df = load_sheet()
    t = normalize(user_text)
    t_words = set(t.split())

    # Single words we consider too generic to match by themselves
    GENERIC_STOP = {
        "high","low","old","new","alone","group","with","without","on","in","at",
        "someone","stranger","known","person","people","being","watched","and",
        "up","down","left","right","over","under","above","below"
    }

    matches = []
    for _, r in df.iterrows():
        hit_multi = []   # matched multi-word phrases (preferred)
        hit_single = []  # matched specific single words

        for k in r["__kw_list"]:
            k = (k or "").strip().lower()
            if not k:
                continue

            if " " in k:  # phrase like "high place"
                if re.search(rf"\b{re.escape(k)}\b", t):
                    hit_multi.append(k)
            else:
                # Accept single-word match only if it's specific enough
                if (k in t_words) and (len(k) >= 5) and (k not in GENERIC_STOP):
                    hit_single.append(k)

        # Keep rows that have phrase hits OR specific single-word hits
        if hit_multi or hit_single:
            why = sorted(set(hit_multi + hit_single))
            score = sum(len(w.split()) for w in hit_multi) * 2.0 + 0.5 * len(hit_single)
            matches.append({
                "symbol": (r["input"] or "").strip(),
                "output": (r["output"] or "").strip(),
                "why": why,
                "score": score
            })

    matches.sort(key=lambda m: m["score"], reverse=True)
    return matches[:limit]

# ===== Routes =====
@app.route("/")
def home():
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

    # Force-refresh to avoid stale caches across instances on Render
    load_sheet(force=True)

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
    df = load_sheet(force=True)
    try:
        preview = df[["input", "keywords"]].head(50).to_dict(orient="records")
    except Exception:
        preview = df.head(10).to_dict(orient="records")
    return jsonify(preview)

@app.route("/debug/test")
def debug_test():
    q = request.args.get("text", "")
    load_sheet(force=True)
    results = find_all_matches(q)
    return jsonify({"text_norm": normalize(q), "hits": results})

@app.route("/healthz")
def healthz():
    return "ok", 200

# ===== Entrypoint =====
if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)