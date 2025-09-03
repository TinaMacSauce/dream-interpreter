# app.py
import os, re, time, difflib, logging
from typing import List, Tuple, Optional
from collections import defaultdict

import gspread
import pandas as pd
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# ───────────────────────── Optional Sentry ─────────────────────────
try:
    import sentry_sdk
    if os.getenv("SENTRY_DSN"):
        sentry_sdk.init(dsn=os.getenv("SENTRY_DSN"), traces_sample_rate=0.1)
except Exception:
    pass

# ───────────────────────── App & Logging ─────────────────────────
app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ───────────────────────── CORS (fixed) ─────────────────────────
allowed_origins = [
    o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()
]
if allowed_origins:
    CORS(app, origins=allowed_origins)
else:
    CORS(app)  # fallback for local/dev

# ───────────────────────── Env Config ─────────────────────────
SHEET_KEY      = os.getenv("SHEET_KEY")  # REQUIRED
WORKSHEET_NAME = os.getenv("WORKSHEET_NAME", "Sheet1")
GCRED_PATH     = os.getenv("GSPREAD_CREDENTIALS", "/etc/secrets/credentials.json")

DEFAULT_FALLBACK_MESSAGE = os.getenv(
    "DEFAULT_FALLBACK_MESSAGE",
    ("Spiritual Meaning: Your spirit is asking for clarity and protection. "
     "Effects in the Physical Realm: You may feel confusion, delays, or mixed signals from people around you. "
     "What to Do: Pray for guidance, fast as you are led, write the dream plainly in one or two simple lines, "
     "and repeat it using clear keywords (e.g., 'falling', 'snake bite on hand', 'teeth falling out'). "
     "Avoid eating or accepting items in dreams; cover yourself spiritually before sleep.")
)

# Tunables
RELOAD_SEC       = int(os.getenv("RELOAD_SEC", "300"))       # hot reload cadence
DISABLE_SEMANTIC = os.getenv("DISABLE_SEMANTIC", "0") == "1" # set 1 to disable SBERT
SEMANTIC_CUTOFF  = float(os.getenv("SEMANTIC_CUTOFF", "0.60"))

# Simple rate limit (per IP)
RL_WINDOW_SEC    = int(os.getenv("RATE_WINDOW_SEC", "60"))
RL_LIMIT         = int(os.getenv("RATE_LIMIT", "60"))

# ───────────────────────── Globals ─────────────────────────
DF: Optional[pd.DataFrame] = None
LAST_LOAD = 0.0
SNAPSHOT = "/tmp/dream_dict.parquet"

EMB_MODEL = None
EMB_MATRIX = None

# ───────────────────────── Utils ─────────────────────────
def normalize_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s,/-]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def _sheet_error(msg: str):
    logging.error("❌ Sheet error: %s", msg)
    return False, msg

# ───────────────────────── Sheet Loading (with snapshot fallback) ─────────────────────────
def load_sheet(force: bool = False):
    """Load Google Sheet → DataFrame. On failure, fall back to a saved snapshot."""
    global DF, LAST_LOAD
    if not force and DF is not None and (time.time() - LAST_LOAD) < RELOAD_SEC:
        return True, "cached"

    if not SHEET_KEY:
        return _sheet_error("Missing SHEET_KEY")
    if not os.path.exists(GCRED_PATH):
        return _sheet_error(f"Missing credentials at {GCRED_PATH}")

    try:
        sa = gspread.service_account(filename=GCRED_PATH)
        ws = sa.open_by_key(SHEET_KEY).worksheet(WORKSHEET_NAME)
        records = ws.get_all_records()
        if not records:
            raise RuntimeError("Sheet is empty")

        df = pd.DataFrame(records)
        if "input" not in df.columns or "output" not in df.columns:
            raise RuntimeError("Sheet must include columns: input, output (keywords optional)")

        df["_input_norm"] = df["input"].astype(str).map(normalize_text)
        if "keywords" in df.columns:
            df["_kw_list"] = df["keywords"].astype(str).apply(
                lambda v: [normalize_text(x) for x in re.split(r"[,\|;/]", v) if x.strip()]
            )
        else:
            df["_kw_list"] = [[] for _ in range(len(df))]

        # Save snapshot for resilience
        try:
            df.to_parquet(SNAPSHOT, index=False)
        except Exception as e:
            logging.warning("Snapshot write failed: %s", e)

        # Assign
        globals()["DF"], globals()["LAST_LOAD"] = df, time.time()
        logging.info("✅ Sheet loaded: %d rows", len(df))

        # Build semantic index (optional)
        if not DISABLE_SEMANTIC:
            build_embeddings(df)

        return True, "loaded"

    except Exception as e:
        logging.error("❌ Live load failed: %s", e)
        # Try snapshot
        if os.path.exists(SNAPSHOT):
            try:
                df = pd.read_parquet(SNAPSHOT)
                globals()["DF"], globals()["LAST_LOAD"] = df, time.time()
                logging.info("⚠️ Using snapshot with %d rows", len(df))
                if not DISABLE_SEMANTIC:
                    build_embeddings(df)
                return True, "snapshot"
            except Exception as e2:
                return _sheet_error(f"Snapshot read failed: {e2}")
        return _sheet_error(str(e))

# ───────────────────────── Matching ─────────────────────────
def score_by_keywords(text_norm: str, kw_list: List[str]) -> int:
    return sum(1 for kw in kw_list if kw and kw in text_norm)

def find_best_match(user_input: str) -> Tuple[Optional[str], dict]:
    ok, msg = load_sheet()
    if not ok or DF is None:
        return None, {"status": "sheet_unavailable", "detail": msg}

    t = normalize_text(user_input)
    inputs = DF["_input_norm"].tolist()

    # 1) Exact
    if t in inputs:
        idx = inputs.index(t)
        return DF.iloc[idx]["output"], {"method": "exact"}

    # 2) Fuzzy
    close = difflib.get_close_matches(t, inputs, n=1, cutoff=0.82)
    if close:
        idx = inputs.index(close[0])
        return DF.iloc[idx]["output"], {"method": "fuzzy", "similar": close[0]}

    # 3) Keywords
    winners = [(score_by_keywords(t, row["_kw_list"]), i) for i, row in DF.iterrows()]
    winners = [(s, i) for s, i in winners if s > 0]
    if winners:
        winners.sort(reverse=True)
        _, idx = winners[0]
        return DF.iloc[idx]["output"], {"method": "keywords", "score": int(winners[0][0])}

    # 4) Semantic (optional)
    if not DISABLE_SEMANTIC:
        res, meta = semantic_search(t)
        if res:
            return res, meta

    return None, {"method": "none"}

# ───────────────────────── Semantic Search (SBERT) ─────────────────────────
def build_embeddings(df: pd.DataFrame):
    global EMB_MODEL, EMB_MATRIX
    try:
        from sentence_transformers import SentenceTransformer
        EMB_MODEL = EMB_MODEL or SentenceTransformer("all-MiniLM-L6-v2")
        texts = df["_input_norm"].tolist()
        EMB_MATRIX = EMB_MODEL.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        logging.info("🔎 Semantic index built: %d vectors", len(texts))
    except Exception as e:
        logging.warning("Semantic disabled (build failed): %s", e)
        EMB_MATRIX = None

def semantic_search(query_norm: str):
    try:
        if EMB_MODEL is None or EMB_MATRIX is None:
            return None, {"method": "semantic", "status": "not_ready"}
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity
        q = EMB_MODEL.encode([query_norm], convert_to_numpy=True, normalize_embeddings=True)
        sims = cosine_similarity(q, EMB_MATRIX)[0]
        idx = int(sims.argmax())
        score = float(sims[idx])
        if score >= SEMANTIC_CUTOFF:
            return DF.iloc[idx]["output"], {"method": "semantic", "score": score}
        return None, {"method": "semantic", "score": score}
    except Exception as e:
        logging.warning("Semantic search error: %s", e)
        return None, {"method": "semantic", "status": "error"}

# ───────────────────────── Rate Limiting ─────────────────────────
_bucket = defaultdict(list)
def _client_ip():
    return request.headers.get("X-Forwarded-For", request.remote_addr or "unknown").split(",")[0].strip()

@app.before_request
def throttle():
    if request.endpoint == "interpret":
        ip = _client_ip()
        now = time.time()
        _bucket[ip] = [t for t in _bucket[ip] if now - t < RL_WINDOW_SEC]
        if len(_bucket[ip]) >= RL_LIMIT:
            return jsonify({"error": "Too many requests"}), 429
        _bucket[ip].append(now)

# ───────────────────────── Routes ─────────────────────────
@app.route("/")
def index():
    try:
        return render_template("index.html")
    except Exception:
        return "Dream Interpreter API", 200

@app.route("/interpret", methods=["POST"])
def interpret():
    data = request.get_json(silent=True) or {}
    dream = (data.get("dream") or "").strip()
    if not dream:
        return jsonify({"interpretation": "Please enter a dream."}), 400

    result, meta = find_best_match(dream)
    logging.info("Dream from %s → %s", _client_ip(), meta)

    if result is None:
        return jsonify({"interpretation": DEFAULT_FALLBACK_MESSAGE,
                        "meta": {**meta, "method": "fallback"}}), 200
    return jsonify({"interpretation": result, "meta": meta}), 200

@app.route("/reload", methods=["POST", "GET"])
def reload_sheet():
    ok, detail = load_sheet(force=True)
    return jsonify({"ok": ok, "detail": detail}), (200 if ok else 500)

@app.route("/healthz")
def healthz():
    return jsonify({
        "status": "OK" if SHEET_KEY and os.path.exists(GCRED_PATH) else "Missing vars",
        "loaded": DF is not None
    }), 200

# ───────────────────────── Entrypoint ─────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
