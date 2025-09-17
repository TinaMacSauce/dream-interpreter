# app.py (stable baseline with optional semantic support)
import os, re, time, difflib, logging
from typing import List, Tuple, Optional
from collections import defaultdict

import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# ── App & logging ─────────────────────────────────────────────
app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ── CORS (restrict in production) ─────────────────────────────
allowed_origins = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()]
CORS(app, origins=allowed_origins or "*")

# ── Env ───────────────────────────────────────────────────────
SHEET_KEY        = os.getenv("SHEET_KEY")              # required
WORKSHEET_NAME   = os.getenv("WORKSHEET_NAME", "Sheet1")
CREDENTIALS_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "/etc/secrets/credentials.json")

DEFAULT_FALLBACK_MESSAGE = os.getenv("DEFAULT_FALLBACK_MESSAGE",
    "Spiritual Meaning: Your spirit is asking for clarity and protection. "
    "Effects in the Physical Realm: You may feel confusion or delays. "
    "What to Do: Pray for guidance, write the dream clearly, and try again with simple keywords."
)

RELOAD_SEC       = int(os.getenv("RELOAD_SEC", "300"))
# Semantic flags (we’ll enable later)
DISABLE_SEMANTIC = os.getenv("DISABLE_SEMANTIC", "1") == "1"
SEMANTIC_CUTOFF  = float(os.getenv("SEMANTIC_CUTOFF", "0.60"))
SEMANTIC_SOFT    = float(os.getenv("SEMANTIC_SOFT", "0.45"))

# Simple rate limit
RL_WINDOW_SEC    = int(os.getenv("RATE_WINDOW_SEC", "60"))
RL_LIMIT         = int(os.getenv("RATE_LIMIT", "60"))

# ── Globals ───────────────────────────────────────────────────
DF: Optional[pd.DataFrame] = None
LAST_LOAD = 0.0
SNAPSHOT_CSV = "/tmp/dream_dict.csv"

_GS_CLIENT = None
EMB_MODEL = None
EMB_MATRIX = None  # filled when semantic enabled

# ── Utils ─────────────────────────────────────────────────────
def normalize_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s,/-]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def _gs_client():
    global _GS_CLIENT
    if _GS_CLIENT is not None:
        return _GS_CLIENT
    if not os.path.exists(CREDENTIALS_PATH):
        raise FileNotFoundError(f"Credentials file not found at {CREDENTIALS_PATH}")
    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    creds = Credentials.from_service_account_file(CREDENTIALS_PATH, scopes=scopes)
    _GS_CLIENT = gspread.authorize(creds)
    return _GS_CLIENT

# ── Snapshot helpers (CSV keeps deps light) ───────────────────
def _save_snapshot(df: pd.DataFrame):
    try:
        df.to_csv(SNAPSHOT_CSV, index=False)
    except Exception as e:
        logging.warning("Snapshot save failed: %s", e)

def _load_snapshot() -> Optional[pd.DataFrame]:
    try:
        if os.path.exists(SNAPSHOT_CSV):
            return pd.read_csv(SNAPSHOT_CSV)
    except Exception as e:
        logging.warning("Snapshot read failed: %s", e)
    return None

# ── Load sheet ────────────────────────────────────────────────
def load_sheet(force: bool = False):
    global DF, LAST_LOAD
    if not force and DF is not None and (time.time() - LAST_LOAD) < RELOAD_SEC:
        return True, "cached"
    if not SHEET_KEY:
        logging.error("Missing SHEET_KEY")
        return False, "missing_key"
    try:
        ws = _gs_client().open_by_key(SHEET_KEY).worksheet(WORKSHEET_NAME)
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
        _save_snapshot(df)
        DF, LAST_LOAD = df, time.time()
        logging.info("✅ Sheet loaded: %d rows", len(df))
        if not DISABLE_SEMANTIC:
            build_embeddings(df)
        return True, "loaded"
    except Exception as e:
        logging.error("Live load failed: %s", e)
        snap = _load_snapshot()
        if snap is not None:
            DF, LAST_LOAD = snap, time.time()
            logging.info("⚠️ Using snapshot with %d rows", len(snap))
            if not DISABLE_SEMANTIC:
                build_embeddings(snap)
            return True, "snapshot"
        return False, str(e)

# ── Matching ──────────────────────────────────────────────────
def score_by_keywords(text_norm: str, kw_list: List[str]) -> int:
    return sum(1 for kw in kw_list if kw and kw in text_norm)

def find_best_match(user_input: str) -> Tuple[Optional[str], dict]:
    ok, msg = load_sheet()
    if not ok or DF is None:
        return None, {"status": "sheet_unavailable", "detail": msg}
    t = normalize_text(user_input)

    # 1) Exact
    inputs = DF["_input_norm"].tolist()
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
    best_kw = None
    if winners:
        winners.sort(reverse=True)
        s, idx = winners[0]
        best_kw = (DF.iloc[idx]["output"], {"method": "keywords", "score": int(s)})

    # 4) Semantic (strong/soft)
    if not DISABLE_SEMANTIC:
        sem_text, meta = semantic_search(t)
        score = float(meta.get("score", 0.0)) if isinstance(meta, dict) else 0.0
        if sem_text and score >= SEMANTIC_CUTOFF:
            return sem_text, {"method": "semantic", "score": score}
        if sem_text and score >= SEMANTIC_SOFT and not best_kw:
            return sem_text, {"method": "semantic_low", "score": score, "low_confidence": True}

    if best_kw:
        return best_kw
    return None, {"method": "none"}

# ── Semantic (loaded only when enabled) ───────────────────────
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
        from sklearn.metrics.pairwise import cosine_similarity
        q = EMB_MODEL.encode([query_norm], convert_to_numpy=True, normalize_embeddings=True)
        sims = cosine_similarity(q, EMB_MATRIX)[0]
        idx = int(sims.argmax())
        score = float(sims[idx])
        return DF.iloc[idx]["output"], {"method": "semantic", "score": score, "idx": idx}
    except Exception as e:
        logging.warning("Semantic search error: %s", e)
        return None, {"method": "semantic", "status": "error"}

# ── Rate limit ────────────────────────────────────────────────
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

# ── Routes ────────────────────────────────────────────────────
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
        return jsonify({"interpretation": DEFAULT_FALLBACK_MESSAGE, "meta": {**meta, "method": "fallback"}}), 200
    return jsonify({"interpretation": result, "meta": meta}), 200

@app.route("/reload", methods=["POST", "GET"])
def reload_sheet():
    ok, detail = load_sheet(force=True)
    return jsonify({"ok": ok, "detail": detail}), (200 if ok else 500)

@app.route("/healthz")
def healthz():
    ok, source = load_sheet(force=False)
    return jsonify({
        "status": "OK" if ok else "ERROR",
        "loaded": DF is not None,
        "source": source,
        "has_sheet_key": bool(SHEET_KEY),
        "has_credentials_file": os.path.exists(CREDENTIALS_PATH),
        "worksheet": WORKSHEET_NAME
    }), (200 if ok else 500)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
