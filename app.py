# app.py (stable baseline with improved matching + clear meta + exact-first)
import os, re, time, difflib, logging
from typing import List, Tuple, Optional
from collections import defaultdict

import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ── CORS (restrict in production with ALLOWED_ORIGINS) ─────────
allowed_origins = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()]
CORS(app, origins=allowed_origins or "*")

# ── Env ───────────────────────────────────────────────────────
SHEET_KEY        = os.getenv("SHEET_KEY")              # required
WORKSHEET_NAME   = os.getenv("WORKSHEET_NAME", "Sheet1")
CREDENTIALS_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "/etc/secrets/credentials.json")

DEFAULT_FALLBACK_MESSAGE = os.getenv(
    "DEFAULT_FALLBACK_MESSAGE",
    "Spiritual Meaning: Your spirit is asking for clarity and protection. "
    "Effects in the Physical Realm: You may feel confusion or delays. "
    "What to Do: Pray for guidance, write the dream clearly, and try again with simple keywords."
)

RELOAD_SEC       = int(os.getenv("RELOAD_SEC", "300"))
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
EMB_MATRIX = None

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

# ── Matching helpers ──────────────────────────────────────────
def _similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()

def _token_set(s: str) -> set:
    return set(s.split())

def _substring_pick(t: str, inputs: List[str], min_len: int = 3) -> Optional[int]:
    """
    Pick the most specific sheet phrase contained in the user text (or vice-versa).
    Lower min_len so short symbols like 'egg' still qualify, but we use this
    AFTER exact/Jaccard so one-word exacts aren't overridden.
    """
    cand = [(i, s) for i, s in enumerate(inputs) if len(s) >= min_len and (s in t or t in s)]
    if not cand:
        return None
    cand.sort(key=lambda x: (len(x[1]), _similarity(t, x[1])), reverse=True)
    return cand[0][0]

def _jaccard_pick(t: str, inputs: List[str], thresh: float = 0.65) -> Optional[int]:
    """Token Jaccard similarity as a robust middle ground for short phrases."""
    T = _token_set(t)
    if not T:
        return None
    best = (-1.0, None)
    for i, s in enumerate(inputs):
        S = _token_set(s)
        if not S:
            continue
        j = len(T & S) / float(len(T | S))
        if j > best[0]:
            best = (j, i)
    return best[1] if best[0] >= thresh else None

def score_by_keywords(text_norm: str, kw_list: List[str]) -> int:
    return sum(1 for kw in kw_list if kw and kw in text_norm)

# ── Main matcher ──────────────────────────────────────────────
def find_best_match(user_input: str) -> Tuple[Optional[str], dict]:
    ok, msg = load_sheet()
    if not ok or DF is None:
        return None, {"status": "sheet_unavailable", "detail": msg}

    t = normalize_text(user_input)
    inputs = DF["_input_norm"].tolist()

    # SPECIAL: if the user typed a single token, try exact on one-token rows first
    if " " not in t:
        for i, s in enumerate(inputs):
            if " " not in s and s == t:
                return DF.iloc[i]["output"], {"method": "exact_single", "matched_input": DF.iloc[i]["input"]}

    # 1) Exact first (prevents substring from stealing one-word perfect hits like 'naked')
    if t in inputs:
        idx = inputs.index(t)
        return DF.iloc[idx]["output"], {"method": "exact", "matched_input": DF.iloc[idx]["input"]}

    # 2) Token Jaccard (great for short phrases)
    jac_idx = _jaccard_pick(t, inputs, thresh=0.65)
    if jac_idx is not None:
        return DF.iloc[jac_idx]["output"], {"method": "jaccard", "matched_input": DF.iloc[jac_idx]["input"]}

    # 3) Substring containment (useful when the sheet phrase is literally inside user text)
    sub_idx = _substring_pick(t, inputs, min_len=3)
    if sub_idx is not None:
        return DF.iloc[sub_idx]["output"], {"method": "substring", "matched_input": DF.iloc[sub_idx]["input"]}

    # 4) Fuzzy (high bar)
    best_ratio, best_idx = -1.0, None
    for i, s in enumerate(inputs):
        r = difflib.SequenceMatcher(None, t, s).ratio()
        if r > best_ratio:
            best_ratio, best_idx = r, i
    if best_idx is not None and best_ratio >= 0.84:
        return DF.iloc[best_idx]["output"], {
            "method": "fuzzy",
            "ratio": round(best_ratio, 4),
            "matched_input": DF.iloc[best_idx]["input"],
        }

    # 5) Keywords with deterministic tie-break
    scored = []
    for i, row in DF.iterrows():
        kw_score = score_by_keywords(t, row["_kw_list"])
        if kw_score > 0:
            sim = _similarity(t, row["_input_norm"])
            scored.append((kw_score, sim, len(row["_input_norm"]), i))
    if scored:
        scored.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
        kw, sim, ln, idx = scored[0]
        return DF.iloc[idx]["output"], {
            "method": "keywords",
            "score": int(kw),
            "similarity": round(float(sim), 3),
            "matched_input": DF.iloc[idx]["input"],
        }

    # 6) Semantic (optional)
    if not DISABLE_SEMANTIC:
        sem_text, meta = semantic_search(t)
        if sem_text:
            meta = dict(meta or {})
            try:
                mi = DF.iloc[int(meta.get("idx", 0))]["input"]
                meta["matched_input"] = mi
            except Exception:
                pass
            if float(meta.get("score", 0.0)) >= SEMANTIC_CUTOFF:
                return sem_text, meta
            if float(meta.get("score", 0.0)) >= SEMANTIC_SOFT:
                meta["low_confidence"] = True
                return sem_text, meta

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
    # Accept both keys to avoid front-end mismatch
    dream = (data.get("dream") or data.get("text") or "").strip()
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

# ── Debug helper (optional) ───────────────────────────────────
@app.route("/debug_match")
def debug_match():
    q = (request.args.get("q") or "").strip()
    if not q:
        return jsonify({"error": "add ?q=your+text"}), 400
    result, meta = find_best_match(q)
    sample_inputs = DF["_input_norm"].head(10).tolist() if DF is not None else []
    return jsonify({"query_norm": normalize_text(q), "result": result, "meta": meta, "sample_inputs": sample_inputs})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
