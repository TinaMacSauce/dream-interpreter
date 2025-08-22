# app.py
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os, re, difflib, unicodedata
import gspread
import pandas as pd
from typing import List, Tuple, Dict, Any

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024  # 2MB

# === CONFIG ===
SHEET_KEY = "1ToWPpJ_u-Z14eqL9U1oJppXX6f63h_eZcjUVglKr4Zk"
WORKSHEET_NAME = "Sheet1"
GCRED_PATH = os.getenv("GSPREAD_CREDENTIALS", "/etc/secrets/credentials.json")

# Allowed Shopify/front-end origins (edit env to match EXACT storefront URLs)
ALLOWED_ORIGINS = [
    o.strip() for o in os.getenv(
        "ALLOWED_ORIGINS",
        (
            "https://jamaicantruestories.com,"
            "https://www.jamaicantruestories.com,"
            "https://plqwhd-jm.myshopify.com,"
            "https://jamaicantruestories.myshopify.com,"
            "https://admin.shopify.com"
        )
    ).split(",") if o.strip()
]

# === CORS (GLOBAL) ===
CORS(
    app,
    origins=ALLOWED_ORIGINS,
    supports_credentials=False,
    methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Accept"],
    max_age=600
)

# === CACHED OBJECTS ===
_gc = None
_ws = None
_df: pd.DataFrame | None = None
_kw_index: Dict[str, set] | None = None   # NEW: exact keyword -> set(row_indices)

# --- helpers ---
def _strip_accents(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", s)
        if not unicodedata.combining(c)
    )

def _norm(s: str | None) -> str:
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = _strip_accents(s)
    # unify punctuation/spaces
    s = re.sub(r"[‐‑–—]", "-", s)                    # normalize dashes
    s = re.sub(r"[^a-z0-9\s,;/\-]", " ", s)          # drop weird punctuation
    s = re.sub(r"\s+", " ", s)                       # collapse spaces
    return s.strip()

def _split_keywords(cell_val: Any) -> List[str]:
    """Split a keywords cell by comma/semicolon and normalize each item."""
    if cell_val is None or (isinstance(cell_val, float) and pd.isna(cell_val)):
        return []
    parts = re.split(r"[;,]", str(cell_val))
    out = []
    for p in parts:
        p = _norm(p)
        if p:
            out.append(p)
    return out

def _make_kw_index(df: pd.DataFrame) -> Dict[str, set]:
    """Build exact keyword -> {row_indices} index for fast lookup."""
    idx: Dict[str, set] = {}
    if "_kw" not in df.columns:
        return idx
    for i, kws in enumerate(df["_kw"]):
        for k in kws:
            if k not in idx:
                idx[k] = set()
            idx[k].add(i)
    return idx

def get_ws():
    """Return worksheet; raise clear errors if not accessible."""
    global _gc, _ws
    try:
        if _gc is None:
            _gc = gspread.service_account(filename=GCRED_PATH)
        if _ws is None:
            _ws = _gc.open_by_key(SHEET_KEY).worksheet(WORKSHEET_NAME)
        return _ws
    except gspread.exceptions.WorksheetNotFound as e:
        raise RuntimeError(f"Worksheet '{WORKSHEET_NAME}' not found.") from e
    except gspread.exceptions.SpreadsheetNotFound as e:
        raise RuntimeError("Spreadsheet not found or not shared with the service account.") from e
    except Exception as e:
        raise RuntimeError(f"Sheets auth error: {e}") from e

def get_df(force_refresh: bool = False) -> pd.DataFrame:
    """Read the sheet, normalize headers & create helper columns and keyword index."""
    global _df, _kw_index
    if force_refresh or _df is None:
        ws = get_ws()
        rows = ws.get_all_records()  # list[dict]
        df = pd.DataFrame(rows)

        if df.empty:
            df = pd.DataFrame(columns=["input", "output", "keywords"])

        # normalize headers
        df.columns = [str(c).strip().lower() for c in df.columns]

        # choose canonical input col
        cand_col = "input" if "input" in df.columns else ("symbol" if "symbol" in df.columns else None)
        if cand_col is None:
            # create empty columns to prevent breakage
            df["_cand"] = ""
            df["_kw"] = [[] for _ in range(len(df))]
            df["_output_txt"] = df["output"].astype(str) if "output" in df.columns else ""
        else:
            df["_cand"] = df[cand_col].astype(str).map(_norm)
            # parse keywords if present
            if "keywords" in df.columns:
                df["_kw"] = df["keywords"].map(_split_keywords)
            else:
                df["_kw"] = [[] for _ in range(len(df))]
            df["_output_txt"] = df["output"].astype(str) if "output" in df.columns else ""

        _df = df
        _kw_index = _make_kw_index(_df)
    return _df

def _similarity(query: str, candidate: str) -> float:
    """Blend token overlap with difflib; boosted for exact match."""
    qn = _norm(query)
    cn = _norm(candidate)
    if not qn or not cn:
        return 0.0
    if qn == cn:
        return 2.0  # exact match SUPER-BOOST (beats everything)
    # token Jaccard + fuzzy
    q = set(qn.split())
    c = set(cn.split())
    jaccard = (len(q & c) / len(q | c)) if (q and c) else 0.0
    fuzzy = difflib.SequenceMatcher(None, qn, cn).ratio()
    return 0.65 * jaccard + 0.35 * fuzzy

def _generate_query_variants(user_text: str) -> List[str]:
    """
    Build a small set of variants from the input to improve matches:
    - original
    - split by commas/semicolons (users may paste multiple tags)
    """
    base = _norm(user_text)
    parts = [base]
    # add pieces split by comma/semicolon if present
    for p in re.split(r"[;,]", base):
        p = p.strip()
        if p and p not in parts:
            parts.append(p)
    return [p for p in parts if p]

def find_matches(user_text: str, df: pd.DataFrame, top_k: int = 3) -> List[Tuple[int, float, str]]:
    """
    Return a list of (row_index, score, matched_string) sorted by score desc.
    Priority:
      1) Exact keyword match (score=2.0)
      2) Keyword fuzzy
      3) Canonical input fuzzy
    """
    if df.empty or "_cand" not in df.columns:
        return []

    variants = _generate_query_variants(user_text)

    # 1) Exact keyword matches first (via index)
    hits: Dict[int, Tuple[float, str]] = {}
    if _kw_index:
        for v in variants:
            if v in _kw_index:
                for row_idx in _kw_index[v]:
                    prev = hits.get(row_idx)
                    score = 2.0
                    if not prev or score > prev[0]:
                        hits[row_idx] = (score, v)

    # 2) Fuzzy over keywords & 3) canonical text
    # Build candidate pool: each row's keywords + canonical string
    pool: List[Tuple[int, str]] = []
    for idx, row in df.iterrows():
        # keywords first (so they tend to outrank canonical text if similar)
        for kw in row.get("_kw", []):
            pool.append((idx, kw))
        pool.append((idx, row.get("_cand", "")))

    seen_sim: Dict[Tuple[str, str], float] = {}
    for idx, cand in pool:
        if not cand:
            continue
        best_local = 0.0
        best_v = None
        for v in variants:
            key = (v, cand)
            score = seen_sim.get(key)
            if score is None:
                score = _similarity(v, cand)
                seen_sim[key] = score
            if score > best_local:
                best_local = score
                best_v = v
        # keep only good ones
        if best_local > 0:
            prev = hits.get(idx)
            if not prev or best_local > prev[0]:
                hits[idx] = (best_local, best_v or cand)

    # sort & take top_k
    ranked = sorted(((i, s, m) for i, (s, m) in hits.items()), key=lambda t: t[1], reverse=True)
    return ranked[:top_k]

# --- routes ---
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ping")
def ping():
    return jsonify({"ok": True})

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

@app.route("/interpret", methods=["POST", "OPTIONS"])
def interpret():
    try:
        payload = request.get_json(silent=True) or {}
        # Accept both keys from different frontends (JSON or form)
        user_text = (
            payload.get("dream_text")
            or payload.get("text")
            or (request.form.get("dream") or request.form.get("dream_text") or request.form.get("text"))
        )

        if not user_text or not _norm(user_text):
            return jsonify({"error": "Provide dream text in 'dream_text' or 'text'"}), 400

        df = get_df()
        cols = list(df.columns)
        if ("input" not in cols and "symbol" not in cols) or ("output" not in cols):
            return jsonify({
                "error": "Sheet must contain columns 'input' (or 'symbol') and 'output'.",
                "columns_found": cols
            }), 500

        # NEW: return multiple candidates (top_k)
        ranked = find_matches(user_text, df, top_k=int(payload.get("top_k", 3)))
        if not ranked:
            return jsonify({
                "matches": [],
                "interpretations": [],
                "message": "No close match yet. Try a simpler phrase (e.g., 'teeth falling out', 'snake bite')."
            }), 200

        results = []
        for idx, score, matched in ranked:
            row = df.iloc[idx]
            interpretation = str(row.get("_output_txt", "")).strip() or "No interpretation found yet."
            canonical = str(row.get("_cand", "")).strip()
            results.append({
                "match": canonical,
                "matched_on": matched,
                "score": round(float(score), 3),
                "interpretation": interpretation
            })

        # Back-compat (first/best item) + full list
        return jsonify({
            "match": results[0]["match"],
            "matched_on": results[0]["matched_on"],
            "score": results[0]["score"],
            "interpretation": results[0]["interpretation"],
            "matches": results,                # list of all candidates returned
        }), 200

    except Exception as e:
        return jsonify({"error": f"Server error: {e}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")))
