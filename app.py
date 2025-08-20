# app.py
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os, re, difflib
import gspread
import pandas as pd

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

# === CORS ===
CORS(
    app,
    resources={
        r"/interpret": {"origins": ALLOWED_ORIGINS},
        r"/refresh":   {"origins": ALLOWED_ORIGINS},
        r"/healthz":   {"origins": ALLOWED_ORIGINS},
        r"/ping":      {"origins": ALLOWED_ORIGINS},
        r"/":          {"origins": ALLOWED_ORIGINS},
    },
    supports_credentials=False,
    methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Accept"],  # Accept helps some browsers/tools
    max_age=600
)

# === OPTIONAL: allow embedding full app in an <iframe> from your shop/domain ===
# Uncomment if you plan to iframe the WHOLE Render app inside Shopify.
# @app.after_request
# def add_csp(resp):
#     resp.headers["Content-Security-Policy"] = (
#         "frame-ancestors "
#         "https://jamaicantruestories.com "
#         "https://www.jamaicantruestories.com "
#         "https://*.myshopify.com"
#     )
#     resp.headers.pop("X-Frame-Options", None)
#     return resp

# === CACHED OBJECTS ===
_gc = None
_ws = None
_df = None

# --- helpers ---
def _norm(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

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

def get_df(force_refresh: bool = False):
    """Read the sheet, normalize headers & create helper columns."""
    global _df
    if force_refresh or _df is None:
        ws = get_ws()
        rows = ws.get_all_records()  # list[dict]
        df = pd.DataFrame(rows)

        if df.empty:
            df = pd.DataFrame(columns=["input", "output"])

        # normalize headers
        df.columns = [str(c).strip().lower() for c in df.columns]

        # create working columns only if required cols present
        cols = set(df.columns)
        if ("input" in cols or "symbol" in cols) and ("output" in cols):
            cand_col = "input" if "input" in cols else "symbol"
            df["_cand"] = df[cand_col].astype(str).map(_norm)

            # keywords: support comma or semicolon separated
            if "keywords" in cols:
                def parse_kw(v):
                    if pd.isna(v) or v is None:
                        return []
                    # split on comma or semicolon
                    parts = re.split(r"[;,]", str(v))
                    return [_norm(p) for p in parts if _norm(p)]
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
    jaccard = (len(q & c) / len(q | c)) if (q and c) else 0.0
    fuzzy = difflib.SequenceMatcher(None, _norm(query), _norm(candidate)).ratio()
    return 0.6 * jaccard + 0.4 * fuzzy

def best_match(user_text: str, df: pd.DataFrame):
    """Return (row, score, matched_string) or (None, 0, None)."""
    if df.empty or "_cand" not in df.columns:
        return None, 0.0, None
    pool = []
    for idx, row in df.iterrows():
        pool.append((idx, row["_cand"]))
        for kw in row["_kw"]:
            pool.append((idx, kw))

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

        row, score, matched = best_match(user_text, df)
        if row is None or score < 0.28:
            return jsonify({
                "match": None,
                "score": round(float(score), 3),
                "interpretation": (
                    "I couldn't find a close match yet. Try a simpler phrase "
                    "(e.g., 'teeth falling out', 'running late', 'snake bite')."
                )
            }), 200

        interpretation = str(row.get("_output_txt", "")).strip() or "No interpretation found yet."
        canonical = str(row.get("_cand", "")).strip()

        return jsonify({
            "match": canonical,
            "score": round(float(score), 3),
            "interpretation": interpretation
        }), 200

    except Exception as e:
        # Surface server-side errors instead of a blank 500
        return jsonify({"error": f"Server error: {e}"}), 500

if __name__ == "__main__":
    # Render sets PORT env var; default to 5000 locally
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")))
