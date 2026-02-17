# app.py — Jamaican True Stories Dream Interpreter (Phase 2.9.6 — ADMIN PANEL + AUTO-ADD + OPTIMIZE)
# Based on your current Phase 2.9.5 (subsumption + compound priority + admin import/clean).
# Adds:
# - ✅ /admin (mobile-friendly Admin Panel UI, locked behind ADMIN_KEY)
# - ✅ /admin/add-symbol (add or upsert ONE symbol row into Sheet1, formatting-only)
# - ✅ /admin/batch-add (add/upsert MANY symbol rows from JSON, formatting-only)
# - ✅ /admin/optimize-dictionary (formatting-only optimization + de-dupe by input)
# Keeps:
# - /, /health, /healthz, /interpret, /track
# - strict matching + compound priority + subsumption pruning
# - doctrine-safe Full Interpretation
# - admin clean/import routes

import os
import json
import time
import re
import secrets
import csv
import io
from typing import Dict, List, Tuple, Any, Optional

from flask import Flask, request, jsonify, make_response, render_template
from flask_cors import CORS

import gspread
from google.oauth2.service_account import Credentials


# ----------------------------
# App setup
# ----------------------------
app = Flask(__name__)

DEFAULT_ALLOWED = [
    "https://jamaicantruestories.com",
    "https://www.jamaicantruestories.com",
    "https://interpreter.jamaicantruestories.com",
]
allowed_origins_env = os.getenv("ALLOWED_ORIGINS", "")
allowed_origins = [o.strip() for o in allowed_origins_env.split(",") if o.strip()] or DEFAULT_ALLOWED

CORS(
    app,
    resources={r"/*": {"origins": allowed_origins}},
    supports_credentials=True,
    allow_headers=["Content-Type", "Authorization", "X-Admin-Key"],
    methods=["GET", "POST", "OPTIONS"],
)

SPREADSHEET_ID = os.getenv("SPREADSHEET_ID", "").strip()
WORKSHEET_NAME = os.getenv("WORKSHEET_NAME", "Sheet1").strip()

# IMPORTANT: write scope is required for admin routes
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive.readonly",
]

CACHE_TTL_SECONDS = int(os.getenv("SHEET_CACHE_TTL", "120"))
_CACHE: Dict[str, Any] = {"loaded_at": 0.0, "rows": [], "headers": []}

DEBUG_MATCH = os.getenv("DEBUG_MATCH", "").strip().lower() in {"1", "true", "yes", "on"}

# Narrative style knobs (optional)
NARRATIVE_ENABLED = os.getenv("NARRATIVE_ENABLED", "1").strip().lower() not in {"0", "false", "no", "off"}
NARRATIVE_MAX_SYMBOLS = int(os.getenv("NARRATIVE_MAX_SYMBOLS", "3"))

# Admin auth (required for admin routes)
ADMIN_KEY = os.getenv("ADMIN_KEY", "").strip()


# ----------------------------
# Helpers
# ----------------------------
def _preflight_ok():
    return make_response("", 204)


def _admin_ok(req) -> bool:
    """
    Admin auth. Accepts either:
    - Header: X-Admin-Key: <key>
    - Query:  ?key=<key>
    """
    if not ADMIN_KEY:
        return False
    k = (req.headers.get("X-Admin-Key") or req.args.get("key") or "").strip()
    return secrets.compare_digest(k, ADMIN_KEY)


def _normalize_header(h: str) -> str:
    h = (h or "").strip().lower()
    h = re.sub(r"\s+", " ", h)
    return h


def _normalize_text(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _log(*args):
    if DEBUG_MATCH:
        print(*args, flush=True)


def _clean_sentence(s: str) -> str:
    if not s:
        return ""
    s = str(s).strip()
    s = re.sub(r"\s+", " ", s)
    s = s.replace(" ,", ",").replace(" .", ".").replace(" :", ":").replace(" ;", ";")
    s = re.sub(r"([.!?]){2,}", r"\1", s)
    return s


def _strip_trailing_punct(s: str) -> str:
    s = _clean_sentence(s)
    return s.rstrip(" .!?\t\r\n")


def _ensure_terminal_punct(s: str) -> str:
    s = _clean_sentence(s)
    if not s:
        return ""
    if s[-1] not in ".!?":
        s += "."
    return s


def _title_case_symbol(sym: str) -> str:
    sym = _clean_sentence(sym)
    if not sym:
        return ""
    return sym[:1].upper() + sym[1:]


def _format_label_value(symbol: str, text: str) -> str:
    symbol = _title_case_symbol(symbol)
    text = _clean_sentence(text)
    if not symbol or not text:
        return ""
    return f"{symbol}: {text}"


def _fix_typos_light(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip()
    s = re.sub(r"\s+", " ", s)

    replacements = [
        (r"\bembarassment\b", "embarrassment"),
        (r"\bembarrasment\b", "embarrassment"),
        (r"\bcommmunity\b", "community"),
        (r"\bcommnity\b", "community"),
        (r"\bcomunity\b", "community"),
    ]
    for pat, rep in replacements:
        s = re.sub(pat, rep, s, flags=re.IGNORECASE)

    s = s.replace(" ,", ",").replace(" .", ".").replace(" :", ":").replace(" ;", ";")
    return s.strip()


def _normalize_keywords_cell(raw: str) -> str:
    if raw is None:
        return ""
    parts = re.split(r"[,\|;]+", str(raw))
    out = []
    seen = set()
    for p in parts:
        p = re.sub(r"\s+", " ", str(p)).strip().lower()
        if not p:
            continue
        if p in seen:
            continue
        seen.add(p)
        out.append(p)
    return ", ".join(out)


def _get_credentials() -> Credentials:
    raw = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON", "").strip()
    if raw:
        raw_fixed = raw.replace("\\n", "\n")
        info = json.loads(raw_fixed)
        return Credentials.from_service_account_info(info, scopes=SCOPES)

    cred_path = os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE", "credentials.json")
    if not os.path.exists(cred_path):
        raise RuntimeError(
            "Missing Google credentials. Set GOOGLE_SERVICE_ACCOUNT_JSON in Render "
            "or provide credentials.json via GOOGLE_SERVICE_ACCOUNT_FILE."
        )
    return Credentials.from_service_account_file(cred_path, scopes=SCOPES)


def _get_ws():
    if not SPREADSHEET_ID:
        raise RuntimeError("SPREADSHEET_ID env var is not set.")
    creds = _get_credentials()
    gc = gspread.authorize(creds)
    sh = gc.open_by_key(SPREADSHEET_ID)
    ws = sh.worksheet(WORKSHEET_NAME)
    return ws


def _load_sheet_rows(force: bool = False) -> List[Dict]:
    now = time.time()
    if (not force) and _CACHE["rows"] and (now - _CACHE["loaded_at"] < CACHE_TTL_SECONDS):
        return _CACHE["rows"]

    ws = _get_ws()
    values = ws.get_all_values()
    if not values or len(values) < 2:
        _CACHE["rows"] = []
        _CACHE["headers"] = []
        _CACHE["loaded_at"] = now
        return []

    headers = [_normalize_header(h) for h in values[0]]
    rows: List[Dict] = []

    for r in values[1:]:
        if len(r) < len(headers):
            r = r + [""] * (len(headers) - len(r))
        item = {headers[i]: (r[i] or "").strip() for i in range(len(headers))}
        rows.append(item)

    _CACHE["rows"] = rows
    _CACHE["headers"] = headers
    _CACHE["loaded_at"] = now
    return rows


def _split_keywords(s: str) -> List[str]:
    if not s:
        return []
    parts = re.split(r"[,\|;]+", s)
    out = []
    for p in parts:
        p = _normalize_text(p)
        if p:
            out.append(p)
    return out


def _compile_boundary_regex(token: str) -> re.Pattern:
    token_n = _normalize_text(token)
    if not token_n:
        return re.compile(r"(?!x)x")
    return re.compile(rf"(?<!\w){re.escape(token_n)}(?!\w)")


# ---------- Subsumption helpers ----------
def _is_subsumed(short_sym: str, long_sym: str) -> bool:
    a = _normalize_text(short_sym)
    b = _normalize_text(long_sym)
    if not a or not b:
        return False
    if a == b:
        return False
    rx = re.compile(rf"(?<!\w){re.escape(a)}(?!\w)")
    return bool(rx.search(b))


def _prune_subsumed_matches(
    matches: List[Tuple[Dict, int, Optional[Dict[str, str]]]]
) -> List[Tuple[Dict, int, Optional[Dict[str, str]]]]:
    if not matches:
        return matches

    syms = [(row.get("input") or row.get("symbol") or "").strip() for row, _, _ in matches]
    keep = [True] * len(matches)

    for i in range(len(matches)):
        if not keep[i]:
            continue
        si = syms[i]
        if not si:
            continue
        for j in range(len(matches)):
            if i == j or not keep[j]:
                continue
            sj = syms[j]
            if not sj:
                continue
            if _is_subsumed(si, sj):
                keep[i] = False
                break

    return [m for m, k in zip(matches, keep) if k]


# Match ranking priorities
_HIT_PRIORITY = {"symbol_phrase": 3, "symbol": 2, "keyword": 1}


def _score_row_strict(dream_norm: str, row: Dict) -> Tuple[int, Optional[Dict[str, str]]]:
    symbol_raw = (row.get("input") or row.get("symbol") or "").strip()
    if not symbol_raw:
        return 0, None

    symbol = _normalize_text(symbol_raw)
    if not symbol:
        return 0, None

    symbol_words = symbol.split()
    keywords = _split_keywords(row.get("keywords", ""))

    if len(symbol_words) == 1:
        if _compile_boundary_regex(symbol).search(dream_norm):
            return 100, {"type": "symbol", "token": symbol}
    else:
        phrase_rx = re.compile(rf"(?<!\w){re.escape(symbol)}(?!\w)")
        if phrase_rx.search(dream_norm):
            return 100, {"type": "symbol_phrase", "token": symbol}

    if len(symbol_words) == 1:
        for kw in keywords:
            if kw and _compile_boundary_regex(kw).search(dream_norm):
                return 96, {"type": "keyword", "token": kw}

    return 0, None


def _match_symbols_strict(dream: str, rows: List[Dict], top_k: int = 3) -> List[Tuple[Dict, int, Optional[Dict[str, str]]]]:
    dream_norm = _normalize_text(dream)
    if not dream_norm:
        return []

    scored: List[Tuple[Dict, int, Optional[Dict[str, str]]]] = []
    for row in rows:
        sc, hit = _score_row_strict(dream_norm, row)
        if sc > 0:
            scored.append((row, sc, hit))

    def _sort_key(item):
        row, sc, hit = item
        sym = (row.get("input") or row.get("symbol") or "").strip()
        sym_len = len(_normalize_text(sym))
        hit_type = (hit or {}).get("type", "")
        hit_pri = _HIT_PRIORITY.get(hit_type, 0)
        return (-sc, -hit_pri, -sym_len)

    scored.sort(key=_sort_key)

    seen = set()
    out: List[Tuple[Dict, int, Optional[Dict[str, str]]]] = []
    for row, sc, hit in scored:
        sym = (row.get("input") or row.get("symbol") or "").strip()
        sym_key = _normalize_text(sym)
        if not sym_key or sym_key in seen:
            continue
        seen.add(sym_key)
        out.append((row, sc, hit))
        if len(out) >= top_k:
            break

    return out


def _combine_fields(matches: List[Tuple[Dict, int, Optional[Dict[str, str]]]]) -> Dict[str, str]:
    spiritual_parts: List[str] = []
    physical_parts: List[str] = []
    action_parts: List[str] = []

    for row, _sc, _hit in matches:
        symbol = (row.get("input") or row.get("symbol") or "").strip()
        sm = (row.get("spiritual meaning") or row.get("spiritual_meaning") or "").strip()
        pe = (row.get("physical effects") or row.get("physical_effects") or "").strip()
        ac = (row.get("action") or "").strip()

        line = _format_label_value(symbol, sm)
        if line:
            spiritual_parts.append(line)

        line = _format_label_value(symbol, pe)
        if line:
            physical_parts.append(line)

        line = _format_label_value(symbol, ac)
        if line:
            action_parts.append(line)

    return {
        "spiritual_meaning": "\n".join(spiritual_parts).strip(),
        "effects_in_physical_realm": "\n".join(physical_parts).strip(),
        "what_to_do": "\n".join(action_parts).strip(),
    }


def _make_receipt_id() -> str:
    return f"JTS-{secrets.token_hex(4).upper()}"


def _compute_seal(matches: List[Tuple[Dict, int, Optional[Dict[str, str]]]]) -> Dict[str, str]:
    if not matches:
        return {"status": "Delayed", "type": "Unclear", "risk": "High"}

    scores = [sc for _, sc, _ in matches]
    avg = sum(scores) / max(len(scores), 1)

    if avg >= 95:
        return {"status": "Live", "type": "Confirmed", "risk": "Low"}
    if avg >= 88:
        return {"status": "Delayed", "type": "Processing", "risk": "Medium"}
    return {"status": "Delayed", "type": "Processing", "risk": "High"}


def _dedupe_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        x = _clean_sentence(x)
        if not x:
            continue
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def _extract_symbol_fields(matches: List[Tuple[Dict, int, Optional[Dict[str, str]]]], max_n: int = 3):
    symbols: List[str] = []
    meanings: List[str] = []
    effects: List[str] = []
    actions: List[str] = []

    for row, _sc, _hit in matches[:max_n]:
        symbol = (row.get("input") or row.get("symbol") or "").strip()
        sm = (row.get("spiritual meaning") or row.get("spiritual_meaning") or "").strip()
        pe = (row.get("physical effects") or row.get("physical_effects") or "").strip()
        ac = (row.get("action") or "").strip()

        if symbol:
            symbols.append(symbol)
            meanings.append(sm)
            effects.append(pe)
            actions.append(ac)

    return symbols, meanings, effects, actions


def build_full_interpretation_from_doctrine(matches: List[Tuple[Dict, int, Optional[Dict[str, str]]]]) -> str:
    if not matches or not NARRATIVE_ENABLED:
        return ""

    symbols, meanings, effects, actions = _extract_symbol_fields(matches, max_n=NARRATIVE_MAX_SYMBOLS)

    opening = (
        "This dream is revealing a spiritual condition, not predicting physical harm. "
        "Dreams speak in symbols, and meaning is shown through what appears and what happens."
    )

    symbol_sentences: List[str] = []
    for sym, mean in zip(symbols, meanings):
        mean_clean = _strip_trailing_punct(mean)
        if sym and mean_clean:
            symbol_sentences.append(_ensure_terminal_punct(f"{_title_case_symbol(sym)} points to {mean_clean}"))

    effect_sentences: List[str] = []
    for sym, eff in zip(symbols, effects):
        eff_clean = _strip_trailing_punct(eff)
        if sym and eff_clean:
            effect_sentences.append(_ensure_terminal_punct(
                f"{_title_case_symbol(sym)} can show up as {eff_clean} in the natural realm"
            ))

    action_lines = _dedupe_preserve_order([_strip_trailing_punct(a) for a in actions if a])
    action_text = ""
    if action_lines:
        if len(action_lines) == 1:
            action_text = _ensure_terminal_punct(f"What to do: {action_lines[0]}")
        else:
            joined = ". Then, ".join([_strip_trailing_punct(a) for a in action_lines])
            action_text = _ensure_terminal_punct(f"What to do: {joined}")

    closing = (
        "Dreams expose what needs attention so you can respond. "
        "This is a warning with mercy, not a sentence."
    )

    parts: List[str] = []
    parts.append(_ensure_terminal_punct(opening))
    if symbol_sentences:
        parts.append(" ".join(symbol_sentences).strip())
    if effect_sentences:
        parts.append(" ".join(effect_sentences).strip())
    if action_text:
        parts.append(action_text.strip())
    parts.append(_ensure_terminal_punct(closing))

    return "\n\n".join([_clean_sentence(p).strip() for p in parts if p.strip()]).strip()


def _clean_values_table(headers_raw: List[str], data_rows: List[List[str]]) -> Tuple[List[str], List[List[str]], Dict[str, int]]:
    headers_norm = [_normalize_header(h) for h in headers_raw]
    required = {"input", "spiritual meaning", "physical effects", "action", "keywords"}
    missing = [h for h in required if h not in set(headers_norm)]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    idx = {h: headers_norm.index(h) for h in required}

    cleaned_rows: List[List[str]] = []
    changes = {
        "rows_in": len(data_rows),
        "rows_out": 0,
        "rows_dropped_blank_input": 0,
        "cells_changed": 0,
        "keywords_normalized": 0,
        "typos_fixed": 0,
    }

    for row in data_rows:
        if len(row) < len(headers_norm):
            row = row + [""] * (len(headers_norm) - len(row))
        row = [("" if v is None else str(v)) for v in row]

        for col in ["input", "spiritual meaning", "physical effects", "action"]:
            i = idx[col]
            before = row[i]
            after = _fix_typos_light(before)
            if after != before:
                changes["cells_changed"] += 1
                if re.search(r"(embarass|commmun|comunity|commnity)", before or "", re.IGNORECASE):
                    changes["typos_fixed"] += 1
            row[i] = after

        kw_i = idx["keywords"]
        before_kw = row[kw_i]
        after_kw = _normalize_keywords_cell(before_kw)
        if after_kw != before_kw:
            changes["cells_changed"] += 1
            changes["keywords_normalized"] += 1
        row[kw_i] = after_kw

        if not str(row[idx["input"]]).strip():
            changes["rows_dropped_blank_input"] += 1
            continue

        cleaned_rows.append(row)

    changes["rows_out"] = len(cleaned_rows)
    return headers_raw, cleaned_rows, changes


# ----------------------------
# NEW: Dictionary automation helpers
# ----------------------------
def _get_sheet_table(ws):
    values = ws.get_all_values()
    if not values:
        return [], []
    headers_raw = values[0]
    data_rows = values[1:] if len(values) > 1 else []
    return headers_raw, data_rows


def _required_idx_from_headers(headers_raw: List[str]) -> Dict[str, int]:
    headers_norm = [_normalize_header(h) for h in headers_raw]
    required = {"input", "spiritual meaning", "physical effects", "action", "keywords"}
    missing = [h for h in required if h not in set(headers_norm)]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return {h: headers_norm.index(h) for h in required}


def _make_row_from_payload(payload: Dict[str, Any], headers_raw: List[str]) -> List[str]:
    idx = _required_idx_from_headers(headers_raw)

    inp = (payload.get("input") or payload.get("symbol") or "").strip()
    sm = (payload.get("spiritual meaning") or payload.get("spiritual_meaning") or "").strip()
    pe = (payload.get("physical effects") or payload.get("physical_effects") or "").strip()
    ac = (payload.get("action") or "").strip()
    kw = (payload.get("keywords") or "").strip()

    inp = _fix_typos_light(inp)
    sm = _fix_typos_light(sm)
    pe = _fix_typos_light(pe)
    ac = _fix_typos_light(ac)
    kw = _normalize_keywords_cell(kw)

    if not inp:
        raise ValueError("Missing required field: input")
    if not sm:
        raise ValueError("Missing required field: spiritual_meaning")
    if not pe:
        raise ValueError("Missing required field: physical_effects")
    if not ac:
        raise ValueError("Missing required field: action")

    row = [""] * len(headers_raw)
    row[idx["input"]] = inp
    row[idx["spiritual meaning"]] = sm
    row[idx["physical effects"]] = pe
    row[idx["action"]] = ac
    row[idx["keywords"]] = kw
    return row


def _find_existing_row_index(data_rows: List[List[str]], headers_raw: List[str], input_value: str) -> Optional[int]:
    idx = _required_idx_from_headers(headers_raw)
    target = _normalize_text(input_value)
    for i, r in enumerate(data_rows):
        if len(r) <= idx["input"]:
            continue
        if _normalize_text(r[idx["input"]] or "") == target:
            return i
    return None


# ----------------------------
# Routes
# ----------------------------
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "ok": True,
        "service": "dream-interpreter",
        "sheet": WORKSHEET_NAME,
        "has_spreadsheet_id": bool(SPREADSHEET_ID),
        "allowed_origins": allowed_origins,
        "debug_match": DEBUG_MATCH,
        "narrative_enabled": NARRATIVE_ENABLED,
        "narrative_max_symbols": NARRATIVE_MAX_SYMBOLS,
        "admin_key_set": bool(ADMIN_KEY),
        "match_mode": "strict_word_boundary_only_with_compound_priority_plus_subsumption",
        "build": "2.9.6",
    })


@app.route("/healthz", methods=["GET"])
def healthz():
    return health()


@app.route("/interpret", methods=["POST", "OPTIONS"])
def interpret():
    if request.method == "OPTIONS":
        return _preflight_ok()

    data = request.get_json(silent=True) or {}
    dream = (data.get("dream") or data.get("text") or "").strip()
    if not dream:
        return jsonify({"error": "Missing 'dream' or 'text'"}), 400

    try:
        rows = _load_sheet_rows()
    except Exception as e:
        return jsonify({"error": "Sheet load failed", "details": str(e)}), 500

    matches = _match_symbols_strict(dream, rows, top_k=8)
    matches = _prune_subsumed_matches(matches)
    matches = matches[:3]

    receipt_id = _make_receipt_id()
    seal = _compute_seal(matches)

    if not matches:
        return jsonify({
            "access": "free",
            "is_paid": False,
            "free_uses_left": 3,
            "seal": seal,
            "receipt": {"id": receipt_id, "top_symbols": [], "share_phrase": "I decoded my dream on Jamaican True Stories."},
            "interpretation": {
                "spiritual_meaning": "No matching symbols were found for the exact words in this dream.",
                "effects_in_physical_realm": "Tip: use clear symbol words (people, animals, places, objects) that exist in your Symbols sheet.",
                "what_to_do": "Add 1–2 more key symbols (objects, people, animals, places) and try again."
            },
            "full_interpretation": ""
        })

    interpretation = _combine_fields(matches)

    top_symbols = [
        (row.get("input") or row.get("symbol") or "").strip()
        for row, _sc, _hit in matches
        if (row.get("input") or row.get("symbol") or "").strip()
    ]

    share_phrase = f"My dream had symbols like: {', '.join(top_symbols[:3])}. I decoded it on Jamaican True Stories."
    full_interpretation = build_full_interpretation_from_doctrine(matches)

    return jsonify({
        "access": "free",
        "is_paid": False,
        "free_uses_left": 3,
        "seal": seal,
        "interpretation": interpretation,
        "full_interpretation": full_interpretation,
        "receipt": {"id": receipt_id, "top_symbols": top_symbols, "share_phrase": share_phrase},
    })


@app.route("/track", methods=["POST", "OPTIONS"])
def track():
    if request.method == "OPTIONS":
        return _preflight_ok()

    payload = request.get_json(silent=True) or {}
    return jsonify({
        "ok": True,
        "event": payload.get("event", "unknown"),
        "free_uses_left": payload.get("free_uses_left", 3),
    })


# ----------------------------
# Admin Panel UI (phone-friendly)
# ----------------------------
@app.route("/admin", methods=["GET"])
def admin_panel():
    if not _admin_ok(request):
        return make_response("Unauthorized", 401)

    html = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>JTS Admin Panel</title>
  <style>
    body{margin:0;font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;background:#0b0b0e;color:#f1f1f1;padding:16px}
    .wrap{max-width:860px;margin:0 auto}
    .card{background:#121218;border:1px solid rgba(212,175,55,.25);border-radius:16px;padding:16px;margin-bottom:14px}
    h1{margin:0 0 10px;font-size:18px;color:#d4af37}
    h2{margin:0 0 10px;font-size:15px;color:#f3e28a}
    label{display:block;margin:10px 0 6px;color:#cfcfda;font-size:13px}
    input,textarea,select{width:100%;box-sizing:border-box;background:#0e0e14;color:#fff;border:1px solid rgba(212,175,55,.25);border-radius:12px;padding:10px 12px;font-size:14px;outline:none}
    textarea{min-height:88px;resize:vertical}
    .btn{margin-top:12px;width:100%;background:linear-gradient(90deg,#d4af37,#f3e28a);color:#111;border:0;border-radius:999px;padding:12px 14px;font-weight:800;cursor:pointer}
    .btn2{margin-top:10px;width:100%;background:transparent;color:#f3e28a;border:1px solid rgba(212,175,55,.6);border-radius:999px;padding:10px 12px;font-weight:700;cursor:pointer}
    .out{margin-top:10px;white-space:pre-wrap;background:#0e0e14;border:1px solid rgba(212,175,55,.18);border-radius:12px;padding:10px 12px;font-size:13px;color:#dcdcf4}
    .muted{color:#b7b7c2;font-size:12px;line-height:1.5}
    a{color:#d4af37}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h1>JTS Admin Panel (Build 2.9.6)</h1>
      <div class="muted">Keep this link private. Health: <a href="/health" target="_blank">/health</a></div>
    </div>

    <div class="card">
      <h2>Add / Update ONE Symbol</h2>
      <label>Mode</label>
      <select id="mode">
        <option value="append">append (error if exists)</option>
        <option value="upsert" selected>upsert (update if exists)</option>
      </select>

      <label>Input (symbol phrase)</label>
      <input id="input" placeholder="e.g., hair falling out"/>

      <label>Spiritual Meaning</label>
      <textarea id="sm" placeholder="Doctrine meaning only. No extra guessing."></textarea>

      <label>Physical Effects</label>
      <textarea id="pe" placeholder="How it shows up in the natural realm."></textarea>

      <label>What to Do (Action)</label>
      <textarea id="ac" placeholder="Clear instruction (prayer + practical steps)."></textarea>

      <label>Keywords (comma-separated)</label>
      <textarea id="kw" placeholder="e.g., hair falling out, shedding, bald spots, clumps"></textarea>

      <button class="btn" id="saveOne">Save Symbol</button>
      <div class="out" id="outOne" style="display:none;"></div>
    </div>

    <div class="card">
      <h2>Batch Add Symbols (JSON)</h2>
      <div class="muted">Paste an array under <code>rows</code>. Each row needs: input, spiritual_meaning, physical_effects, action, keywords.</div>

      <label>Mode</label>
      <select id="modeBatch">
        <option value="append">append (skip existing)</option>
        <option value="upsert" selected>upsert (update existing)</option>
      </select>

      <label>Rows JSON</label>
      <textarea id="batchJson" placeholder='{"rows":[{"input":"hair","spiritual_meaning":"Problems.","physical_effects":"Stress.","action":"Pray.","keywords":"hair"}]}'></textarea>

      <button class="btn2" id="saveBatch">Batch Add</button>
      <div class="out" id="outBatch" style="display:none;"></div>
    </div>

    <div class="card">
      <h2>Optimize Dictionary (formatting-only)</h2>
      <div class="muted">Fixes typos, trims spacing, normalizes keywords, removes blank inputs, removes duplicate inputs. Does NOT rewrite meanings.</div>
      <button class="btn2" id="optimize">Run Optimize</button>
      <div class="out" id="outOpt" style="display:none;"></div>
    </div>
  </div>

<script>
  const ADMIN_KEY = new URLSearchParams(window.location.search).get("key") || "";

  async function post(path, payload) {
    const res = await fetch(path, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-Admin-Key": ADMIN_KEY
      },
      body: payload ? JSON.stringify(payload) : null
    });
    const txt = await res.text();
    let data;
    try { data = JSON.parse(txt); } catch { data = { raw: txt }; }
    if (!res.ok) throw new Error(JSON.stringify(data, null, 2));
    return data;
  }

  function show(el, obj) {
    el.style.display = "block";
    el.textContent = JSON.stringify(obj, null, 2);
  }

  document.getElementById("saveOne").addEventListener("click", async () => {
    const out = document.getElementById("outOne");
    out.style.display = "none";
    const payload = {
      mode: document.getElementById("mode").value,
      input: document.getElementById("input").value.trim(),
      spiritual_meaning: document.getElementById("sm").value.trim(),
      physical_effects: document.getElementById("pe").value.trim(),
      action: document.getElementById("ac").value.trim(),
      keywords: document.getElementById("kw").value.trim()
    };
    try {
      const data = await post("/admin/add-symbol", payload);
      show(out, data);
    } catch (e) {
      show(out, { error: String(e.message || e) });
    }
  });

  document.getElementById("saveBatch").addEventListener("click", async () => {
    const out = document.getElementById("outBatch");
    out.style.display = "none";
    let payload = {};
    try {
      payload = JSON.parse(document.getElementById("batchJson").value || "{}");
    } catch (e) {
      show(out, { error: "Invalid JSON" });
      return;
    }
    payload.mode = document.getElementById("modeBatch").value;

    try {
      const data = await post("/admin/batch-add", payload);
      show(out, data);
    } catch (e) {
      show(out, { error: String(e.message || e) });
    }
  });

  document.getElementById("optimize").addEventListener("click", async () => {
    const out = document.getElementById("outOpt");
    out.style.display = "none";
    try {
      const data = await post("/admin/optimize-dictionary", {});
      show(out, data);
    } catch (e) {
      show(out, { error: String(e.message || e) });
    }
  });
</script>
</body>
</html>
"""
    return make_response(html, 200)


# ----------------------------
# Admin Dictionary Automation Routes
# ----------------------------
@app.route("/admin/add-symbol", methods=["POST", "OPTIONS"])
def admin_add_symbol():
    if request.method == "OPTIONS":
        return _preflight_ok()

    if not _admin_ok(request):
        return jsonify({"error": "Unauthorized"}), 401

    payload = request.get_json(silent=True) or {}
    mode = (payload.get("mode") or "append").strip().lower()  # append | upsert
    if mode not in {"append", "upsert"}:
        mode = "append"

    try:
        ws = _get_ws()
        headers_raw, data_rows = _get_sheet_table(ws)
        if not headers_raw:
            return jsonify({"error": "Sheet has no header row."}), 400

        new_row = _make_row_from_payload(payload, headers_raw)
        idx = _required_idx_from_headers(headers_raw)
        new_input = new_row[idx["input"]]

        existing_idx = _find_existing_row_index(data_rows, headers_raw, new_input)

        if existing_idx is not None and mode != "upsert":
            return jsonify({"error": "Symbol already exists. Use mode=upsert to overwrite.", "input": new_input}), 409

        if existing_idx is not None and mode == "upsert":
            data_rows[existing_idx] = new_row
            ws.update("A2", data_rows)
            action = "updated"
        else:
            ws.append_row(new_row, value_input_option="RAW")
            action = "added"

        _load_sheet_rows(force=True)
        return jsonify({"ok": True, "action": action, "input": new_input})

    except ValueError as ve:
        return jsonify({"error": "Validation failed", "details": str(ve)}), 400
    except Exception as e:
        return jsonify({"error": "Add-symbol failed", "details": str(e)}), 500


@app.route("/admin/batch-add", methods=["POST", "OPTIONS"])
def admin_batch_add():
    if request.method == "OPTIONS":
        return _preflight_ok()

    if not _admin_ok(request):
        return jsonify({"error": "Unauthorized"}), 401

    payload = request.get_json(silent=True) or {}
    rows_payload = payload.get("rows") or []
    mode = (payload.get("mode") or "append").strip().lower()  # append | upsert
    if mode not in {"append", "upsert"}:
        mode = "append"

    if not isinstance(rows_payload, list) or not rows_payload:
        return jsonify({"error": "Provide JSON with a 'rows' array."}), 400

    try:
        ws = _get_ws()
        headers_raw, data_rows = _get_sheet_table(ws)
        if not headers_raw:
            return jsonify({"error": "Sheet has no header row."}), 400

        idx = _required_idx_from_headers(headers_raw)

        existing_map = {}
        for i, r in enumerate(data_rows):
            if len(r) > idx["input"]:
                existing_map[_normalize_text(r[idx["input"]] or "")] = i

        added = 0
        updated = 0
        skipped = 0
        errors = []
        to_append = []

        for n, row_obj in enumerate(rows_payload, start=1):
            try:
                new_row = _make_row_from_payload(row_obj, headers_raw)
                key = _normalize_text(new_row[idx["input"]])
                if key in existing_map:
                    if mode == "upsert":
                        data_rows[existing_map[key]] = new_row
                        updated += 1
                    else:
                        skipped += 1
                else:
                    to_append.append(new_row)
                    added += 1
            except Exception as ex:
                errors.append({"row": n, "error": str(ex)})

        if updated:
            ws.update("A2", data_rows)

        if to_append:
            ws.append_rows(to_append, value_input_option="RAW")

        _load_sheet_rows(force=True)

        return jsonify({
            "ok": True,
            "mode": mode,
            "added": added,
            "updated": updated,
            "skipped": skipped,
            "errors": errors[:50],
        })

    except ValueError as ve:
        return jsonify({"error": "Validation failed", "details": str(ve)}), 400
    except Exception as e:
        return jsonify({"error": "Batch-add failed", "details": str(e)}), 500


@app.route("/admin/optimize-dictionary", methods=["POST", "OPTIONS"])
def admin_optimize_dictionary():
    if request.method == "OPTIONS":
        return _preflight_ok()

    if not _admin_ok(request):
        return jsonify({"error": "Unauthorized"}), 401

    try:
        ws = _get_ws()
        values = ws.get_all_values()
        if not values or len(values) < 2:
            return jsonify({"ok": True, "message": "Sheet empty. Nothing to optimize."})

        headers_raw = values[0]
        data_rows = values[1:]

        headers_raw, cleaned_rows, changes = _clean_values_table(headers_raw, data_rows)

        idx = _required_idx_from_headers(headers_raw)
        seen = set()
        deduped = []
        dropped_dupes = 0
        for r in cleaned_rows:
            key = _normalize_text(r[idx["input"]] if len(r) > idx["input"] else "")
            if not key:
                continue
            if key in seen:
                dropped_dupes += 1
                continue
            seen.add(key)
            deduped.append(r)

        changes["rows_dropped_duplicate_input"] = dropped_dupes
        changes["rows_out"] = len(deduped)

        ws.clear()
        ws.update("A1", [headers_raw] + deduped)

        _load_sheet_rows(force=True)

        return jsonify({"ok": True, "message": "Dictionary optimized (formatting-only).", "changes": changes})

    except Exception as e:
        return jsonify({"error": "Optimize-dictionary failed", "details": str(e)}), 500


# -------- EXISTING ADMIN ROUTES (kept) --------
@app.route("/admin/clean-sheet", methods=["POST", "OPTIONS"])
def admin_clean_sheet():
    if request.method == "OPTIONS":
        return _preflight_ok()

    if not _admin_ok(request):
        return jsonify({"error": "Unauthorized"}), 401

    try:
        ws = _get_ws()
        values = ws.get_all_values()
        if not values or len(values) < 2:
            return jsonify({"ok": True, "message": "Sheet is empty. Nothing to clean."})

        headers_raw = values[0]
        data_rows = values[1:]

        headers_raw, cleaned_rows, changes = _clean_values_table(headers_raw, data_rows)

        ws.clear()
        ws.update("A1", [headers_raw] + cleaned_rows)

        _load_sheet_rows(force=True)

        return jsonify({"ok": True, "message": "Sheet cleaned and updated.", "changes": changes})

    except Exception as e:
        return jsonify({"error": "Clean-sheet failed", "details": str(e)}), 500


@app.route("/admin/import-sheet", methods=["POST", "OPTIONS"])
def admin_import_sheet():
    if request.method == "OPTIONS":
        return _preflight_ok()

    if not _admin_ok(request):
        return jsonify({"error": "Unauthorized"}), 401

    try:
        content_type = (request.headers.get("Content-Type") or "").lower()

        headers_raw: List[str]
        data_rows: List[List[str]]

        if "application/json" in content_type:
            payload = request.get_json(silent=True) or {}
            headers_raw = payload.get("headers") or []
            data_rows = payload.get("rows") or []
            if not isinstance(headers_raw, list) or not isinstance(data_rows, list) or not headers_raw:
                return jsonify({"error": "Invalid JSON. Provide 'headers' list and 'rows' list."}), 400
        else:
            raw = request.get_data(as_text=True) or ""
            raw = raw.strip("\ufeff").strip()
            if not raw:
                return jsonify({"error": "Empty body. Send JSON or CSV."}), 400

            reader = csv.reader(io.StringIO(raw))
            table = list(reader)
            if len(table) < 2:
                return jsonify({"error": "CSV must include header row + at least 1 data row."}), 400

            headers_raw = table[0]
            data_rows = table[1:]

        headers_raw, cleaned_rows, changes = _clean_values_table(headers_raw, data_rows)

        ws = _get_ws()
        ws.clear()
        ws.update("A1", [headers_raw] + cleaned_rows)

        _load_sheet_rows(force=True)

        return jsonify({
            "ok": True,
            "message": "Sheet imported + cleaned + updated.",
            "changes": changes,
            "rows_written": len(cleaned_rows),
            "columns_written": len(headers_raw),
        })

    except ValueError as ve:
        return jsonify({"error": "Import validation failed", "details": str(ve)}), 400
    except Exception as e:
        return jsonify({"error": "Import failed", "details": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=True)
