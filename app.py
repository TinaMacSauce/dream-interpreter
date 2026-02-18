# app.py — Jamaican True Stories Dream Interpreter (Phase 2.9.3 — QUICK DECODE + FULL INTERPRETATION + DEBUG + ADMIN)
# What this version includes:
# - ✅ Interpreter API: /interpret, /track
# - ✅ Health: /health and /healthz (Render-friendly)
# - ✅ Debug: /debug/config and /debug/sheet (guarded by DEBUG_MATCH=1)
# - ✅ JTS Admin Panel (RESTORED): /admin (HTML) + /admin/upsert (writes to Google Sheet)
# - ✅ Robust header normalization + flexible field getters
# - ✅ Cache invalidation after admin writes (so new symbols show up fast)
#
# IMPORTANT:
# - Your old code had read-only scopes; Admin needs WRITE scopes to update Google Sheets.

import os
import json
import time
import re
import secrets
from typing import Dict, List, Tuple, Any, Optional

from flask import Flask, request, jsonify, make_response, Response
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
    allow_headers=["Content-Type", "Authorization"],
    methods=["GET", "POST", "OPTIONS"],
)

SPREADSHEET_ID = os.getenv("SPREADSHEET_ID", "").strip()
WORKSHEET_NAME = os.getenv("WORKSHEET_NAME", "Sheet1").strip()

# Admin key (you can name it either way)
ADMIN_KEY = (
    os.getenv("JTS_ADMIN_KEY", "").strip()
    or os.getenv("ADMIN_KEY", "").strip()
    or os.getenv("JTS_ADMIN", "").strip()
)

# ✅ WRITE scopes needed for Admin to edit the sheet
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

CACHE_TTL_SECONDS = int(os.getenv("SHEET_CACHE_TTL", "120"))
_CACHE: Dict[str, Any] = {"loaded_at": 0.0, "rows": [], "headers": []}

DEBUG_MATCH = os.getenv("DEBUG_MATCH", "").strip().lower() in {"1", "true", "yes", "on"}

# Narrative style knobs (optional)
NARRATIVE_ENABLED = os.getenv("NARRATIVE_ENABLED", "1").strip().lower() not in {"0", "false", "no", "off"}
NARRATIVE_MAX_SYMBOLS = int(os.getenv("NARRATIVE_MAX_SYMBOLS", "3"))  # keep short & consistent


# ----------------------------
# Helpers
# ----------------------------
def _preflight_ok():
    return make_response("", 204)


def _normalize_header(h: str) -> str:
    """
    Normalizes sheet headers so real-world names still map correctly.
    Example: "Input (symbol)" -> "input symbol"
    """
    h = (h or "").strip().lower()
    h = re.sub(r"[^a-z0-9\s_]", " ", h)  # punctuation -> spaces
    h = re.sub(r"\s+", " ", h).strip()
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
    """Normalize spacing/punctuation without changing meaning."""
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
    """Formats 'Symbol: Text' cleanly for the Quick Decode boxes."""
    symbol = _title_case_symbol(symbol)
    text = _clean_sentence(text)
    if not symbol or not text:
        return ""
    return f"{symbol}: {text}"


def _invalidate_cache():
    _CACHE["loaded_at"] = 0.0
    _CACHE["rows"] = []
    _CACHE["headers"] = []


# ----------------------------
# Sheet field getters (robust)
# ----------------------------
def _get_symbol_cell(row: Dict) -> str:
    return (
        row.get("input")
        or row.get("symbol")
        or row.get("symbols")
        or row.get("input symbol")
        or row.get("dream symbol")
        or row.get("symbol name")
        or ""
    ).strip()


def _get_spiritual_meaning_cell(row: Dict) -> str:
    return (
        row.get("spiritual meaning")
        or row.get("spiritual_meaning")
        or row.get("spiritual")
        or row.get("meaning")
        or ""
    ).strip()


def _get_effects_cell(row: Dict) -> str:
    return (
        row.get("effects in the physical realm")
        or row.get("effects_in_the_physical_realm")
        or row.get("physical effects")
        or row.get("physical_effects")
        or row.get("effects")
        or ""
    ).strip()


def _get_what_to_do_cell(row: Dict) -> str:
    return (
        row.get("what to do")
        or row.get("what_to_do")
        or row.get("action")
        or row.get("actions")
        or ""
    ).strip()


def _get_keywords_cell(row: Dict) -> str:
    return (row.get("keywords") or row.get("keyword") or row.get("tags") or "").strip()


# ----------------------------
# Google credentials + sheet access
# ----------------------------
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


# ----------------------------
# Matching logic (strict + guarded keywords)
# ----------------------------
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


def _symbol_length_penalty(symbol: str) -> int:
    words = [w for w in _normalize_text(symbol).split() if w]
    n = len(words)
    if n <= 2:
        return 0
    if n == 3:
        return 2
    if n == 4:
        return 6
    if n == 5:
        return 12
    return 18


def _score_row_strict(dream_norm: str, row: Dict) -> Tuple[int, Optional[Dict[str, str]]]:
    symbol_raw = _get_symbol_cell(row)
    if not symbol_raw:
        return 0, None

    symbol = _normalize_text(symbol_raw)
    if not symbol:
        return 0, None

    symbol_words = symbol.split()
    keywords = _split_keywords(_get_keywords_cell(row))

    if len(symbol_words) == 1:
        if _compile_boundary_regex(symbol).search(dream_norm):
            return 100, {"type": "symbol", "token": symbol}
    else:
        phrase_rx = re.compile(rf"(?<!\w){re.escape(symbol)}(?!\w)")
        if phrase_rx.search(dream_norm):
            score = 100 - _symbol_length_penalty(symbol_raw)
            return int(score), {"type": "symbol_phrase", "token": symbol}

    if len(symbol_words) == 1:
        for kw in keywords:
            if kw and _compile_boundary_regex(kw).search(dream_norm):
                return 96, {"type": "keyword", "token": kw}

    return 0, None


def _match_symbols_strict(
    dream: str, rows: List[Dict], top_k: int = 3
) -> List[Tuple[Dict, int, Optional[Dict[str, str]]]]:
    dream_norm = _normalize_text(dream)
    if not dream_norm:
        return []

    _log("\n--- DEBUG_MATCH ON ---")
    _log("DREAM_RAW:", repr(dream))
    _log("DREAM_NORM:", repr(dream_norm))

    scored: List[Tuple[Dict, int, Optional[Dict[str, str]]]] = []
    for row in rows:
        sc, hit = _score_row_strict(dream_norm, row)
        if sc > 0:
            scored.append((row, sc, hit))

    def _sort_key(item: Tuple[Dict, int, Optional[Dict[str, str]]]):
        row, sc, _hit = item
        sym = _get_symbol_cell(row)
        return (-sc, len(_normalize_text(sym)))

    scored.sort(key=_sort_key)

    seen = set()
    out: List[Tuple[Dict, int, Optional[Dict[str, str]]]] = []
    for row, sc, hit in scored:
        sym = _get_symbol_cell(row)
        sym_key = _normalize_text(sym)
        if not sym_key or sym_key in seen:
            continue
        seen.add(sym_key)
        out.append((row, sc, hit))
        if len(out) >= top_k:
            break

    if out:
        _log("MATCHES:")
        for row, sc, hit in out:
            _log(f" - {_get_symbol_cell(row)!r} score={sc} via={hit}")
    else:
        _log("MATCHES: (none)")

    _log("--- END DEBUG_MATCH ---\n")
    return out


# ----------------------------
# Output building
# ----------------------------
def _combine_fields(matches: List[Tuple[Dict, int, Optional[Dict[str, str]]]]) -> Dict[str, str]:
    spiritual_parts: List[str] = []
    physical_parts: List[str] = []
    action_parts: List[str] = []

    for row, _sc, _hit in matches:
        symbol = _get_symbol_cell(row)
        sm = _get_spiritual_meaning_cell(row)
        pe = _get_effects_cell(row)
        ac = _get_what_to_do_cell(row)

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

    avg = sum(sc for _, sc, _ in matches) / max(len(matches), 1)
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
        symbols.append(_get_symbol_cell(row))
        meanings.append(_get_spiritual_meaning_cell(row))
        effects.append(_get_effects_cell(row))
        actions.append(_get_what_to_do_cell(row))

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
            effect_sentences.append(_ensure_terminal_punct(f"{_title_case_symbol(sym)} can show up as {eff_clean} in the natural realm"))

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

    parts = [
        _ensure_terminal_punct(opening),
        " ".join(symbol_sentences).strip() if symbol_sentences else "",
        " ".join(effect_sentences).strip() if effect_sentences else "",
        action_text.strip() if action_text else "",
        _ensure_terminal_punct(closing),
    ]
    return "\n\n".join([_clean_sentence(p).strip() for p in parts if p and p.strip()]).strip()


# ----------------------------
# Admin auth + HTML
# ----------------------------
def _get_admin_key_from_request() -> str:
    # Accept either query string ?key=... OR header X-Admin-Key
    q = (request.args.get("key") or "").strip()
    h = (request.headers.get("X-Admin-Key") or "").strip()
    return q or h


def _require_admin() -> Optional[Response]:
    if not ADMIN_KEY:
        # Safer to fail closed if key isn't configured
        return make_response("Admin is not configured (missing JTS_ADMIN_KEY).", 403)

    provided = _get_admin_key_from_request()
    if not provided or provided != ADMIN_KEY:
        return make_response("Forbidden", 403)
    return None


ADMIN_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>JTS Admin Panel</title>
  <style>
    :root{--bg:#0b0b0e;--panel:#121218;--ink:#e9e9ef;--muted:#b7b7c2;--gold:#d4af37;--gold2:#b8921e;--edge:#2b2730;}
    *{box-sizing:border-box}
    body{margin:0;background:var(--bg);color:var(--ink);font-family:system-ui,Segoe UI,Roboto,Arial,sans-serif;padding:22px;}
    .wrap{max-width:860px;margin:0 auto;}
    .card{background:linear-gradient(180deg, rgba(212,175,55,.05), transparent 160px), var(--panel);
      border:1px solid rgba(212,175,55,.18);border-radius:18px; padding:18px 16px; box-shadow:0 10px 30px rgba(0,0,0,.45);}
    h1{margin:0 0 8px;font-size:18px}
    .mini{color:var(--muted);font-size:13px;margin-bottom:14px}
    label{display:block;margin-top:10px;margin-bottom:6px;color:var(--muted);font-size:13px}
    input, textarea, select{
      width:100%; background:#0e0e14; color:var(--ink);
      border:1px solid rgba(212,175,55,.2); border-radius:12px;
      padding:10px 12px; outline:none;
    }
    textarea{min-height:74px; resize:vertical}
    .btn{
      margin-top:14px; width:100%;
      border:none; border-radius:999px; padding:12px 16px; font-weight:700; cursor:pointer;
      background:linear-gradient(180deg, var(--gold), var(--gold2)); color:#0b0b0e;
    }
    pre{white-space:pre-wrap;background:rgba(0,0,0,.35);padding:12px;border-radius:12px;border:1px solid rgba(212,175,55,.16);margin-top:12px}
    a{color:var(--gold)}
  </style>
</head>
<body>
<div class="wrap">
  <div class="card">
    <h1>JTS Admin Panel</h1>
    <div class="mini">Keep this link private. Health: <a href="/health" target="_blank">/health</a></div>

    <label>Mode</label>
    <select id="mode">
      <option value="upsert">upsert (update if exists)</option>
      <option value="add">add (always append)</option>
    </select>

    <label>Input (symbol phrase)</label>
    <input id="input" placeholder="e.g., teeth falling out" />

    <label>Spiritual Meaning</label>
    <textarea id="spiritual"></textarea>

    <label>Physical Effects</label>
    <textarea id="effects"></textarea>

    <label>What to Do (Action)</label>
    <textarea id="action"></textarea>

    <label>Keywords (comma-separated)</label>
    <textarea id="keywords" placeholder="e.g., teeth, falling, mouth"></textarea>

    <button class="btn" id="save">Save Symbol</button>

    <pre id="out">{}</pre>
  </div>
</div>

<script>
  const out = document.getElementById('out');
  const saveBtn = document.getElementById('save');

  function getKeyFromUrl(){
    const u = new URL(window.location.href);
    return u.searchParams.get('key') || '';
  }

  saveBtn.addEventListener('click', async () => {
    out.textContent = 'Saving...';
    const payload = {
      mode: document.getElementById('mode').value,
      input: document.getElementById('input').value.trim(),
      spiritual_meaning: document.getElementById('spiritual').value.trim(),
      physical_effects: document.getElementById('effects').value.trim(),
      action: document.getElementById('action').value.trim(),
      keywords: document.getElementById('keywords').value.trim(),
    };
    const key = getKeyFromUrl();
    const res = await fetch('/admin/upsert?key=' + encodeURIComponent(key), {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload),
    });
    const data = await res.json().catch(() => ({error:'Bad JSON'}));
    out.textContent = JSON.stringify(data, null, 2);
  });
</script>
</body>
</html>
"""


def _build_col_map(header_row: List[str]) -> Dict[str, int]:
    """
    Returns normalized header -> 1-based column index for sheet updates.
    """
    col_map: Dict[str, int] = {}
    for idx, h in enumerate(header_row, start=1):
        col_map[_normalize_header(h)] = idx
    return col_map


def _find_existing_row_index(ws, input_value: str, input_col: int) -> Optional[int]:
    """
    Finds an existing row by comparing normalized input text.
    Returns 1-based row index, or None.
    """
    target = _normalize_text(input_value)
    if not target:
        return None

    # Read the entire input column
    col_vals = ws.col_values(input_col)  # includes header at row 1
    for i, v in enumerate(col_vals[1:], start=2):  # start=2 because row 1 is header
        if _normalize_text(v) == target:
            return i
    return None


def _admin_upsert_to_sheet(payload: Dict[str, Any]) -> Dict[str, Any]:
    ws = _get_ws()
    header_row = ws.row_values(1)
    if not header_row:
        raise RuntimeError("Sheet has no header row.")

    col_map = _build_col_map(header_row)

    # Accept your known schema names
    input_col = col_map.get("input") or col_map.get("symbol")
    spiritual_col = col_map.get("spiritual meaning") or col_map.get("spiritual_meaning") or col_map.get("spiritual")
    effects_col = col_map.get("physical effects") or col_map.get("physical_effects") or col_map.get("effects in the physical realm")
    action_col = col_map.get("action") or col_map.get("what to do") or col_map.get("what_to_do")
    keywords_col = col_map.get("keywords")

    if not input_col:
        raise RuntimeError("Missing 'input' column in header row.")
    # Other columns can be optional, but you probably want them
    # We'll still allow save even if one is missing.

    mode = (payload.get("mode") or "upsert").strip().lower()
    input_value = (payload.get("input") or "").strip()
    if not input_value:
        return {"ok": False, "error": "Missing input"}

    spiritual = (payload.get("spiritual_meaning") or payload.get("spiritual") or "").strip()
    effects = (payload.get("physical_effects") or payload.get("effects") or "").strip()
    action = (payload.get("action") or "").strip()
    keywords = (payload.get("keywords") or "").strip()

    existing_row = None
    if mode != "add":
        existing_row = _find_existing_row_index(ws, input_value, input_col)

    if existing_row:
        row_index = existing_row
        op = "updated"
    else:
        # Append row at bottom (ensure we provide enough columns)
        row_index = len(ws.get_all_values()) + 1
        op = "added"

    # Write cells (only write columns that exist)
    updates = []
    updates.append((row_index, input_col, input_value))
    if spiritual_col:
        updates.append((row_index, spiritual_col, spiritual))
    if effects_col:
        updates.append((row_index, effects_col, effects))
    if action_col:
        updates.append((row_index, action_col, action))
    if keywords_col:
        updates.append((row_index, keywords_col, keywords))

    # Batch update
    cell_list = []
    for r, c, v in updates:
        cell_list.append(gspread.Cell(r, c, v))
    ws.update_cells(cell_list, value_input_option="RAW")

    # Bust cache so /interpret sees it fast
    _invalidate_cache()

    return {
        "ok": True,
        "action": op,
        "input": input_value,
        "written_row": row_index,
        "spreadsheet_id": SPREADSHEET_ID,
        "worksheet_name": WORKSHEET_NAME,
    }


# ----------------------------
# Routes
# ----------------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "ok": True,
        "service": "dream-interpreter",
        "sheet": WORKSHEET_NAME,
        "has_spreadsheet_id": bool(SPREADSHEET_ID),
        "allowed_origins": allowed_origins,
        "match_mode": "strict_word_boundary_only_with_compound_guard",
        "debug_match": DEBUG_MATCH,
        "narrative_enabled": NARRATIVE_ENABLED,
        "narrative_max_symbols": NARRATIVE_MAX_SYMBOLS,
        "admin_configured": bool(ADMIN_KEY),
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

    matches = _match_symbols_strict(dream, rows, top_k=3)

    receipt_id = _make_receipt_id()
    seal = _compute_seal(matches)

    if not matches:
        return jsonify({
            "access": "free",
            "is_paid": False,
            "free_uses_left": 3,
            "seal": seal,
            "receipt": {
                "id": receipt_id,
                "top_symbols": [],
                "share_phrase": "I decoded my dream on Jamaican True Stories."
            },
            "interpretation": {
                "spiritual_meaning": "No matching symbols were found for the exact words in this dream.",
                "effects_in_physical_realm": "Tip: use clear symbol words (people, animals, places, objects) that exist in your Symbols sheet.",
                "what_to_do": "Add 1–2 more key symbols (objects, people, animals, places) and try again."
            },
            "full_interpretation": ""
        })

    interpretation = _combine_fields(matches)
    top_symbols = [_get_symbol_cell(row) for row, _sc, _hit in matches if _get_symbol_cell(row)]
    share_phrase = f"My dream had symbols like: {', '.join(top_symbols[:3])}. I decoded it on Jamaican True Stories."
    full_interpretation = build_full_interpretation_from_doctrine(matches)

    return jsonify({
        "access": "free",
        "is_paid": False,
        "free_uses_left": 3,
        "seal": seal,
        "interpretation": interpretation,
        "full_interpretation": full_interpretation,
        "receipt": {
            "id": receipt_id,
            "top_symbols": top_symbols,
            "share_phrase": share_phrase
        },
    })


@app.route("/track", methods=["POST", "OPTIONS"])
def track():
    if request.method == "OPTIONS":
        return _preflight_ok()

    payload = request.get_json(silent=True) or {}
    event_name = payload.get("event") or payload.get("event_type") or "unknown"
    return jsonify({"ok": True, "event": event_name, "free_uses_left": payload.get("free_uses_left", 3)})


# ----------------------------
# Admin routes (RESTORED)
# ----------------------------
@app.route("/admin", methods=["GET"])
def admin():
    auth_fail = _require_admin()
    if auth_fail:
        return auth_fail
    return Response(ADMIN_HTML, mimetype="text/html")


@app.route("/admin/upsert", methods=["POST", "OPTIONS"])
def admin_upsert():
    if request.method == "OPTIONS":
        return _preflight_ok()

    auth_fail = _require_admin()
    if auth_fail:
        return jsonify({"ok": False, "error": "Forbidden"}), 403

    payload = request.get_json(silent=True) or {}
    try:
        result = _admin_upsert_to_sheet(payload)
        return jsonify(result)
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


# ----------------------------
# Debug routes (guarded)
# ----------------------------
@app.route("/debug/config", methods=["GET"])
def debug_config():
    if not DEBUG_MATCH:
        return jsonify({"error": "Debug disabled"}), 403
    return jsonify({
        "spreadsheet_id": SPREADSHEET_ID,
        "worksheet_name": WORKSHEET_NAME,
        "cache_ttl_seconds": CACHE_TTL_SECONDS,
        "allowed_origins": allowed_origins,
        "admin_configured": bool(ADMIN_KEY),
    })


@app.route("/debug/sheet", methods=["GET"])
def debug_sheet():
    if not DEBUG_MATCH:
        return jsonify({"error": "Debug disabled"}), 403

    rows = _load_sheet_rows(force=True)
    headers = _CACHE.get("headers", [])
    sample = rows[0] if rows else {}

    return jsonify({
        "worksheet": WORKSHEET_NAME,
        "row_count": len(rows),
        "headers_seen": headers,
        "sample_keys": list(sample.keys()),
        "sample_symbol_cell": _get_symbol_cell(sample),
        "sample_spiritual_meaning": _get_spiritual_meaning_cell(sample),
        "sample_effects": _get_effects_cell(sample),
        "sample_what_to_do": _get_what_to_do_cell(sample),
        "sample_keywords": _get_keywords_cell(sample),
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=True)
