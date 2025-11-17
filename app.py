# app.py — Dream Interpreter with paragraph decoding, free tier, and Stripe paywall

import os, re, json, logging
from typing import Dict, List
from flask import Flask, request, jsonify
from flask_cors import CORS
import gspread
from google.oauth2.service_account import Credentials
import stripe  # Stripe library

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ── Environment ─────────────────────────────────────────────────────
SHEET_ID       = os.environ.get("SHEET_ID", "").strip()              # must be set
WORKSHEET_NAME = os.environ.get("WORKSHEET_NAME", "Sheet1").strip()
CREDS_PATH     = os.environ.get("CREDS_PATH", "credentials.json").strip()

FREE_QUOTA     = int(os.environ.get("FREE_QUOTA", "20"))
COUNTS_FILE    = os.environ.get("COUNTS_FILE", "usage_counts.json")

STRIPE_SECRET_KEY = os.environ.get("STRIPE_SECRET_KEY", "").strip()
PRICE_WEEKLY      = os.environ.get("PRICE_WEEKLY", "").strip()
PRICE_MONTHLY     = os.environ.get("PRICE_MONTHLY", "").strip()

if STRIPE_SECRET_KEY:
    stripe.api_key = STRIPE_SECRET_KEY

SUBSCRIBERS_FILE = os.environ.get("SUBSCRIBERS_FILE", "subscribers.json")

DISCLAIMER = (
    "Reflective and educational guidance based on cultural symbolism—"
    "no divination, no predictions, and no medical or legal advice."
)
NOTE_TEXT = "Based on traditional symbolism, here is a reflective interpretation:"

# ── Globals (cache) ─────────────────────────────────────────────────
LUT: Dict[str, str] = {}        # 'input' -> 'output' (kept for backward compatibility)
INPUTS: List[str] = []          # list of input phrases (lowercased)
SHEET_ROWS: List[Dict] = []     # full rows including keywords


# ── Google Sheets helpers ───────────────────────────────────────────
def _gs_client():
    scopes = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
    creds  = Credentials.from_service_account_file(CREDS_PATH, scopes=scopes)
    return gspread.authorize(creds)


def load_sheet_rows() -> List[Dict]:
    """
    Read rows from WORKSHEET_NAME.

    Expected headers in the Google Sheet:
      input | output | keywords
    """
    if not SHEET_ID:
        raise RuntimeError("Missing SHEET_ID environment variable.")
    gc = _gs_client()
    ws = gc.open_by_key(SHEET_ID).worksheet(WORKSHEET_NAME)
    rows = ws.get_all_records()
    return rows


def refresh_cache():
    """
    Load Google Sheet into in-memory caches:
      - SHEET_ROWS: full dict rows
      - INPUTS / LUT for quick access
    """
    global INPUTS, LUT, SHEET_ROWS
    SHEET_ROWS = load_sheet_rows()

    lut: Dict[str, str] = {}
    inputs: List[str] = []

    for r in SHEET_ROWS:
        k = (r.get("input") or "").strip().lower()
        v = (r.get("output") or "").strip()
        if k:
            lut[k] = v
            inputs.append(k)

    LUT = lut
    INPUTS = inputs
    logging.info("Loaded %d rows from %s", len(INPUTS), WORKSHEET_NAME)


# ── Usage counter (local JSON) ──────────────────────────────────────
def _load_counts() -> Dict[str, int]:
    try:
        with open(COUNTS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def _save_counts(counts: Dict[str, int]) -> None:
    try:
        with open(COUNTS_FILE, "w", encoding="utf-8") as f:
            json.dump(counts, f)
    except Exception:
        pass

def get_identifier(email: str | None) -> str:
    if email:
        return email.strip().lower()
    return f"ip:{request.remote_addr}"

def get_count(identifier: str) -> int:
    return _load_counts().get(identifier, 0)

def bump_count(identifier: str) -> int:
    counts = _load_counts()
    counts[identifier] = counts.get(identifier, 0) + 1
    _save_counts(counts)
    return counts[identifier]


# ── PRO subscribers JSON ────────────────────────────────────────────
def _load_subscribers() -> Dict[str, str]:
    try:
        with open(SUBSCRIBERS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def _save_subscribers(d: Dict[str, str]) -> None:
    try:
        with open(SUBSCRIBERS_FILE, "w", encoding="utf-8") as f:
            json.dump(d, f)
    except Exception:
        pass

def is_pro(email: str | None) -> bool:
    if not email:
        return False
    subs = _load_subscribers()
    return subs.get(email.strip().lower()) == "pro"

def make_pro(email: str) -> None:
    subs = _load_subscribers()
    subs[email.strip().lower()] = "pro"
    _save_subscribers(subs)


# ── Matching (using input + keywords) ───────────────────────────────

def find_best_rows_for_dream(dream_text: str, max_hits: int = 8) -> List[Dict]:
    """
    Use 'input' and 'keywords' columns to find the best matching row(s).

    Rules:
      1. If the user types exactly the symbol (input), return that row only.
      2. Otherwise, score each row based on keyword matches:
         - multi-word keyword inside dream → +3
         - single keyword inside dream    → +1
      3. Return all rows whose score == highest_score (ties allowed, capped by max_hits).
    """
    dream = (dream_text or "").lower().strip()
    if not dream:
        return []

    exact_matches: List[Dict] = []
    keyword_matches: List[tuple[int, str, Dict]] = []

    for row in SHEET_ROWS:
        symbol = str(row.get("input", "")).strip().lower()
        keywords_str = str(row.get("keywords", "")).lower()
        keywords = [k.strip() for k in keywords_str.split(",") if k.strip()]

        # 1) Exact symbol match
        if dream == symbol:
            exact_matches.append(row)
            continue

        # 2) Keyword scoring
        score = 0
        for kw in keywords:
            if not kw:
                continue
            if " " in kw and kw in dream:
                score += 3      # multi-word phrase, more specific
            elif kw in dream:
                score += 1      # single word

        if score > 0:
            keyword_matches.append((score, symbol, row))

    # Exact match wins immediately
    if exact_matches:
        return exact_matches[:max_hits]

    # Otherwise, pick rows with highest keyword score
    if keyword_matches:
        keyword_matches.sort(key=lambda x: x[0], reverse=True)
        top_score = keyword_matches[0][0]
        best_rows = [row for score, sym, row in keyword_matches
                     if score == top_score][:max_hits]
        return best_rows

    return []


# ── Parse "output" into three sections ──────────────────────────────

def split_output_to_sections(output_text: str) -> Dict[str, str]:
    """
    Your 'output' column is stored as a single string like:
      "Spiritual Meaning: ... Effects in the Physical Realm: ... What to Do: ..."

    This splits it into:
      spiritual_meaning
      effects_in_the_physical_realm
      what_to_do
    """
    if not output_text:
        return {
            "spiritual_meaning": "",
            "effects_in_the_physical_realm": "",
            "what_to_do": "",
        }

    # Normalize whitespace
    text = " ".join(str(output_text).split())

    sm_label  = "spiritual meaning:"
    ef_label  = "effects in the physical realm:"
    wtd_label = "what to do:"

    lower = text.lower()
    sm_idx  = lower.find(sm_label)
    ef_idx  = lower.find(ef_label)
    wtd_idx = lower.find(wtd_label)

    def slice_section(label: str, start_idx: int, end_idx: int) -> str:
        if start_idx == -1:
            return ""
        if end_idx == -1:
            section = text[start_idx + len(label):]
        else:
            section = text[start_idx + len(label):end_idx]
        return section.strip(" .")

    sm  = slice_section(sm_label,  sm_idx,  ef_idx if ef_idx != -1 else wtd_idx)
    ef  = slice_section(ef_label,  ef_idx,  wtd_idx)
    wtd = slice_section(wtd_label, wtd_idx, -1)

    return {
        "spiritual_meaning": sm,
        "effects_in_the_physical_realm": ef,
        "what_to_do": wtd,
    }


def compose_output(rows: List[Dict]) -> Dict[str, object]:
    """
    Turn matched sheet rows into the JSON structure used by the frontend.
    """
    if not rows:
        return {
            "note": NOTE_TEXT,
            "symbols_matched": [],
            "interpretation": {
                "spiritual_meaning": "No specific entry matched. Clarity often grows as you record and review your dreams.",
                "effects_in_the_physical_realm": "Notice patterns and changes around you this week.",
                "what_to_do": "Use simple terms; reflect, repent, and read Psalm 91 before bed."
            },
            "disclaimer": DISCLAIMER
        }

    combined = {
        "spiritual_meaning": "",
        "effects_in_the_physical_realm": "",
        "what_to_do": ""
    }
    matched_symbols: List[str] = []

    for row in rows:
        symbol = (row.get("input") or "").strip()
        if symbol:
            matched_symbols.append(symbol)

        sections = split_output_to_sections(row.get("output", ""))

        if sections["spiritual_meaning"]:
            combined["spiritual_meaning"] += sections["spiritual_meaning"] + " "
        if sections["effects_in_the_physical_realm"]:
            combined["effects_in_the_physical_realm"] += sections["effects_in_the_physical_realm"] + " "
        if sections["what_to_do"]:
            combined["what_to_do"] += sections["what_to_do"] + " "

    # Trim trailing spaces
    for k in combined:
        combined[k] = combined[k].strip()

    return {
        "note": NOTE_TEXT,
        "symbols_matched": matched_symbols,
        "interpretation": combined,
        "disclaimer": DISCLAIMER
    }


# ── Flask app ───────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

# Load the Google Sheet into memory when the app starts
try:
    refresh_cache()
except Exception as e:
    logging.error("Error loading sheet at startup: %s", e)


@app.get("/health")
def health():
    return {"ok": True, "loaded_rows": len(INPUTS), "worksheet": WORKSHEET_NAME}

@app.post("/refresh")
def refresh():
    refresh_cache()
    return {"reloaded": len(INPUTS)}

@app.post("/interpret")
def interpret():
    data = request.get_json(silent=True) or {}
    text  = (data.get("text")  or "").strip()
    email = (data.get("email") or "").strip().lower() or None

    if not text:
        return jsonify({"error": "Please paste your dream text."}), 400

    # PRO users skip free limit
    if email and is_pro(email):
        rows = find_best_rows_for_dream(text)
        result = compose_output(rows)
        result["free_uses_left"] = None
        result["pro"] = True
        return jsonify(result), 200

    ident = get_identifier(email)
    used  = get_count(ident)
    if used >= FREE_QUOTA:
        base = ""  # change to full domain when deployed
        weekly_url  = f"{base}/subscribe/create-checkout?plan=weekly"
        monthly_url = f"{base}/subscribe/create-checkout?plan=monthly"
        if email:
            weekly_url  += f"&email={email}"
            monthly_url += f"&email={email}"

        return jsonify({
            "paywall": True,
            "message": f"Free limit reached ({FREE_QUOTA}). Please subscribe to continue.",
            "free_uses_left": 0,
            "checkout_weekly": weekly_url,
            "checkout_monthly": monthly_url
        }), 402

    rows = find_best_rows_for_dream(text)
    result = compose_output(rows)
    used_now = bump_count(ident)
    result["free_uses_left"] = max(0, FREE_QUOTA - used_now)
    result["pro"] = False
    return jsonify(result), 200


# ── Stripe checkout routes ──────────────────────────────────────────
@app.post("/subscribe/create-checkout")
def create_checkout():
    """Create a Stripe Checkout session for weekly or monthly plan."""
    if not STRIPE_SECRET_KEY or not (PRICE_WEEKLY or PRICE_MONTHLY):
        return jsonify({"error": "Stripe not configured"}), 500

    data = request.get_json(silent=True) or {}
    plan  = (request.args.get("plan") or data.get("plan") or "monthly").lower()
    email = (request.args.get("email") or data.get("email") or "").strip().lower()

    if plan not in ("weekly", "monthly"):
        plan = "monthly"

    price_id = PRICE_WEEKLY if plan == "weekly" else PRICE_MONTHLY

    session = stripe.checkout.Session.create(
        mode="subscription",
        customer_email=email if email else None,
        line_items=[{"price": price_id, "quantity": 1}],
        success_url="http://interpreter.jamaicantruestories.com/stripe/success?session_id={CHECKOUT_SESSION_ID}",
        cancel_url="http://interpreter.jamaicantruestories.com/stripe/cancel",
        client_reference_id=email or None,
        allow_promotion_codes=True
    )

    return jsonify({"url": session.url})


@app.get("/stripe/success")
def stripe_success():
    """After Checkout, verify payment and mark email as PRO."""
    session_id = request.args.get("session_id", "")
    if not session_id:
        return "Missing session_id", 400

    try:
        session = stripe.checkout.Session.retrieve(session_id)
    except Exception as e:
        logging.error("Stripe retrieve error: %s", e)
        return "Error verifying payment.", 400

    email = (
        (session.get("customer_details") or {}).get("email")
        or session.get("customer_email")
        or ""
    )
    status = session.get("status")
    payment_status = session.get("payment_status")

    if email and (status == "complete" or payment_status == "paid"):
        make_pro(email)
        return f"{email} is now PRO. You can return to the Dream Interpreter."
    else:
        return "Payment not completed."


@app.get("/stripe/cancel")
def stripe_cancel():
    return "Payment cancelled. You can return to the Dream Interpreter."


# ── Entrypoint ──────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
