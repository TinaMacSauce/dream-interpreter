# app.py — Dream Interpreter with paragraph decoding, free tier, and Stripe paywall

import os, re, json, logging
from typing import Dict, List, Tuple
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
LUT: Dict[str, str] = {}   # 'input' -> 'output'
INPUTS: List[str] = []     # list of input phrases (lowercased)


# ── Google Sheets helpers ───────────────────────────────────────────
def _gs_client():
    scopes = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
    creds  = Credentials.from_service_account_file(CREDS_PATH, scopes=scopes)
    return gspread.authorize(creds)

def load_sheet() -> Tuple[List[str], Dict[str, str]]:
    """Read columns A: input, B: output from WORKSHEET_NAME."""
    if not SHEET_ID:
        raise RuntimeError("Missing SHEET_ID environment variable.")
    gc = _gs_client()
    ws = gc.open_by_key(SHEET_ID).worksheet(WORKSHEET_NAME)
    rows = ws.get_all_records()

    lut: Dict[str, str] = {}
    for r in rows:
        k = (r.get("input") or "").strip().lower()
        v = (r.get("output") or "").strip()
        if k:
            lut[k] = v
    inputs = list(lut.keys())
    logging.info("Loaded %d rows from %s", len(inputs), WORKSHEET_NAME)
    return inputs, lut

def refresh_cache():
    global INPUTS, LUT
    INPUTS, LUT = load_sheet()


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


# ── Matching (paragraph → symbols) ──────────────────────────────────
WORD = re.compile(r"[a-z']+")

def best_matches(paragraph: str, max_hits: int = 8) -> List[str]:
    text = paragraph.lower().strip()
    if not text:
        return []

    # direct phrase hits
    found = [k for k in INPUTS if k and k in text]

    seen, ordered = set(), []
    for k in found:
        if k not in seen:
            ordered.append(k)
            seen.add(k)

    if ordered:
        return ordered[:max_hits]

    # token overlap fallback
    toks = set(WORD.findall(text))
    scored = []
    for k in INPUTS:
        ktoks = set(WORD.findall(k))
        score = len(toks & ktoks)
        if score:
            scored.append((score, k))
    scored.sort(reverse=True)
    return [k for _, k in scored[:max_hits]]

def compose_output(keys: List[str]) -> Dict[str, object]:
    if not keys:
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

    merged = "\n\n".join([LUT[k] for k in keys if k in LUT])
    return {
        "note": NOTE_TEXT,
        "symbols_matched": keys,
        "interpretation": {
            "spiritual_meaning": merged,
            "effects_in_the_physical_realm": "",
            "what_to_do": ""
        },
        "disclaimer": DISCLAIMER
    }


# ── Flask app ───────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

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
        keys = best_matches(text)
        result = compose_output(keys)
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

    keys = best_matches(text)
    result = compose_output(keys)
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
        success_url="http://127.0.0.1:5000/stripe/success?session_id={CHECKOUT_SESSION_ID}",
        cancel_url="http://127.0.0.1:5000/stripe/cancel",
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
    refresh_cache()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
 