# app.py — Dream Interpreter with paragraph decoding, free tier, Stripe paywall,
# Level 2 AI combiner using Google Sheets as the knowledge base,
# dream text normalizer, and top-N symbol selection.

import os, re, json, logging
from typing import Dict, List
from flask import Flask, request, jsonify
from flask_cors import CORS
import gspread
from google.oauth2.service_account import Credentials
import stripe  # Stripe library

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# TRY to import OpenAI client safely
try:
    from openai import OpenAI  # requires openai >= 1.x
except ImportError:
    OpenAI = None
    logging.warning("Could not import OpenAI client. Will use fallback interpreter.")

# ── Environment ─────────────────────────────────────────────────────
SHEET_ID       = os.environ.get("SHEET_ID", "").strip()              # must be set
WORKSHEET_NAME = os.environ.get("WORKSHEET_NAME", "Sheet1").strip()
CREDS_PATH     = os.environ.get("CREDS_PATH", "credentials.json").strip()

# DEFAULT: 1 free interpretation. Set FREE_QUOTA=0 later to remove free tier entirely.
FREE_QUOTA     = int(os.environ.get("FREE_QUOTA", "1"))
COUNTS_FILE    = os.environ.get("COUNTS_FILE", "usage_counts.json")

STRIPE_SECRET_KEY = os.environ.get("STRIPE_SECRET_KEY", "").strip()
PRICE_WEEKLY      = os.environ.get("PRICE_WEEKLY", "").strip()
PRICE_MONTHLY     = os.environ.get("PRICE_MONTHLY", "").strip()

OPENAI_API_KEY    = os.environ.get("OPENAI_API_KEY", "").strip()

if STRIPE_SECRET_KEY:
    stripe.api_key = STRIPE_SECRET_KEY

# Initialize OpenAI client only if import and key both exist
if OPENAI_API_KEY and OpenAI is not None:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    logging.info("OpenAI client initialized.")
else:
    openai_client = None
    logging.warning("OpenAI client not available (missing key or package). Using fallback interpreter.")

SUBSCRIBERS_FILE = os.environ.get("SUBSCRIBERS_FILE", "subscribers.json")

DISCLAIMER = (
    "Reflective and educational guidance based on cultural symbolism—"
    "no divination, no predictions, and no medical or legal advice."
)
NOTE_TEXT = "Based on traditional symbolism, here is a reflective interpretation:"

# ── Globals (cache) ─────────────────────────────────────────────────
LUT: Dict[str, str] = {}        # 'input' -> 'output'
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


# ── Dream text normalizer ───────────────────────────────────────────

def normalize_dream_text(text: str) -> str:
    """
    Remove common starters like 'I was', 'he was', 'she was' etc.
    to better match sheet symbols (e.g. 'standing on a high place naked').
    """
    if not text:
        return ""
    text = text.strip()
    lower = text.lower()

    starters = [
        "i was ", "i am ", "i'm ",
        "he was ", "he is ", "he's ",
        "she was ", "she is ", "she's ",
        "we were ", "we are ", "we're ",
        "they were ", "they are ", "they're ",
        "in my dream ", "dreamt that i ", "dreamed that i "
    ]

    for s in starters:
        if lower.startswith(s):
            text = text[len(s):].lstrip()
            lower = text.lower()
            break

    return text


# ── Matching (using input + keywords) ───────────────────────────────

def find_best_rows_for_dream(dream_text: str, max_hits: int = 8) -> List[Dict]:
    """
    Use 'input' and 'keywords' columns to find the best matching row(s).
    """
    dream_norm = normalize_dream_text(dream_text)
    dream = dream_norm.lower().strip()
    if not dream:
        return []

    exact_matches: List[Dict] = []
    keyword_matches: List[tuple[int, str, Dict]] = []

    for row in SHEET_ROWS:
        symbol = str(row.get("input", "")).strip().lower()
        keywords_str = str(row.get("keywords", "")).lower()
        keywords = [k.strip() for k in keywords_str.split(",") if k.strip()]

        score = 0

        if dream == symbol and symbol:
            exact_matches.append(row)
            continue

        if symbol and symbol in dream:
            score += 4

        for kw in keywords:
            if not kw:
                continue
            if " " in kw and kw in dream:
                score += 3
            elif kw in dream:
                score += 1

        if score > 0:
            keyword_matches.append((score, symbol, row))

    if exact_matches:
        return exact_matches[:max_hits]

    if keyword_matches:
        keyword_matches.sort(key=lambda x: x[0], reverse=True)
        best_rows = [row for score, sym, row in keyword_matches[:max_hits]]
        return best_rows

    return []


# ── Parse "output" into three sections (fallback) ───────────────────

def split_output_to_sections(output_text: str) -> Dict[str, str]:
    """
    'output' column format:
      "Spiritual Meaning: ... Effects in the Physical Realm: ... What to Do: ..."
    """
    if not output_text:
        return {
            "spiritual_meaning": "",
            "effects_in_the_physical_realm": "",
            "what_to_do": "",
        }

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
    Legacy: combine rows by concatenating sections.
    Used as fallback if OpenAI is unavailable.
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
    seen_symbols = set()

    for row in rows:
        symbol = (row.get("input") or "").strip()
        if symbol:
            key = symbol.lower()
            if key in seen_symbols:
                continue
            seen_symbols.add(key)
            matched_symbols.append(symbol)

        sections = split_output_to_sections(row.get("output", ""))

        if sections["spiritual_meaning"]:
            combined["spiritual_meaning"] += sections["spiritual_meaning"] + " "
        if sections["effects_in_the_physical_realm"]:
            combined["effects_in_the_physical_realm"] += sections["effects_in_the_physical_realm"] + " "
        if sections["what_to_do"]:
            combined["what_to_do"] += sections["what_to_do"] + " "

    for k in combined:
        combined[k] = combined[k].strip()

    return {
        "note": NOTE_TEXT,
        "symbols_matched": matched_symbols,
        "interpretation": combined,
        "disclaimer": DISCLAIMER
    }


# ── Level 2: AI combiner using OpenAI + Sheet rows ──────────────────

def ai_combine_interpretation(dream_text: str, candidate_rows: List[Dict]) -> Dict[str, str]:
    """
    Try to use OpenAI to combine multiple symbol meanings into one interpretation.
    If OpenAI is unavailable or errors, fall back to compose_output().
    """
    if not candidate_rows:
        return {
            "spiritual_meaning": "No specific entry matched yet. Keep recording your dreams; more symbols will be added over time.",
            "effects_in_the_physical_realm": "",
            "what_to_do": "Pray for clarity and protection as you sleep."
        }

    if not openai_client:
        logging.warning("OpenAI client missing or key not set; using compose_output fallback for all rows.")
        fallback = compose_output(candidate_rows)
        return fallback["interpretation"]

    symbols_block = []
    for row in candidate_rows:
        symbol = (row.get("input") or "").strip()
        output = (row.get("output") or "").strip()
        if symbol and output:
            symbols_block.append({
                "symbol": symbol,
                "meaning": output
            })

    system_message = (
        "You are the official Jamaican True Stories Dream Interpreter. "
        "You MUST follow these rules strictly:\n"
        "\n"
        "1. Use ONLY the meanings provided in the 'symbols' list. "
        "Do NOT invent new symbols, new doctrines, new spiritual meanings, "
        "or extra ideas beyond what is written in those meanings. "
        "Do NOT use psychology, generic spirituality, or Western dream books.\n"
        "\n"
        "2. The final answer MUST always have exactly three fields: "
        "'spiritual_meaning', 'effects_in_the_physical_realm', and 'what_to_do'. "
        "Each field should be one clear, concise paragraph. "
        "Do NOT add extra sections, long sermons, or repeated content.\n"
        "\n"
        "3. You are COMBINING the given meanings, not changing doctrine. "
        "If a meaning already encodes special rules (for example: crying = joy, "
        "smiling/laughing = sadness, marriage = death or spirit spouse, "
        "food = witchcraft feeding, clear water = clarity, dirty water = contamination, etc.), "
        "you must preserve that logic and NEVER contradict it.\n"
        "\n"
        "4. 'spiritual_meaning' should summarize the spiritual picture using ONLY the symbol meanings. "
        "'effects_in_the_physical_realm' should describe what might show up in life based on those same meanings. "
        "'what_to_do' should give short, practical spiritual steps (pray for clarity, break covenants, "
        "reject witchcraft feeding, ask for protection, etc.), again based ONLY on the provided meanings.\n"
        "\n"
        "5. Do NOT randomly add prosperity, pregnancy, promotion, or generic encouragement unless the "
        "provided meanings explicitly say so. Do NOT add Bible verses or long teachings. "
        "Keep it direct, Caribbean, and spiritually serious.\n"
        "\n"
        "6. Respond ONLY with valid JSON in this exact shape:\n"
        "{\n"
        '  \"spiritual_meaning\": \"...\",\n'
        '  \"effects_in_the_physical_realm\": \"...\",\n'
        '  \"what_to_do\": \"...\"\n'
        "}\n"
    )

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": system_message},
                {
                    "role": "user",
                    "content": (
                        "Dream:\n"
                        + dream_text
                        + "\n\n"
                        "Here are the relevant dictionary entries (symbol + meaning):\n"
                        + json.dumps(symbols_block, ensure_ascii=False)
                        + "\n\n"
                        "Using ONLY these meanings, generate a combined interpretation in this JSON format:\n"
                        "{\n"
                        '  \"spiritual_meaning\": \"...\",\n'
                        '  \"effects_in_the_physical_realm\": \"...\",\n'
                        '  \"what_to_do\": \"...\"\n'
                        "}"
                    )
                }
            ],
            temperature=0.3,
        )

        raw = response.choices[0].message.content.strip()
        data = json.loads(raw)
        return {
            "spiritual_meaning": data.get("spiritual_meaning", "").strip(),
            "effects_in_the_physical_realm": data.get("effects_in_the_physical_realm", "").strip(),
            "what_to_do": data.get("what_to_do", "").strip(),
        }
    except Exception as e:
        logging.error("Error in ai_combine_interpretation: %s", e)
        fallback = compose_output(candidate_rows)
        return fallback["interpretation"]


def build_level2_result(dream_text: str, rows: List[Dict]) -> Dict[str, object]:
    """
    Build the final result object using Level 2 AI combiner.
    Falls back to compose_output if something goes wrong.
    """
    try:
        if not rows:
            return compose_output(rows)
        interpretation = ai_combine_interpretation(dream_text, rows)
        matched_symbols = [(r.get("input") or "").strip() for r in rows if (r.get("input") or "").strip()]
        return {
            "note": NOTE_TEXT,
            "symbols_matched": matched_symbols,
            "interpretation": interpretation,
            "disclaimer": DISCLAIMER
        }
    except Exception as e:
        logging.error("build_level2_result error: %s", e)
        return compose_output(rows)


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


@app.get("/refresh")
def refresh():
    refresh_cache()
    return {"reloaded": len(INPUTS)}


@app.post("/interpret")
def interpret():
    data = request.get_json(silent=True) or {}
    raw_text = (data.get("text")  or "").strip()
    email = (data.get("email") or "").strip().lower() or None

    if not raw_text:
        return jsonify({"error": "Please paste your dream text."}), 400

    text = normalize_dream_text(raw_text)

    # PRO users skip limits
    if email and is_pro(email):
        rows = find_best_rows_for_dream(text, max_hits=10)
        result = build_level2_result(text, rows)
        result["free_uses_left"] = None
        result["pro"] = True
        return jsonify(result), 200

    # Non-PRO users: 1 free interpretation (or FREE_QUOTA uses), then paywall.
    ident = get_identifier(email)
    used  = get_count(ident)

    # If FREE_QUOTA <= 0 → no free tier at all
    if FREE_QUOTA <= 0:
        base = ""
        weekly_url  = f"{base}/subscribe/create-checkout?plan=weekly"
        monthly_url = f"{base}/subscribe/create-checkout?plan=monthly"
        if email:
            weekly_url  += f"&email={email}"
            monthly_url += f"&email={email}"

        return jsonify({
            "paywall": True,
            "message": "Please subscribe to access the Dream Interpreter.",
            "free_uses_left": 0,
            "checkout_weekly": weekly_url,
            "checkout_monthly": monthly_url
        }), 200

    # If used up free quota → paywall
    if used >= FREE_QUOTA:
        base = ""
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
        }), 200

    # Still has free uses
    rows = find_best_rows_for_dream(text, max_hits=10)
    result = build_level2_result(text, rows)
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
        success_url="https://interpreter.jamaicantruestories.com/stripe/success?session_id={CHECKOUT_SESSION_ID}",
        cancel_url="https://interpreter.jamaicantruestories.com/stripe/cancel",
        client_reference_id=email or None,
        allow_promotion_codes=True
    )

    return jsonify({"url": session.url})


@app.get("/stripe/success")
def stripe_success():
    """After Checkout, verify payment and mark email as PRO, then show a JTS-themed success page."""
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

    interpreter_url = "https://jamaicantruestories.com/pages/dream-interpreter"

    if email and (status == "complete" or payment_status == "paid"):
        make_pro(email)

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Jamaican True Stories • Dream Interpreter PRO</title>
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <style>
    body {{
      margin: 0;
      padding: 0;
      font-family: system-ui,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
      background: radial-gradient(circle at top, #1c1510 0, #050505 55%);
      color: #f5f5f5;
      display: flex;
      align-items: center;
      justify-content: center;
      min-height: 100vh;
    }}
    .jts-card {{
      max-width: 480px;
      width: 90%;
      padding: 24px 22px 26px;
      border-radius: 18px;
      background: rgba(0,0,0,0.86);
      border: 1px solid rgba(212,175,55,0.45);
      box-shadow: 0 24px 60px rgba(0,0,0,0.9);
      text-align: center;
    }}
    .jts-tag {{
      display:inline-block;
      padding:4px 10px;
      border-radius:999px;
      border:1px solid rgba(212,175,55,0.6);
      font-size:0.7rem;
      letter-spacing:0.14em;
      text-transform:uppercase;
      opacity:0.85;
      margin-bottom:8px;
    }}
    h1 {{
      font-size:1.5rem;
      letter-spacing:0.08em;
      text-transform:uppercase;
      margin:4px 0 8px;
    }}
    p {{
      font-size:0.95rem;
      line-height:1.6;
      margin:0 0 10px;
      opacity:0.9;
    }}
    .email {{
      font-size:0.85rem;
      opacity:0.8;
      margin-bottom:14px;
    }}
    .btn-main {{
      display:inline-flex;
      align-items:center;
      justify-content:center;
      padding:12px 22px;
      border-radius:999px;
      background:linear-gradient(90deg,#d4af37,#f3e28a);
      color:#111;
      border:0;
      font-weight:700;
      font-size:0.95rem;
      letter-spacing:0.08em;
      text-transform:uppercase;
      text-decoration:none;
      box-shadow:0 12px 30px rgba(0,0,0,0.85);
      margin-top:14px;
    }}
    .btn-main:hover {{
      filter:brightness(1.05);
    }}
    .note {{
      margin-top:16px;
      font-size:0.78rem;
      opacity:0.65;
      letter-spacing:0.08em;
      text-transform:uppercase;
    }}
  </style>
</head>
<body>
  <div class="jts-card">
    <div class="jts-tag">Dream Interpreter • PRO Activated</div>
    <h1>Thank You for Supporting<br>Jamaican True Stories</h1>
    <p class="email">{email} is now <strong>PRO</strong>.</p>
    <p>You now have full access to deeper dream interpretations without hitting the free limit.</p>
    <p>Return to the Dream Interpreter to continue decoding your dreams with unlimited access.</p>
    <a class="btn-main" href="{interpreter_url}">Return to Dream Interpreter</a>
    <p class="note">Jamaican True Stories • Spiritual Dream Decoder</p>
  </div>
</body>
</html>"""
        return html

    html_fail = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Payment Not Completed • Jamaican True Stories</title>
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <style>
    body {{
      margin: 0;
      padding: 0;
      font-family: system-ui,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
      background: radial-gradient(circle at top, #2a1515 0, #050505 55%);
      color: #f5f5f5;
      display: flex;
      align-items: center;
      justify-content: center;
      min-height: 100vh;
    }}
    .jts-card {{
      max-width: 440px;
      width: 90%;
      padding: 22px 20px 24px;
      border-radius: 18px;
      background: rgba(0,0,0,0.86);
      border: 1px solid rgba(212,175,55,0.35);
      box-shadow: 0 24px 60px rgba(0,0,0,0.9);
      text-align: center;
    }}
    h1 {{
      font-size:1.4rem;
      letter-spacing:0.08em;
      text-transform:uppercase;
      margin:4px 0 8px;
    }}
    p {{
      font-size:0.95rem;
      line-height:1.6;
      margin:0 0 10px;
      opacity:0.9;
    }}
    .btn-main {{
      display:inline-flex;
      align-items:center;
      justify-content:center;
      padding:11px 20px;
      border-radius:999px;
      background:transparent;
      color:#f5f5f5;
      border:1px solid rgba(212,175,55,0.6);
      font-weight:600;
      font-size:0.9rem;
      letter-spacing:0.08em;
      text-transform:uppercase;
      text-decoration:none;
      margin-top:14px;
    }}
    .btn-main:hover {{
      filter:brightness(1.05);
    }}
  </style>
</head>
<body>
  <div class="jts-card">
    <h1>Payment Not Completed</h1>
    <p>Your subscription payment was not confirmed.</p>
    <p>If this was a mistake, you can return to the Dream Interpreter and try again.</p>
    <a class="btn-main" href="{interpreter_url}">Back to Dream Interpreter</a>
  </div>
</body>
</html>"""
    return html_fail


@app.get("/stripe/cancel")
def stripe_cancel():
    return "Payment cancelled. You can return to the Dream Interpreter."


# ── Entrypoint ──────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
