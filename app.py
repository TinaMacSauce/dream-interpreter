# app.py — Jamaican True Stories Dream Interpreter
# (SSE + Level 4 Seal Engine + Level 5 Formatter + Airtable Storage)
#
# Features:
# - Paragraph decoding (Top-5 symbols, no duplicates)
# - Google Sheets driven meanings (Symbols tab + SEALS tab)
# - Output format enforced: Spiritual Meaning / Effects in the Physical Realm / What To Do
# - Optional OpenAI combiner (ONLY uses sheet meanings; no invention)
# - AI Voice Layer (polishes your final result; no new meanings)
# - Seal-specific tone (warning / instruction / confirmation)
# - Dream Receipt share_phrase optimized for captions
# - Level 5 Formatter (cleans output + removes duplicates + punctuation guard)
# - Free tier (N free) -> Email gate -> Paywall
# - Stripe Checkout subscription flow + Webhook confirmation
# - Paid users stored in Airtable (Subscribers table)
# - Usage counts stored in Airtable (Usage table)
# - Dream Receipts saved to Airtable (Receipts table)
# - BASIC USER TRACKING to Google Sheets (Events sheet): /track endpoint
# - TEMP DEBUG: /debug/creds shows service account + scopes + creds path

import os
import re
import json
import logging
import secrets
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from flask import Flask, request, jsonify, session
from flask_cors import CORS

import gspread
from google.oauth2.service_account import Credentials
import stripe

# AI Voice layer (writing only)
from ai_voice import write_interpretation

# Airtable storage (Level 5)
from airtable_store import (
    is_paid as airtable_is_paid,
    mark_paid as airtable_mark_paid,
    get_usage,
    set_email_captured,
    increment_free_used,
    save_receipt,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# Optional OpenAI client (used only for combiner)
try:
    from openai import OpenAI  # requires openai >= 1.x
except ImportError:
    OpenAI = None
    logging.warning("Could not import OpenAI client. Will use fallback interpreter.")


# =========================
# ENV / CONFIG
# =========================
SHEET_ID = os.environ.get("SHEET_ID", "").strip()
CREDS_PATH = os.environ.get("CREDS_PATH", "credentials.json").strip()

SYMBOLS_WORKSHEET_NAME = os.environ.get(
    "SYMBOLS_WORKSHEET_NAME",
    os.environ.get("WORKSHEET_NAME", "Sheet1")
).strip()
SEALS_WORKSHEET_NAME = os.environ.get("SEALS_WORKSHEET_NAME", "SEALS").strip()

# BASIC TRACKING SHEET (by NAME, separate from SHEET_ID sheet)
EVENTS_SHEET_NAME = os.environ.get("EVENTS_SHEET_NAME", "Dream Interpreter Events").strip()

FREE_QUOTA = int(os.environ.get("FREE_QUOTA", "3"))
EMAIL_GATE_AT = int(os.environ.get("EMAIL_GATE_AT", "4"))  # UI helper only

STRIPE_SECRET_KEY = os.environ.get("STRIPE_SECRET_KEY", "").strip()
STRIPE_WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET", "").strip()
PRICE_WEEKLY = os.environ.get("PRICE_WEEKLY", "").strip()
PRICE_MONTHLY = os.environ.get("PRICE_MONTHLY", "").strip()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4.1-mini").strip()
USE_AI_VOICE = os.environ.get("USE_AI_VOICE", "1").strip() == "1"

SHOPIFY_INTERPRETER_URL = "https://jamaicantruestories.com/pages/dream-interpreter"
BACKEND_BASE_URL = os.environ.get("BACKEND_BASE_URL", "https://interpreter.jamaicantruestories.com").strip()

DISCLAIMER = (
    "Reflective and educational guidance based on cultural symbolism - "
    "no divination, no predictions, and no medical or legal advice."
)
NOTE_TEXT = "Based on traditional symbolism, here is a reflective interpretation:"
DOCTRINE_LINE = "The ending of your dream is the receipt. It seals the message."

if STRIPE_SECRET_KEY:
    stripe.api_key = STRIPE_SECRET_KEY
else:
    logging.warning("STRIPE_SECRET_KEY is missing. Stripe routes will fail.")

if OPENAI_API_KEY and OpenAI is not None:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    logging.info("OpenAI client initialized (combiner).")
else:
    openai_client = None
    logging.warning("OpenAI client not available (missing key or package). Using fallback interpreter.")


# =========================
# GLOBAL CACHES
# =========================
SYMBOL_ROWS: List[Dict] = []
SEAL_ROWS: List[Dict] = []


# =========================
# LEVEL 5 — OUTPUT FORMATTER
# =========================
def format_text(s: Optional[str], max_len: int = 1200) -> str:
    if s is None:
        return ""

    s = str(s).strip()
    s = s.replace("None", "").replace("null", "").strip()
    s = re.sub(r"\s+", " ", s).strip()

    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", s) if p.strip()]
    deduped: List[str] = []
    seen = set()
    for p in parts:
        key = p.lower()
        if key not in seen:
            deduped.append(p)
            seen.add(key)
    s = " ".join(deduped).strip()

    if max_len and len(s) > max_len:
        s = s[:max_len].rstrip()

    if s and s[-1] not in ".!?":
        s += "."

    return s


def format_interpretation_payload(result: Dict[str, object]) -> Dict[str, object]:
    interp = result.get("interpretation") or {}
    if isinstance(interp, dict):
        interp["spiritual_meaning"] = format_text(interp.get("spiritual_meaning", ""), 1200)
        interp["effects_in_the_physical_realm"] = format_text(interp.get("effects_in_the_physical_realm", ""), 1200)
        interp["what_to_do"] = format_text(interp.get("what_to_do", ""), 1200)
        result["interpretation"] = interp

    seal = result.get("seal")
    if isinstance(seal, dict):
        seal["seal_name"] = format_text(seal.get("seal_name", ""), 160)
        seal["seal_summary"] = format_text(seal.get("seal_summary", ""), 300)
        if "spiritual_meaning" in seal:
            seal["spiritual_meaning"] = format_text(seal.get("spiritual_meaning", ""), 700)
        if "physical_effects" in seal:
            seal["physical_effects"] = format_text(seal.get("physical_effects", ""), 700)
        if "what_to_do" in seal:
            seal["what_to_do"] = format_text(seal.get("what_to_do", ""), 700)
        result["seal"] = seal

    receipt = result.get("receipt")
    if isinstance(receipt, dict):
        receipt["seal_summary"] = format_text(receipt.get("seal_summary", ""), 260)
        receipt["doctrine_line"] = format_text(receipt.get("doctrine_line", ""), 180)
        if "share_phrase" in receipt:
            receipt["share_phrase"] = format_text(receipt.get("share_phrase", ""), 220)
        if "receipt_id" in receipt and receipt["receipt_id"]:
            receipt["receipt_id"] = str(receipt["receipt_id"]).replace(" ", "")
        result["receipt"] = receipt

    return result


# =========================
# GOOGLE SHEETS HELPERS
# =========================
def _gs_client():
    """
    IMPORTANT: must be read/write so we can append tracking events.
    """
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = Credentials.from_service_account_file(CREDS_PATH, scopes=scopes)
    return gspread.authorize(creds)


def load_sheet_rows(worksheet_name: str) -> List[Dict]:
    if not SHEET_ID:
        raise RuntimeError("Missing SHEET_ID environment variable.")
    gc = _gs_client()
    ws = gc.open_by_key(SHEET_ID).worksheet(worksheet_name)
    return ws.get_all_records()


def refresh_cache() -> None:
    global SYMBOL_ROWS, SEAL_ROWS
    SYMBOL_ROWS = load_sheet_rows(SYMBOLS_WORKSHEET_NAME)
    SEAL_ROWS = load_sheet_rows(SEALS_WORKSHEET_NAME)
    logging.info(
        "Loaded %d symbol rows (%s) and %d seal rows (%s)",
        len(SYMBOL_ROWS), SYMBOLS_WORKSHEET_NAME,
        len(SEAL_ROWS), SEALS_WORKSHEET_NAME
    )


# =========================
# BASIC USER TRACKING (EVENTS SHEET)
# =========================
def _events_ws():
    """
    Opens the tracking sheet by NAME (EVENTS_SHEET_NAME).
    Your service account MUST have Editor access to this sheet.
    """
    gc = _gs_client()
    sh = gc.open(EVENTS_SHEET_NAME)
    return sh.sheet1


def log_event(user_id: str, event_type: str, metadata: Optional[dict] = None) -> None:
    """
    Appends one event row into the Dream Interpreter Events sheet.
    Columns: timestamp | user_id | event_type | metadata
    """
    user_id = (user_id or "").strip()
    event_type = (event_type or "").strip()
    if not user_id or not event_type:
        return

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    meta_str = json.dumps(metadata or {}, ensure_ascii=False)

    ws = _events_ws()
    ws.append_row([ts, user_id, event_type, meta_str], value_input_option="RAW")


# =========================
# SESSION BOOTSTRAP
# =========================
def session_bootstrap():
    session.setdefault("free_used", 0)
    session.setdefault("user_email", "")
    session.setdefault("is_paid", False)


def set_email_in_session(email: str):
    email = (email or "").strip().lower()
    if not email:
        return
    session["user_email"] = email
    try:
        set_email_captured(email)
    except Exception:
        pass


def mark_paid_in_session(email: Optional[str] = None):
    session["is_paid"] = True
    if email:
        set_email_in_session(email)


def effective_email(user_email_from_body: Optional[str]) -> str:
    se = (session.get("user_email") or "").strip().lower()
    if se:
        return se
    return (user_email_from_body or "").strip().lower()


def user_is_paid(email: Optional[str]) -> bool:
    if session.get("is_paid"):
        return True
    if email:
        try:
            return airtable_is_paid(email)
        except Exception:
            return False
    return False


# =========================
# DREAM NORMALIZER
# =========================
def normalize_dream_text(text: str) -> str:
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
            break

    return text


# =========================
# SMART SYMBOL ENGINE (SSE)
# =========================
def _csv_keywords(cell: str) -> List[str]:
    if not cell:
        return []
    return [k.strip().lower() for k in str(cell).split(",") if k.strip()]


def find_best_rows_for_dream(dream_text: str, max_hits: int = 5) -> List[Dict]:
    dream = normalize_dream_text(dream_text).lower().strip()
    if not dream:
        return []

    exact_matches: List[Dict] = []
    scored_rows: List[Tuple[int, str, Dict]] = []

    has_monitoring = ("stranger" in dream or "strange woman" in dream) and ("ask" in dream or "question" in dream)
    has_wedding = ("wedding" in dream or "getting married" in dream or "marriage" in dream)
    has_sex = ("sex" in dream or "intercourse" in dream)

    for row in SYMBOL_ROWS:
        symbol_raw = (row.get("input") or "").strip()
        symbol = symbol_raw.lower()

        keywords = _csv_keywords(str(row.get("keywords", "")))
        try:
            weight = float(row.get("weight") or 1.0)
        except Exception:
            weight = 1.0

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

        if has_monitoring and ("monitoring spirit" in symbol or "stranger asking questions" in symbol):
            score += 5
        if has_wedding and ("wedding" in symbol or "getting married" in symbol):
            score += 3
        if has_sex and ("sexual" in symbol or "spiritual spouse" in symbol or "sexual intercourse" in symbol):
            score += 3

        if score > 0:
            score = int(score * weight)
            scored_rows.append((score, symbol_raw, row))

    if exact_matches:
        seen = set()
        unique_rows: List[Dict] = []
        for r in exact_matches:
            key = (r.get("input") or "").strip().lower()
            if key and key not in seen:
                seen.add(key)
                unique_rows.append(r)
                if len(unique_rows) >= max_hits:
                    break
        return unique_rows

    if scored_rows:
        scored_rows.sort(key=lambda x: x[0], reverse=True)
        seen_symbols = set()
        chosen: List[Dict] = []
        for score, symbol_raw, row in scored_rows:
            key = symbol_raw.strip().lower()
            if not key or key in seen_symbols:
                continue
            seen_symbols.add(key)
            chosen.append(row)
            if len(chosen) >= max_hits:
                break
        return chosen

    return []


# =========================
# OUTPUT SECTION SPLITTER
# =========================
def split_output_to_sections(output_text: str) -> Dict[str, str]:
    if not output_text:
        return {"spiritual_meaning": "", "effects_in_the_physical_realm": "", "what_to_do": ""}

    text = " ".join(str(output_text).split())

    sm_label = "spiritual meaning:"
    ef_label = "effects in the physical realm:"
    wtd_label = "what to do:"

    lower = text.lower()
    sm_idx = lower.find(sm_label)
    ef_idx = lower.find(ef_label)
    wtd_idx = lower.find(wtd_label)

    def slice_section(label: str, start: int, end: int) -> str:
        if start == -1:
            return ""
        chunk = text[start + len(label):] if end == -1 else text[start + len(label):end]
        return chunk.strip(" .")

    sm = slice_section(sm_label, sm_idx, ef_idx if ef_idx != -1 else wtd_idx)
    ef = slice_section(ef_label, ef_idx, wtd_idx)
    wtd = slice_section(wtd_label, wtd_idx, -1)

    return {
        "spiritual_meaning": sm,
        "effects_in_the_physical_realm": ef,
        "what_to_do": wtd,
    }


def compose_output(rows: List[Dict]) -> Dict[str, object]:
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

    combined = {"spiritual_meaning": "", "effects_in_the_physical_realm": "", "what_to_do": ""}
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


def ai_combine_interpretation(dream_text: str, candidate_rows: List[Dict]) -> Dict[str, str]:
    if not candidate_rows:
        return {
            "spiritual_meaning": "No specific entry matched yet. Keep recording your dreams; more symbols will be added over time.",
            "effects_in_the_physical_realm": "",
            "what_to_do": "Pray for clarity, protection, and wisdom."
        }

    if not openai_client:
        fallback = compose_output(candidate_rows)
        return fallback["interpretation"]

    symbols_block = []
    for row in candidate_rows:
        symbol = (row.get("input") or "").strip()
        output = (row.get("output") or "").strip()
        if symbol and output:
            symbols_block.append({"symbol": symbol, "meaning": output})

    system_message = (
        "You are the dream interpreter for Jamaican True Stories. "
        "You ONLY use the meanings provided in the 'symbols' list. "
        "Do not invent new spiritual meanings. "
        "Your output MUST be valid JSON with exactly three keys: "
        "'spiritual_meaning', 'effects_in_the_physical_realm', 'what_to_do'. "
        "Keep the tone gentle, spiritual, and clear."
    )

    try:
        response = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_message},
                {
                    "role": "user",
                    "content": (
                        "Dream:\n"
                        + dream_text
                        + "\n\nHere are the relevant dictionary entries (symbol + meaning):\n"
                        + json.dumps(symbols_block, ensure_ascii=False)
                        + "\n\nReturn ONLY JSON:\n"
                        "{\n"
                        '  "spiritual_meaning": "...",\n'
                        '  "effects_in_the_physical_realm": "...",\n'
                        '  "what_to_do": "..."\n'
                        "}"
                    )
                }
            ],
            temperature=0.35,
        )
        raw = response.choices[0].message.content.strip()
        data = json.loads(raw)
        return {
            "spiritual_meaning": str(data.get("spiritual_meaning", "")).strip(),
            "effects_in_the_physical_realm": str(data.get("effects_in_the_physical_realm", "")).strip(),
            "what_to_do": str(data.get("what_to_do", "")).strip(),
        }
    except Exception as e:
        logging.error("ai_combine_interpretation error: %s", e)
        fallback = compose_output(candidate_rows)
        return fallback["interpretation"]


def build_level2_result(dream_text: str, rows: List[Dict]) -> Dict[str, object]:
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


# =========================
# SEAL TONE
# =========================
def tone_from_seal(seal: Optional[Dict[str, object]]) -> str:
    if not seal:
        return "instruction"

    risk = str(seal.get("risk_level") or "").strip().lower()
    category = str(seal.get("seal_category") or "").strip().lower()
    dream_type = str(seal.get("dream_type") or "").strip().lower()

    if risk in ("high", "high risk", "severe", "critical", "danger", "red"):
        return "warning"
    if risk in ("low", "safe", "protected", "confirmation", "green"):
        return "confirmation"

    if any(k in category for k in ("warning", "attack", "threat", "demonic", "trap")):
        return "warning"
    if any(k in category for k in ("confirmation", "protection", "covered", "victory")):
        return "confirmation"
    if any(k in dream_type for k in ("warning", "attack")):
        return "warning"
    if any(k in dream_type for k in ("confirmation", "protection")):
        return "confirmation"

    return "instruction"


# =========================
# SHARE PHRASE
# =========================
def share_phrase_from_receipt(receipt: Dict[str, object], tone: str) -> str:
    dream_type = str(receipt.get("dream_type") or "Unclassified").strip()
    risk = str(receipt.get("risk_level") or "Unknown").strip()
    code = str(receipt.get("code_name") or "SEAL-NONE").strip()
    summary = str(receipt.get("seal_summary") or "").strip()

    if tone == "warning":
        base = f"{code} • {dream_type} • {risk}. The ending is a warning."
    elif tone == "confirmation":
        base = f"{code} • {dream_type} • {risk}. The ending confirms stability."
    else:
        base = f"{code} • {dream_type} • {risk}. The ending gives instruction."

    return (f"{base} {summary}".strip() if summary else base)


# =========================
# AI VOICE (writing only)
# =========================
def polish_with_ai_voice(dream_text: str, interpretation: Dict[str, str], tone: str) -> Dict[str, str]:
    if not USE_AI_VOICE:
        return interpretation
    if not OPENAI_API_KEY:
        return interpretation

    sm = (interpretation.get("spiritual_meaning") or "").strip()
    ef = (interpretation.get("effects_in_the_physical_realm") or "").strip()
    wtd = (interpretation.get("what_to_do") or "").strip()
    if not (sm or ef or wtd):
        return interpretation

    doctrine_summary = (
        "IMPORTANT: Return your final answer in EXACTLY this labeled format:\n"
        "Spiritual Meaning: ...\n"
        "Effects in the Physical Realm: ...\n"
        "What to Do: ...\n\n"
        "Use ONLY the content below. Do NOT add new symbols or meanings.\n\n"
        f"Spiritual Meaning: {sm}\n"
        f"Effects in the Physical Realm: {ef}\n"
        f"What to Do: {wtd}\n"
    )

    action_guidance = (
        "Rewrite clearly, smoothly, and engagingly, but KEEP the same meaning. "
        "No new claims. No new symbols. Keep it beginner-friendly and spiritually grounded. "
        f"Tone mode: {tone} (warning=direct/protective, instruction=grounded/practical, confirmation=calm/reassuring)."
    )

    try:
        try:
            polished_text = write_interpretation(
                dream_text=dream_text,
                doctrine_summary=doctrine_summary,
                action_guidance=action_guidance,
                tone=tone
            )
        except TypeError:
            polished_text = write_interpretation(
                dream_text=dream_text,
                doctrine_summary=doctrine_summary,
                action_guidance=action_guidance
            )

        sections = split_output_to_sections(polished_text)
        if not (sections["spiritual_meaning"] or sections["effects_in_the_physical_realm"] or sections["what_to_do"]):
            return interpretation

        return {
            "spiritual_meaning": sections["spiritual_meaning"] or sm,
            "effects_in_the_physical_realm": sections["effects_in_the_physical_realm"] or ef,
            "what_to_do": sections["what_to_do"] or wtd,
        }
    except Exception as e:
        logging.error("AI Voice polish failed; using original interpretation. Error: %s", e)
        return interpretation


# =========================
# SEAL ENGINE + RECEIPT
# =========================
def extract_ending_window(raw_text: str, sentences: int = 3) -> str:
    if not raw_text:
        return ""
    parts = re.split(r"(?<=[.!?])\s+", raw_text.strip())
    parts = [p.strip() for p in parts if p.strip()]
    if not parts:
        return raw_text.strip()
    return " ".join(parts[-sentences:]).strip()


def first_sentence(text: str) -> str:
    if not text:
        return ""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return (parts[0] if parts else text).strip()


def _seal_keywords(row: Dict) -> List[str]:
    return _csv_keywords(str(row.get("keywords", "")))


def match_seal(ending_window: str) -> Optional[Dict[str, object]]:
    if not ending_window:
        return None
    hay = ending_window.lower()

    best_row = None
    best_priority = -1

    for row in SEAL_ROWS:
        kws = _seal_keywords(row)
        if not kws:
            continue
        try:
            priority = int(row.get("priority") or 0)
        except Exception:
            priority = 0

        hit = any((kw in hay) for kw in kws if kw)
        if hit and priority > best_priority:
            best_row = row
            best_priority = priority

    if not best_row:
        return None

    return {
        "seal_id": str(best_row.get("seal_id") or "").strip(),
        "seal_name": str(best_row.get("seal_name") or "").strip(),
        "seal_category": str(best_row.get("seal_category") or "").strip(),
        "dream_type": str(best_row.get("dream_type") or "").strip(),
        "risk_level": str(best_row.get("risk_level") or "").strip(),
        "spiritual_meaning": str(best_row.get("spiritual_meaning") or "").strip(),
        "physical_effects": str(best_row.get("physical_effects") or "").strip(),
        "what_to_do": str(best_row.get("what_to_do") or "").strip(),
        "priority": best_priority,
        "ending_window": ending_window,
    }


def build_receipt(seal: Optional[Dict[str, object]], symbols: List[str]) -> Dict[str, object]:
    seal_summary = ""
    dream_type = "Unclassified"
    risk_level = "Unknown"

    if seal:
        seal_summary = first_sentence(str(seal.get("spiritual_meaning") or ""))
        dream_type = str(seal.get("dream_type") or dream_type) or dream_type
        risk_level = str(seal.get("risk_level") or risk_level) or risk_level

    top_symbols = symbols[:3]
    code_name = f"SEAL-{seal.get('seal_id')}" if seal and seal.get("seal_id") else "SEAL-NONE"

    return {
        "receipt_id": "rec_" + secrets.token_urlsafe(10),
        "dream_type": dream_type,
        "top_symbols": top_symbols,
        "seal_summary": seal_summary,
        "risk_level": risk_level,
        "code_name": code_name,
        "doctrine_line": DOCTRINE_LINE
    }


def seal_preview(seal: Optional[Dict[str, object]]) -> Dict[str, object]:
    if not seal:
        return {"locked": True, "seal_summary": "", "seal_name": "", "risk_level": "", "dream_type": ""}
    return {
        "locked": True,
        "seal_name": str(seal.get("seal_name") or ""),
        "seal_category": str(seal.get("seal_category") or ""),
        "dream_type": str(seal.get("dream_type") or ""),
        "risk_level": str(seal.get("risk_level") or ""),
        "seal_summary": first_sentence(str(seal.get("spiritual_meaning") or "")),
    }


# =========================
# FLASK APP
# =========================
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "change-this-secret")
CORS(app)

try:
    refresh_cache()
except Exception as e:
    logging.error("Error loading sheets at startup: %s", e)


@app.get("/health")
def health():
    return {
        "ok": True,
        "symbols_loaded": len(SYMBOL_ROWS),
        "seals_loaded": len(SEAL_ROWS),
        "symbols_tab": SYMBOLS_WORKSHEET_NAME,
        "seals_tab": SEALS_WORKSHEET_NAME,
        "free_quota": FREE_QUOTA,
        "email_gate_at": EMAIL_GATE_AT,
        "use_ai_voice": USE_AI_VOICE,
        "events_sheet_name": EVENTS_SHEET_NAME,
    }


# =========================
# TEMP DEBUG — CREDS + SCOPES
# =========================
@app.get("/debug/creds")
def debug_creds():
    """
    TEMP: Shows what Render actually loaded (service account + scopes + path).
    SAFE: Does NOT print private key.
    """
    try:
        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive",
        ]
        creds = Credentials.from_service_account_file(CREDS_PATH, scopes=scopes)
        return jsonify({
            "ok": True,
            "client_email": creds.service_account_email,
            "scopes": list(creds.scopes or []),
            "creds_path": CREDS_PATH,
            "sheet_id_present": bool(SHEET_ID),
            "events_sheet_name": EVENTS_SHEET_NAME,
        }), 200
    except Exception as e:
        logging.error("DEBUG CREDS ERROR: %s", e)
        return jsonify({"ok": False, "error": str(e), "creds_path": CREDS_PATH}), 500


@app.post("/track")
def track():
    """
    Receives tracking events from the frontend.
    Body: { user_id: "...", event_type: "...", metadata: {...} }
    """
    try:
        data = request.get_json(silent=True) or {}
        user_id = (data.get("user_id") or "").strip()
        event_type = (data.get("event_type") or "").strip()
        metadata = data.get("metadata") or {}

        if not user_id or not event_type:
            return jsonify({"ok": False, "error": "Missing user_id or event_type"}), 400

        log_event(user_id, event_type, metadata)
        return jsonify({"ok": True}), 200

    except Exception as e:
        logging.error("TRACK ERROR: %s", e)
        return jsonify({"ok": False, "error": str(e)}), 500


@app.get("/refresh")
def refresh():
    refresh_cache()
    return {"reloaded_symbols": len(SYMBOL_ROWS), "reloaded_seals": len(SEAL_ROWS)}


@app.post("/interpret")
def interpret():
    session_bootstrap()

    data = request.get_json(silent=True) or {}
    raw_text = (data.get("text") or "").strip()
    email_body = (data.get("email") or "").strip().lower() or None

    if not raw_text:
        return jsonify({"error": "Please paste your dream text."}), 400

    if email_body:
        set_email_in_session(email_body)

    email = effective_email(email_body) or None
    is_paid_user = user_is_paid(email)

    normalized_text = normalize_dream_text(raw_text)

    ending_window = extract_ending_window(raw_text, sentences=3)
    seal_obj = match_seal(ending_window)
    tone = tone_from_seal(seal_obj)

    # Airtable usage is source of truth when email exists
    if email:
        usage = get_usage(email)
        free_used = int(usage.get("free_used", 0))
    else:
        free_used = int(session.get("free_used", 0))

    free_left = max(0, FREE_QUOTA - free_used)

    # =====================
    # PAID USERS
    # =====================
    if is_paid_user:
        rows = find_best_rows_for_dream(normalized_text, max_hits=5)
        result = build_level2_result(normalized_text, rows)

        result["interpretation"] = polish_with_ai_voice(raw_text, result.get("interpretation", {}), tone)
        result["ai_voice_used"] = bool(USE_AI_VOICE)
        result["tone"] = tone

        result["access"] = "paid"
        result["is_paid"] = True
        result["free_uses_left"] = None

        receipt = build_receipt(seal_obj, result.get("symbols_matched", []))
        receipt["share_phrase"] = share_phrase_from_receipt(receipt, tone)
        result["receipt"] = receipt
        result["seal"] = seal_obj  # full seal

        try:
            if email:
                save_receipt(receipt, email=email)
        except Exception:
            pass

        result = format_interpretation_payload(result)
        return jsonify(result), 200

    # =====================
    # FREE USERS (quota -> email gate -> paywall)
    # =====================
    if free_used < FREE_QUOTA:
        rows = find_best_rows_for_dream(normalized_text, max_hits=5)
        result = build_level2_result(normalized_text, rows)

        result["interpretation"] = polish_with_ai_voice(raw_text, result.get("interpretation", {}), tone)
        result["ai_voice_used"] = bool(USE_AI_VOICE)
        result["tone"] = tone

        if email:
            free_used = increment_free_used(email)
        else:
            free_used += 1
            session["free_used"] = free_used

        free_left = max(0, FREE_QUOTA - free_used)

        result["access"] = "free"
        result["is_paid"] = False
        result["free_uses_left"] = free_left

        receipt = build_receipt(seal_obj, result.get("symbols_matched", []))
        receipt["share_phrase"] = share_phrase_from_receipt(receipt, tone)
        result["receipt"] = receipt
        result["seal"] = seal_preview(seal_obj)

        try:
            if email:
                save_receipt(receipt, email=email)
        except Exception:
            pass

        if free_left == 0:
            result["next_step"] = "email_gate"

        result = format_interpretation_payload(result)
        return jsonify(result), 200

    # quota exhausted: require email
    if not email:
        receipt = build_receipt(seal_obj, [])
        receipt["share_phrase"] = share_phrase_from_receipt(receipt, tone)

        payload = {
            "email_gate": True,
            "message": "You’ve used your free interpretations. Enter your email to continue.",
            "free_uses_left": 0,
            "tone": tone,
            "receipt": receipt,
            "seal": seal_preview(seal_obj)
        }
        payload = format_interpretation_payload(payload)
        return jsonify(payload), 200

    # email exists -> paywall
    weekly_url = f"{BACKEND_BASE_URL}/subscribe/create-checkout?plan=weekly&email={email}"
    monthly_url = f"{BACKEND_BASE_URL}/subscribe/create-checkout?plan=monthly&email={email}"

    receipt = build_receipt(seal_obj, [])
    receipt["share_phrase"] = share_phrase_from_receipt(receipt, tone)

    try:
        save_receipt(receipt, email=email)
    except Exception:
        pass

    payload = {
        "paywall": True,
        "message": f"Free limit reached ({FREE_QUOTA}). Subscribe to unlock full interpretations.",
        "free_uses_left": 0,
        "tone": tone,
        "checkout_weekly": weekly_url,
        "checkout_monthly": monthly_url,
        "receipt": receipt,
        "seal": seal_preview(seal_obj)
    }
    payload = format_interpretation_payload(payload)
    return jsonify(payload), 200


# =========================
# STRIPE CHECKOUT
# =========================
@app.route("/subscribe/create-checkout", methods=["GET", "POST"])
def create_checkout():
    """
    Supports GET (your paywall links) and POST (fetch/ajax).
    Returns JSON: {"url": "<stripe_checkout_url>"}
    """
    if not STRIPE_SECRET_KEY or not PRICE_WEEKLY or not PRICE_MONTHLY:
        logging.error("Stripe not configured. Missing secret key or price IDs.")
        return jsonify({"error": "Stripe is not configured on the server."}), 500

    data = request.get_json(silent=True) or {}
    plan = (request.args.get("plan") or data.get("plan") or "monthly").lower()
    email = (request.args.get("email") or data.get("email") or "").strip().lower()

    if plan not in ("weekly", "monthly"):
        plan = "monthly"

    price_id = PRICE_WEEKLY if plan == "weekly" else PRICE_MONTHLY

    success_url = f"{BACKEND_BASE_URL}/stripe/success?session_id={{CHECKOUT_SESSION_ID}}"
    cancel_url = SHOPIFY_INTERPRETER_URL

    try:
        session_obj = stripe.checkout.Session.create(
            mode="subscription",
            customer_email=email if email else None,
            line_items=[{"price": price_id, "quantity": 1}],
            success_url=success_url,
            cancel_url=cancel_url,
            client_reference_id=email or None,
            allow_promotion_codes=True,
        )
        return jsonify({"url": session_obj.url})
    except Exception as e:
        logging.error("Stripe checkout error: %s", e)
        return jsonify({"error": str(e)}), 500


@app.post("/stripe/webhook")
def stripe_webhook():
    """
    Configure Stripe to send: checkout.session.completed
    """
    if not STRIPE_WEBHOOK_SECRET:
        return "Webhook secret not configured", 500

    payload = request.data
    sig_header = request.headers.get("Stripe-Signature", "")

    try:
        event = stripe.Webhook.construct_event(
            payload=payload,
            sig_header=sig_header,
            secret=STRIPE_WEBHOOK_SECRET
        )
    except Exception as e:
        logging.error("Webhook signature verification failed: %s", e)
        return "Invalid signature", 400

    if event.get("type") == "checkout.session.completed":
        session_obj = event["data"]["object"]
        email = (
            (session_obj.get("customer_details") or {}).get("email")
            or session_obj.get("customer_email")
            or session_obj.get("client_reference_id")
            or ""
        ).strip().lower()

        if email:
            try:
                airtable_mark_paid(email)
                logging.info("Webhook marked paid (Airtable): %s", email)
            except Exception as e:
                logging.error("Airtable mark_paid failed: %s", e)

    return "ok", 200


@app.get("/stripe/success")
def stripe_success():
    session_id = request.args.get("session_id", "")
    if not session_id:
        return "Missing session_id", 400

    try:
        session_obj = stripe.checkout.Session.retrieve(session_id)
    except Exception as e:
        logging.error("Stripe retrieve error: %s", e)
        return "Error verifying payment.", 400

    email = (
        (session_obj.get("customer_details") or {}).get("email")
        or session_obj.get("customer_email")
        or session_obj.get("client_reference_id")
        or ""
    ).strip().lower()

    status = session_obj.get("status")
    payment_status = session_obj.get("payment_status")

    interpreter_url = SHOPIFY_INTERPRETER_URL

    if email and (status == "complete" or payment_status == "paid"):
        try:
            airtable_mark_paid(email)
        except Exception as e:
            logging.error("Airtable mark_paid failed: %s", e)

        mark_paid_in_session(email)

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Jamaican True Stories - Dream Interpreter PRO</title>
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <style>
    body {{
      margin: 0; padding: 0;
      font-family: system-ui,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
      background: radial-gradient(circle at top, #1c1510 0, #050505 55%);
      color: #f5f5f5;
      display: flex; align-items: center; justify-content: center;
      min-height: 100vh;
    }}
    .jts-card {{
      max-width: 480px; width: 90%;
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
      display:inline-flex; align-items:center; justify-content:center;
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
    .btn-main:hover {{ filter:brightness(1.05); }}
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
    <div class="jts-tag">Dream Interpreter - PRO Activated</div>
    <h1>Thank You for Supporting<br>Jamaican True Stories</h1>
    <p class="email">{email} is now <strong>PRO</strong>.</p>
    <p>You now have full access to deeper dream interpretations without hitting the free limit.</p>
    <p>Return to the Dream Interpreter to continue decoding your dreams with unlimited access.</p>
    <a class="btn-main" href="{interpreter_url}">Return to Dream Interpreter</a>
    <p class="note">Jamaican True Stories - Spiritual Dream Decoder</p>
  </div>
</body>
</html>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Payment Not Completed - Jamaican True Stories</title>
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <style>
    body {{
      margin: 0; padding: 0;
      font-family: system-ui,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
      background: radial-gradient(circle at top, #2a1515 0, #050505 55%);
      color: #f5f5f5;
      display: flex; align-items: center; justify-content: center;
      min-height: 100vh;
    }}
    .jts-card {{
      max-width: 440px; width: 90%;
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
      display:inline-flex; align-items:center; justify-content:center;
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
    .btn-main:hover {{ filter:brightness(1.05); }}
  </style>
</head>
<body>
  <div class="jts-card">
    <h1>Payment Not Completed</h1>
    <p>Your subscription payment was not confirmed.</p>
    <p>If this was a mistake, return to the Dream Interpreter and try again.</p>
    <a class="btn-main" href="{interpreter_url}">Back to Dream Interpreter</a>
  </div>
</body>
</html>"""


@app.get("/stripe/cancel")
def stripe_cancel():
    return "Payment cancelled. You can return to the Dream Interpreter."


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
