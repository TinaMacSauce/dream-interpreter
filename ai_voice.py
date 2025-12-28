# ai_voice.py
# Purpose: STRICT writing layer for Jamaican True Stories Dream Interpreter
# This file NEVER invents meaning.
# It ONLY rewrites what the engine already decided.
#
# UPDATE:
# - Accepts optional `tone` param: "warning" | "instruction" | "confirmation"
# - Uses tone ONLY to adjust writing style (NOT meaning)
# - Still enforces the same 3 labeled sections (hard parser)
# - NEW: Adds a "formatter" pass to remove duplicates + clean punctuation
# - NEW: Safer OpenAI client init via OPENAI_API_KEY (but still works if env is set)

import os
import re
from typing import Dict

from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


# -------------------------
# Formatting helpers (NEW)
# -------------------------
def _collapse_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def _dedupe_sentences(s: str) -> str:
    """
    Removes obvious repeated sentences while keeping order.
    Safe, content-preserving (doesn't invent, only removes repeats).
    """
    s = _
