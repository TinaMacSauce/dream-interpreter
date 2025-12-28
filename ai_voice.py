# ai_voice.py
# Purpose: STRICT writing layer for Jamaican True Stories Dream Interpreter
# This file NEVER invents meaning.
# It ONLY rewrites what the engine already decided.
#
# UPDATE:
# - Accepts optional `tone` param: "warning" | "instruction" | "confirmation"
# - Uses tone ONLY to adjust writing style (NOT meaning)
# - Still enforces the same 3 labeled sections (hard parser)

import os
import re
from typing import Dict

from openai import OpenAI

# Uses env var OPENAI_API_KEY automatically if set on Render
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def _normalize_sections(text: str) -> Dict[str, str]:
    """
    Hard parser.
    If the model drifts, this function forces the 3 sections back into shape.
    """
    if not text:
        return {
            "spiritual_meaning": "",
            "effects_in_the_physical_realm": "",
            "what_to_do": "",
        }

    # Collapse whitespace to make regex reliable
    text = " ".join(str(text).split())

    patterns = {
        "spiritual_meaning": r"spiritual meaning:\s*(.*?)\s*(effects in the physical realm:|what to do:|$)",
        "effects_in_the_physical_realm": r"effects in the physical realm:\s*(.*?)\s*(what to do:|$)",
        "what_to_do": r"what to do:\s*(.*)$",
    }

    results = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            results[key] = match.group(1).strip(" .")
        else:
            results[key] = ""

    return results


def _tone_style_instructions(tone: str) -> str:
    """
    Tone changes ONLY the delivery, not the content.
    """
    t = (tone or "").strip().lower()

    if t == "warning":
        return (
            "- Tone mode: WARNING\n"
            "- Be direct, protective, and serious (but not dramatic).\n"
            "- Use clear caution language.\n"
            "- No fluff. No jokes. No extra claims.\n"
        )

    if t == "confirmation":
        return (
            "- Tone mode: CONFIRMATION\n"
            "- Be calm, reassuring, and steady.\n"
            "- Emphasize clarity, peace, and stability.\n"
            "- No hype. No extra claims.\n"
        )

    # default: instruction
    return (
        "- Tone mode: INSTRUCTION\n"
        "- Be grounded, practical, and clear.\n"
        "- Focus on actionable wording.\n"
        "- No extra claims.\n"
    )


def write_interpretation(
    dream_text: str,
    doctrine_summary: str,
    action_guidance: str,
    tone: str = "instruction",
    model: str = "gpt-4.1-mini",
) -> str:
    """
    STRICT writer.
    Output MUST contain exactly:
    - Spiritual Meaning:
    - Effects in the Physical Realm:
    - What to Do:

    `tone` is allowed to adjust WRITING STYLE ONLY.
    It must not change meaning or add content.
    """
    tone_instructions = _tone_style_instructions(tone)

    prompt = f"""
You are NOT a dream interpreter.
You are a WRITER for an existing doctrine-based system.

ABSOLUTE RULES:
- Do NOT add symbols.
- Do NOT add interpretations.
- Do NOT explain dreams yourself.
- Use ONLY the provided content.
- Do NOT add new claims, warnings, promises, timelines, or predictions.
- Output MUST contain EXACTLY these three labeled sections (no others):

Spiritual Meaning:
Effects in the Physical Realm:
What to Do:

Dream (context only):
{dream_text}

AUTHORIZED CONTENT (DO NOT ADD TO IT):
{doctrine_summary}

STYLE + DELIVERY:
{tone_instructions}

WRITING INSTRUCTIONS:
- Clarify wording
- Improve flow
- Beginner-friendly
- No metaphors
- No extra sections

ENGINE GUIDANCE (do NOT expand beyond it):
{action_guidance}
"""

    # OpenAI Responses API (works with openai>=1.x)
    response = client.responses.create(
        model=model,
        input=prompt
    )

    raw_text = getattr(response, "output_text", "") or ""
    sections = _normalize_sections(raw_text)

    # If model returns blanks (rare), fallback to parsing from doctrine_summary itself
    if not (sections["spiritual_meaning"] or sections["effects_in_the_physical_realm"] or sections["what_to_do"]):
        sections = _normalize_sections(doctrine_summary)

    # Guaranteed structure
    final_output = (
        f"Spiritual Meaning: {sections['spiritual_meaning']}\n\n"
        f"Effects in the Physical Realm: {sections['effects_in_the_physical_realm']}\n\n"
        f"What to Do: {sections['what_to_do']}"
    )

    return final_output
