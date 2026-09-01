import re
from typing import Any, Dict, List

from app.utils import normalize_text


RELATIONSHIP_TERMS = (
    "mother", "father", "mom", "mum", "dad", "sister", "brother",
    "son", "daughter", "child", "husband", "wife", "spouse", "friend",
    "aunt", "uncle", "grandmother", "grandfather", "grandma", "grandpa",
    "cousin", "niece", "nephew",
)

MULTIPLE_WORDS = {
    "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "several", "many", "multiple", "all", "both",
}


def _tokens(text: str) -> List[str]:
    return [token for token in normalize_text(text).split() if token]


def _has_teeth(words: List[str]) -> bool:
    return "tooth" in words or "teeth" in words


def _extract_owner(words: List[str]) -> Dict[str, str]:
    if not _has_teeth(words):
        return {"owner": "unknown", "owner_relationship": ""}

    for idx, token in enumerate(words):
        if token not in {"tooth", "teeth"}:
            continue

        window = words[max(0, idx - 4):idx]
        if any(value in window for value in ("my", "mine")):
            return {"owner": "dreamer", "owner_relationship": ""}

        for relationship in RELATIONSHIP_TERMS:
            if relationship in window:
                return {"owner": "other", "owner_relationship": relationship}

        if any(value in window for value in ("his", "her", "their", "someone", "stranger")):
            return {"owner": "other", "owner_relationship": ""}

    return {"owner": "unknown", "owner_relationship": ""}


def _extract_count(words: List[str]) -> str:
    if not _has_teeth(words):
        return "unknown"

    if "teeth" in words:
        for idx, token in enumerate(words):
            if token != "teeth":
                continue
            window = words[max(0, idx - 3):idx + 1]
            if any(value in window for value in MULTIPLE_WORDS):
                return "multiple"
        # Plural teeth is structurally multiple unless a construction explicitly
        # narrows to one tooth elsewhere.
        if "one" not in words and "single" not in words:
            return "multiple"

    for idx, token in enumerate(words):
        if token != "tooth":
            continue
        window = words[max(0, idx - 3):idx + 1]
        if any(value in window for value in MULTIPLE_WORDS):
            return "multiple"
        if any(value in window for value in ("one", "single", "a", "my")):
            return "one"

    return "unknown"


def _extract_pain(words: List[str]) -> str:
    if not _has_teeth(words):
        return "unknown"

    joined = " ".join(words)
    painless_patterns = (
        "without pain", "no pain", "did not hurt", "didn t hurt", "wasn t painful",
        "was not painful", "painless",
    )
    if any(pattern in joined for pattern in painless_patterns):
        return "painless"

    pain_patterns = (
        "with pain", "painful", "hurt", "hurting", "ached", "aching", "toothache",
    )
    if any(pattern in joined for pattern in pain_patterns):
        return "painful"

    return "unknown"


def _extract_positions(words: List[str]) -> List[str]:
    if not _has_teeth(words):
        return []

    positions: List[str] = []
    joined = " ".join(words)
    candidates = (
        ("front", ("front tooth", "front teeth")),
        ("back", ("back tooth", "back teeth", "molar", "molars")),
        ("upper", ("upper tooth", "upper teeth", "top tooth", "top teeth")),
        ("lower", ("lower tooth", "lower teeth", "bottom tooth", "bottom teeth")),
    )
    for label, phrases in candidates:
        if any(phrase in joined for phrase in phrases):
            positions.append(label)
    return positions


def extract_teeth_context(dream: str) -> Dict[str, Any]:
    """Extract factual Teeth modifiers without assigning cultural meanings.

    The output is intentionally structural. It does not infer death, illness,
    kinship classes from tooth position, or any other doctrine conclusion.
    """
    words = _tokens(dream)
    owner = _extract_owner(words)

    return {
        "has_teeth": _has_teeth(words),
        "owner": owner["owner"],
        "owner_relationship": owner["owner_relationship"],
        "count": _extract_count(words),
        "pain": _extract_pain(words),
        "positions": _extract_positions(words),
    }
