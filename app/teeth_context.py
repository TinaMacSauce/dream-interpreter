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

TEETH_TOKENS = {"tooth", "teeth", "molar", "molars"}
GUM_TOKENS = {"gum", "gums"}

BLEEDING_CAUSE_TERMS = {
    "brush", "brushed", "brushing", "floss", "flossed", "flossing",
    "injury", "injured", "cut", "accident", "dentist", "dental",
}

RESTORATION_PATTERNS = (
    "became firm again",
    "became firm",
    "was firm again",
    "turned firm again",
    "stopped wobbling",
    "stopped being loose",
    "was no longer loose",
    "wasn t loose anymore",
    "was not loose anymore",
    "tightened up",
)


def _tokens(text: str) -> List[str]:
    return [token for token in normalize_text(text).split() if token]


def _has_teeth(words: List[str]) -> bool:
    return any(token in TEETH_TOKENS for token in words)


def _has_gums(words: List[str]) -> bool:
    return any(token in GUM_TOKENS for token in words)


def _extract_owner(words: List[str]) -> Dict[str, str]:
    if not _has_teeth(words):
        return {"owner": "unknown", "owner_relationship": ""}

    for idx, token in enumerate(words):
        if token not in TEETH_TOKENS:
            continue

        window = words[max(0, idx - 4):idx]

        # A named relationship must outrank a nearby first-person possessive.
        # For example, "my sister's tooth" belongs to the sister, not the dreamer.
        for relationship in RELATIONSHIP_TERMS:
            if relationship in window:
                return {"owner": "other", "owner_relationship": relationship}

        if any(value in window for value in ("his", "her", "their", "someone", "stranger")):
            return {"owner": "other", "owner_relationship": ""}

        if any(value in window for value in ("my", "mine")):
            return {"owner": "dreamer", "owner_relationship": ""}

    return {"owner": "unknown", "owner_relationship": ""}


def _extract_count(words: List[str]) -> str:
    if not _has_teeth(words):
        return "unknown"

    joined = " ".join(words)

    # Handle "one of my front teeth", "one of her lower teeth", etc. Position
    # words may sit between the possessive phrase and the Teeth token, so an
    # exact phrase check such as "one of my teeth" is too narrow.
    for idx, token in enumerate(words):
        if token not in TEETH_TOKENS:
            continue
        window = words[max(0, idx - 6):idx]
        if "one" in window and "of" in window and window.index("one") < window.index("of"):
            return "one"

    if "single tooth" in joined or "one tooth" in joined or "a tooth" in joined:
        return "one"

    for idx, token in enumerate(words):
        if token not in TEETH_TOKENS:
            continue
        window = words[max(0, idx - 3):idx + 1]
        if any(value in window for value in MULTIPLE_WORDS):
            return "multiple"

    if "teeth" in words or "molars" in words:
        return "multiple"
    if "tooth" in words or "molar" in words:
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

    # Strongly physical wording can be accepted directly in a Teeth dream.
    physical_pain_patterns = (
        "with pain", "painful", "aching", "ached", "toothache", "sore tooth",
        "tooth was sore", "teeth were sore",
    )
    if any(pattern in joined for pattern in physical_pain_patterns):
        return "painful"

    # "Hurt" is ambiguous because it can describe an emotional reaction. Only
    # treat it as tooth pain when it appears locally around the tooth event and
    # there is no explicit emotional-hurt phrase.
    emotional_hurt_patterns = (
        "felt hurt", "emotionally hurt", "hurt emotionally", "hurt inside",
        "felt emotionally hurt",
    )
    if any(pattern in joined for pattern in emotional_hurt_patterns):
        return "unknown"

    for idx, token in enumerate(words):
        if token not in TEETH_TOKENS:
            continue
        window = words[max(0, idx - 4):min(len(words), idx + 9)]
        if any(value in window for value in ("hurt", "hurting")):
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


def _extract_loose_state(words: List[str]) -> bool:
    if not _has_teeth(words):
        return False

    joined = " ".join(words)
    patterns = (
        r"\b(?:loose|wobbly) (?:tooth|teeth|molar|molars)\b",
        r"\b(?:tooth|teeth|molar|molars) (?:was|were|felt|became|is|are)? ?(?:loose|wobbly)\b",
    )
    return any(re.search(pattern, joined) for pattern in patterns)


def _extract_gum_bleeding(words: List[str]) -> bool:
    if not _has_gums(words):
        return False

    joined = " ".join(words)
    patterns = (
        r"\bgums? (?:(?:was|were|is|are|started|starts|began) )?(?:bleeding|bled)\b",
        r"\b(?:bleeding|bloody) gums?\b",
        r"\bblood (?:was )?(?:from|on|around) (?:my |the )?gums?\b",
    )
    return any(re.search(pattern, joined) for pattern in patterns)


def _extract_blood_on_tooth(words: List[str]) -> bool:
    if not _has_teeth(words):
        return False

    joined = " ".join(words)
    patterns = (
        r"\bblood (?:was )?(?:on|covering) (?:my |the |a |one )?(?:tooth|teeth|molar|molars)\b",
        r"\b(?:tooth|teeth|molar|molars) (?:was|were)? ?(?:covered in blood|bloody|bleeding)\b",
        r"\b(?:bloody) (?:tooth|teeth|molar|molars)\b",
    )
    return any(re.search(pattern, joined) for pattern in patterns)


def _extract_bleeding_physical_cause(words: List[str], gum_bleeding: bool) -> bool:
    if not gum_bleeding:
        return False
    return any(token in BLEEDING_CAUSE_TERMS for token in words)


def _extract_restorative_state(words: List[str]) -> bool:
    joined = " ".join(words)
    return any(pattern in joined for pattern in RESTORATION_PATTERNS)


def extract_teeth_context(dream: str) -> Dict[str, Any]:
    """Extract factual Teeth modifiers without assigning cultural meanings.

    The output is intentionally structural. It does not infer death, illness,
    kinship classes from tooth position, or any other doctrine conclusion.
    """
    words = _tokens(dream)
    owner = _extract_owner(words)
    gum_bleeding = _extract_gum_bleeding(words)

    return {
        "has_teeth": _has_teeth(words),
        "has_teeth_cluster": _has_teeth(words) or _has_gums(words),
        "owner": owner["owner"],
        "owner_relationship": owner["owner_relationship"],
        "count": _extract_count(words),
        "pain": _extract_pain(words),
        "positions": _extract_positions(words),
        "loose_or_wobbly": _extract_loose_state(words),
        "gum_bleeding": gum_bleeding,
        "blood_on_tooth": _extract_blood_on_tooth(words),
        "bleeding_physical_cause": _extract_bleeding_physical_cause(words, gum_bleeding),
        "restorative_state": _extract_restorative_state(words),
    }
