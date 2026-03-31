import re
from typing import Any, Dict, List, Optional, Tuple


def normalize_header(value: str) -> str:
    value = (value or "").strip().lower()
    value = re.sub(r"[^a-z0-9\s_]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def normalize_text(value: str) -> str:
    value = (value or "").lower()
    value = re.sub(r"[^a-z0-9\s]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def normalize_email(email: str) -> str:
    return (email or "").strip().lower()


def validate_email(email: str) -> bool:
    email = (email or "").strip()
    return bool(email and "@" in email and "." in email.split("@")[-1])


def clean_sentence(value: str) -> str:
    if not value:
        return ""
    value = str(value).strip()
    value = re.sub(r"\s+", " ", value)
    value = value.replace(" ,", ",").replace(" .", ".").replace(" :", ":").replace(" ;", ";")
    value = re.sub(r"([.!?]){2,}", r"\1", value)
    return value


def strip_trailing_punct(value: str) -> str:
    return clean_sentence(value).rstrip(" .!?\t\r\n")


def ensure_terminal_punct(value: str) -> str:
    value = clean_sentence(value)
    if not value:
        return ""
    if value[-1] not in ".!?":
        value += "."
    return value


def capitalize_first(value: str) -> str:
    value = clean_sentence(value)
    if not value:
        return ""
    return value[:1].upper() + value[1:]


def dedupe_words_soft(text: str) -> str:
    text = clean_sentence(text)
    if not text:
        return ""

    words = text.split()
    out: List[str] = []
    prev = ""

    for word in words:
        wl = word.lower().strip(".,;:!?")
        if wl and wl == prev:
            continue
        out.append(word)
        prev = wl

    return " ".join(out)


def sentence(text: str) -> str:
    text = dedupe_words_soft(text)
    text = capitalize_first(text)
    return ensure_terminal_punct(text)


def compress_phrase_list(parts: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()

    for part in parts:
        part = strip_trailing_punct(part)
        if not part:
            continue
        key = normalize_text(part)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(part)

    return out


def human_join(parts: List[str]) -> str:
    parts = compress_phrase_list(parts)
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return f"{parts[0]} and {parts[1]}"
    return ", ".join(parts[:-1]) + f", and {parts[-1]}"


def tokenize_words(value: str) -> List[str]:
    return [word for word in normalize_text(value).split() if word]


def compile_boundary_regex(token: str) -> re.Pattern:
    token_n = normalize_text(token)
    if not token_n:
        return re.compile(r"(?!x)x")
    return re.compile(rf"(?<!\w){re.escape(token_n)}(?!\w)")


def contains_phrase(text_norm: str, phrase: str) -> bool:
    if not text_norm or not phrase:
        return False
    return bool(compile_boundary_regex(phrase).search(text_norm))


def find_phrase_spans(text: str, phrase: str) -> List[Tuple[int, int]]:
    text_n = normalize_text(text)
    phrase_n = normalize_text(phrase)

    if not text_n or not phrase_n:
        return []

    rx = re.compile(rf"(?<!\w){re.escape(phrase_n)}(?!\w)")
    return [match.span() for match in rx.finditer(text_n)]


def spans_overlap(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
    return not (a[1] <= b[0] or b[1] <= a[0])


def is_span_blocked(span: Tuple[int, int], used_spans: List[Tuple[int, int]]) -> bool:
    for used in used_spans:
        if spans_overlap(span, used):
            return True
    return False


def split_sentences(text: str) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    chunks = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [chunk.strip() for chunk in chunks if chunk.strip()]


def extract_dream_ending_text(dream: str) -> str:
    sentences = split_sentences(dream)
    if not sentences:
        return dream.strip()
    if len(sentences) == 1:
        return sentences[-1]
    return " ".join(sentences[-2:]).strip()


def contains_any_phrase(text: str, phrases: List[str]) -> bool:
    text_n = normalize_text(text)
    if not text_n:
        return False
    return any(contains_phrase(text_n, phrase) for phrase in phrases if phrase)


def normalize_yes_no(value: str) -> bool:
    return str(value or "").strip().lower() in {"yes", "y", "true", "1", "active", "on"}


def row_get(row: Dict[str, Any], *keys: str) -> str:
    normalized = {normalize_header(k): v for k, v in row.items()}
    for key in keys:
        nk = normalize_header(key)
        if nk in normalized and str(normalized.get(nk, "")).strip():
            return str(normalized.get(nk, "")).strip()
    return ""


def col_to_a1(col_num: int) -> str:
    out = ""
    while col_num > 0:
        col_num, rem = divmod(col_num - 1, 26)
        out = chr(65 + rem) + out
    return out or "A"


def safe_debug_payload_preview(data: Dict[str, Any], max_len: int = 500) -> Dict[str, Any]:
    try:
        preview: Dict[str, Any] = {}
        for key, value in (data or {}).items():
            if isinstance(value, str):
                preview[key] = value[:max_len] + ("..." if len(value) > max_len else "")
            else:
                preview[key] = value
        return preview
    except Exception:
        return {"_debug_preview_error": True}


def validate_dream_text(dream: str, min_length: int, max_length: int) -> Optional[str]:
    if not dream:
        return "Missing 'dream' or 'text'."
    if len(dream.strip()) < min_length:
        return f"Dream text is too short. Please enter at least {min_length} characters."
    if len(dream) > max_length:
        return f"Dream text is too long. Maximum allowed is {max_length} characters."
    return None


def normalize_action_phrase(text: str) -> str:
    text = strip_trailing_punct(text)
    if not text:
        return ""

    text = re.sub(r"\s+", " ", text).strip().lower()

    replacements = {
        "is chasing": "chasing",
        "are chasing": "chasing",
        "was chasing": "chasing",
        "were chasing": "chasing",
        "is following": "following",
        "are following": "following",
        "was following": "following",
        "were following": "following",
        "is attacking": "attacking",
        "are attacking": "attacking",
        "was attacking": "attacking",
        "were attacking": "attacking",
        "is fighting": "fighting",
        "are fighting": "fighting",
        "was fighting": "fighting",
        "were fighting": "fighting",
        "is biting": "biting",
        "are biting": "biting",
        "was biting": "biting",
        "were biting": "biting",
        "is watching": "watching",
        "are watching": "watching",
        "was watching": "watching",
        "were watching": "watching",
        "is crying": "crying",
        "are crying": "crying",
        "was crying": "crying",
        "were crying": "crying",
        "is running": "running",
        "are running": "running",
        "was running": "running",
        "were running": "running",
        "is hiding": "hiding",
        "are hiding": "hiding",
        "was hiding": "hiding",
        "were hiding": "hiding",
        "is speaking": "speaking",
        "are speaking": "speaking",
        "was speaking": "speaking",
        "were speaking": "speaking",
        "is looking at": "looking at",
        "are looking at": "looking at",
        "was looking at": "looking at",
        "were looking at": "looking at",
        "is standing": "standing",
        "are standing": "standing",
        "was standing": "standing",
        "were standing": "standing",
    }

    return replacements.get(text, text)


def normalize_effect_phrase(text: str) -> str:
    text = strip_trailing_punct(text)
    if not text:
        return ""

    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ""

    return text[:1].lower() + text[1:]
