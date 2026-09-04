import re
from typing import Any, Dict, List, Optional, Tuple

from app.utils import normalize_text


RELATIONSHIP_TERMS = (
    "mother", "father", "mom", "mum", "dad", "sister", "brother",
    "son", "daughter", "child", "husband", "wife", "spouse", "friend",
    "aunt", "uncle", "grandmother", "grandfather", "grandma", "grandpa",
    "cousin", "niece", "nephew",
)

TEETH_CONTEXT_VERSION = "teeth-context-v2"
RESTORATION_ATTEMPT_CONTRACT_VERSION = "restoration-attempt-binding/1.0"

MULTIPLE_WORDS = {
    "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "several", "many", "multiple", "all", "both",
}

TEETH_TOKENS = {"tooth", "teeth", "molar", "molars"}
GUM_TOKENS = {"gum", "gums"}
NEGATORS = {
    "not", "never", "no", "none", "neither",
    "isn", "wasn", "weren", "aren", "didn", "doesn", "don",
}

BLEEDING_CAUSE_TERMS = {
    "brush", "brushed", "brushing", "floss", "flossed", "flossing",
    "injury", "injured", "cut", "accident", "dentist", "dental",
}

BREAKAGE_TERMS = {"broken", "cracked", "chipped"}
DECAY_TERMS = {"rotten", "decayed", "decaying"}
GOLD_TERMS = {"gold", "golden"}

NON_HUMAN_TERMS = {
    "animal", "bird", "cat", "cow", "dog", "goat", "horse", "lion",
    "monkey", "pig", "snake", "tiger", "wolf",
}

ARTIFICIAL_TEETH_TERMS = {
    "denture", "dentures", "implant", "implants", "prosthetic", "prosthetics",
}

REPETITION_PATTERNS = (
    "again and again",
    "over and over",
    "kept happening",
    "keeps happening",
    "repeatedly",
    "recurring dream",
    "same dream again",
)

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

NEAR_MISS_FALLOUT_PATTERNS = (
    ("almost", "fell", "out"),
    ("nearly", "fell", "out"),
    ("almost", "came", "out"),
    ("nearly", "came", "out"),
    ("almost", "falling", "out"),
)

HYPOTHETICAL_FALLOUT_PATTERNS = (
    ("might", "fall", "out"),
    ("may", "fall", "out"),
    ("could", "fall", "out"),
    ("would", "fall", "out"),
    ("will", "fall", "out"),
    ("gonna", "fall", "out"),
    ("going", "to", "fall", "out"),
    ("might", "come", "out"),
    ("could", "come", "out"),
)

REPLACEMENT_GROWTH_PATTERNS = (
    "healthy tooth grew",
    "healthy tooth grew back",
    "new healthy tooth grew",
    "new healthy tooth grew back",
    "new tooth grew",
    "new tooth grew back",
    "tooth grew back",
)


def _tokens(text: str) -> List[str]:
    return [token for token in normalize_text(text).split() if token]


def _has_teeth(words: List[str]) -> bool:
    return any(token in TEETH_TOKENS for token in words)


def _has_gums(words: List[str]) -> bool:
    return any(token in GUM_TOKENS for token in words)


def _find_phrase_starts(words: List[str], phrase: tuple) -> List[int]:
    if not phrase or len(phrase) > len(words):
        return []
    width = len(phrase)
    return [
        idx
        for idx in range(len(words) - width + 1)
        if tuple(words[idx:idx + width]) == phrase
    ]


def _nearby_tooth_before(words: List[str], start: int, window: int = 8) -> bool:
    return any(token in TEETH_TOKENS for token in words[max(0, start - window):start])


def _affirmative_local_tooth_state(words: List[str], terms: set) -> bool:
    if not _has_teeth(words):
        return False

    for tooth_idx, token in enumerate(words):
        if token not in TEETH_TOKENS:
            continue

        start = max(0, tooth_idx - 5)
        end = min(len(words), tooth_idx + 6)
        for idx in range(start, end):
            if words[idx] not in terms:
                continue
            preceding = words[max(start, idx - 3):idx]
            if any(value in NEGATORS for value in preceding):
                continue
            return True

    return False


def _extract_owner(words: List[str]) -> Dict[str, str]:
    if not _has_teeth(words):
        return {"owner": "unknown", "owner_relationship": ""}

    for idx, token in enumerate(words):
        if token not in TEETH_TOKENS:
            continue

        window_start = max(0, idx - 7)
        window = words[window_start:idx]

        # Use the closest ownership marker. This keeps "my sister's tooth" tied
        # to the sister while correctly treating "my sister pulled my tooth" as
        # the dreamer's tooth.
        markers: List[tuple] = []
        for offset, value in enumerate(window):
            absolute = window_start + offset
            if value in ("my", "mine"):
                markers.append((absolute, "dreamer", ""))
            elif value in ("his", "her", "their", "someone", "stranger"):
                markers.append((absolute, "other", ""))
            elif value in RELATIONSHIP_TERMS:
                markers.append((absolute, "other", value))

        if markers:
            _position, owner, relationship = max(markers, key=lambda marker: marker[0])
            return {"owner": owner, "owner_relationship": relationship}

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
    return _affirmative_local_tooth_state(words, {"loose", "wobbly"})


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
        r"\bblood (?:was )?(?:on|covering) (?:my |the |a |one )?(?:fallen )?(?:tooth|teeth|molar|molars)\b",
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


_ATTEMPT_PATTERN = re.compile(
    r"\b(?P<actor>i|she|he|they|my\s+(?:sister|brother|mother|father|mom|mum|dad|"
    r"aunt|uncle|friend)|the\s+dentist|a\s+dentist)\s+"
    r"(?:(?P<modal>would|might|could)\s+)?(?:(?P<negator>never|did\s+not|didn't)\s+)?"
    r"(?P<attempt>try|tried|attempt|attempted)\s+"
    r"(?:to\s+)?(?P<action>put|place|fit|push|insert|putting|placing|fitting|pushing|inserting)\s+"
    r"(?P<target>(?:(?:the|my|her|his)\s+)?(?:same\s+)?(?:left\s+|right\s+)?"
    r"(?:tooth|teeth)|it)\s+back\b",
    re.IGNORECASE,
)


def _quoted_ranges(text: str) -> List[Tuple[int, int]]:
    ranges: List[Tuple[int, int]] = []
    opened: Optional[int] = None
    for index, char in enumerate(text):
        if char not in {'"', "“", "”"}:
            continue
        if opened is None:
            opened = index
        else:
            ranges.append((opened, index + 1))
            opened = None
    return ranges


def _relationship_actor(actor_text: str, prefix: str) -> str:
    actor = normalize_text(actor_text)
    if actor == "i":
        return "dreamer"
    for relationship in RELATIONSHIP_TERMS:
        if relationship in actor:
            return relationship
    if actor in {"she", "he", "they"}:
        normalized_prefix = normalize_text(prefix)
        candidates = [
            (normalized_prefix.rfind(relationship), relationship)
            for relationship in RELATIONSHIP_TERMS
            if relationship in normalized_prefix
        ]
        if candidates:
            return max(candidates)[1]
    if "dentist" in actor:
        return "dentist"
    return "ambiguous"


def _attempt_owner_and_target(
    *,
    dream: str,
    match: re.Match,
    actor: str,
    channel: str,
) -> Tuple[str, str, str]:
    prefix = normalize_text(dream[:match.start()])
    target_text = normalize_text(match.group("target"))

    if channel == "quoted_speech" and target_text == "it":
        return "ambiguous", "ambiguous", "low"

    if "her tooth" in target_text and actor not in {"dreamer", "ambiguous"}:
        owner = actor
    elif "his tooth" in target_text and actor not in {"dreamer", "ambiguous"}:
        owner = actor
    elif re.search(r"my tooth and my (?:sister|brother|mother|father|aunt|uncle|friend) s tooth fell out", prefix):
        return "ambiguous", "ambiguous", "low"
    else:
        dreamer_loss = max(prefix.rfind("my tooth fell out"), prefix.rfind("my left tooth fell out"))
        other_losses = [
            (prefix.rfind(f"my {relationship} s tooth fell out"), relationship)
            for relationship in RELATIONSHIP_TERMS
        ]
        other_index, other_owner = max(other_losses)
        if other_index > dreamer_loss:
            owner = other_owner
        elif dreamer_loss >= 0:
            owner = "dreamer"
        elif actor not in {"ambiguous", "dentist"}:
            owner = actor
        else:
            return "ambiguous", "ambiguous", "low"

    if "left" in target_text or "left tooth" in prefix[-80:]:
        target = "left-tooth-1"
    elif owner == "dreamer":
        normalized_dream = normalize_text(dream)
        multiple_owners = bool(
            re.search(
                r"my (?:sister|brother|mother|father|aunt|uncle|friend) s tooth fell out",
                normalized_dream,
            )
        )
        target = "dreamer-tooth-1" if multiple_owners else "tooth-1"
    else:
        target = f"{owner}-tooth-1"
    return owner, target, "high"


def _extract_restoration_attempt_records(dream: str) -> List[Dict[str, Any]]:
    """Return every reinsertion attempt as a scoped structural event record."""
    quote_ranges = _quoted_ranges(dream)
    records: List[Dict[str, Any]] = []

    for index, match in enumerate(_ATTEMPT_PATTERN.finditer(dream), start=1):
        prefix = dream[:match.start()]
        normalized_prefix = normalize_text(prefix)
        matched_text = match.group(0)
        in_quote = any(start <= match.start() < end for start, end in quote_ranges)
        actor = _relationship_actor(match.group("actor"), prefix)
        if in_quote:
            speaker_candidates = [
                (normalized_prefix.rfind(relationship), relationship)
                for relationship in RELATIONSHIP_TERMS
                if relationship in normalized_prefix
            ]
            if speaker_candidates:
                actor = max(speaker_candidates)[1]
        reported = bool(re.search(r"\b(?:told me(?: that)?|said that)\b", normalized_prefix[-120:]))
        waking = bool(re.search(r"\bafter (?:i )?woke up\b|\bafter waking\b", normalized_prefix[-160:]))
        imagined = "imagination" in normalize_text(dream[match.start():match.end() + 80])
        conditional = bool(match.group("modal")) or bool(
            re.search(r"(?:^|[.!?])\s*if\b", normalize_text(prefix)[-180:])
        )
        negated = bool(match.group("negator"))

        if in_quote:
            channel = "quoted_speech"
            phase = "dream_speech_or_thought"
            modality = "quoted"
            actuality = "reported_nonactual"
        elif reported:
            channel = "reported_speech"
            phase = "dream_speech_or_thought"
            modality = "reported"
            actuality = "reported_nonactual"
        elif waking:
            channel = "narrative"
            phase = "waking"
            modality = "imagined" if imagined else "waking_action"
            actuality = "nonactual"
        elif conditional:
            channel = "thought_or_hypothetical"
            phase = "dream_speech_or_thought"
            modality = "conditional_hypothetical"
            actuality = "nonactual"
        elif negated:
            channel = "narrative"
            phase = "dream"
            modality = "negated"
            actuality = "not_actual"
        else:
            channel = "narrative"
            phase = "dream"
            modality = "experienced"
            actuality = "actual_attempt"

        owner, target, confidence = _attempt_owner_and_target(
            dream=dream,
            match=match,
            actor=actor,
            channel=channel,
        )
        reasons: List[str] = []
        if negated:
            reasons.append("negated_attempt")
        if actuality != "actual_attempt":
            reasons.append("nonactual_or_out_of_dream")
        if channel != "narrative":
            reasons.append("ineligible_channel")
        if phase != "dream":
            reasons.append("ineligible_phase")
        if owner == "ambiguous" or target == "ambiguous":
            reasons.append("ambiguous_binding")
        eligible = not reasons
        chain = None if channel in {"quoted_speech", "reported_speech"} or conditional else (
            f"chain-{target}" if target != "ambiguous" else None
        )

        records.append(
            {
                "attempt_id": f"attempt-{index}",
                "action": "manual_reinsertion_attempt",
                "actor_id_or_ambiguous": actor,
                "target_tooth_ids_or_ambiguous": [target] if target != "ambiguous" else "ambiguous",
                "owner_id_or_ambiguous": owner,
                "event_chain_id_or_null": chain,
                "scene_id": "scene-1",
                "phase": phase,
                "channel": channel,
                "polarity": "negated" if negated else "affirmed",
                "modality": modality,
                "actuality": actuality,
                "completion": "attempted_only",
                "source_span": {
                    "start": match.start(),
                    "end": match.end(),
                    "text": matched_text,
                },
                "binding_confidence": confidence,
                "narration_eligibility": eligible,
                "ineligibility_reasons": reasons,
            }
        )
    return records


def _extract_near_miss_loss(words: List[str]) -> bool:
    for phrase in NEAR_MISS_FALLOUT_PATTERNS:
        for start in _find_phrase_starts(words, phrase):
            if _nearby_tooth_before(words, start):
                return True
    return False


def _extract_hypothetical_loss(words: List[str]) -> bool:
    joined = " ".join(words)
    if re.search(r"(?:^|\b)if\b[^.?!]{0,80}\b(?:tooth|teeth|molar|molars)\b[^.?!]{0,30}\b(?:fell|came) out\b", joined):
        return True
    for phrase in HYPOTHETICAL_FALLOUT_PATTERNS:
        for start in _find_phrase_starts(words, phrase):
            if _nearby_tooth_before(words, start):
                return True
    return False


def _extract_replacement_growth(words: List[str]) -> bool:
    joined = " ".join(words)
    growth_indexes = [joined.find(pattern) for pattern in REPLACEMENT_GROWTH_PATTERNS if pattern in joined]
    if not growth_indexes:
        return False

    fallout_indexes = [
        joined.find(pattern)
        for pattern in ("fell out", "came out")
        if pattern in joined
    ]
    if not fallout_indexes:
        return False

    earliest_fallout = min(index for index in fallout_indexes if index >= 0)
    return any(index > earliest_fallout for index in growth_indexes if index >= 0)


def _extract_removal_actor(words: List[str]) -> str:
    """Return the explicit actor for a completed tooth-removal event."""
    if not _has_teeth(words):
        return ""

    joined = " ".join(words)
    tooth = r"(?:tooth|teeth|molar|molars)"
    self_patterns = (
        rf"\bi (?:pulled|yanked|removed|extracted|took) (?:out )?(?:one of )?(?:my )?{tooth}(?: out)?\b",
        rf"\bi (?:pulled|yanked) (?:one of )?(?:my )?{tooth} out\b",
    )
    if any(re.search(pattern, joined) for pattern in self_patterns):
        return "self"

    actor = (
        r"(?:someone|somebody|stranger|dentist|doctor|man|woman|he|she|they|"
        r"my (?:mother|father|mom|mum|dad|sister|brother|son|daughter|child|"
        r"husband|wife|spouse|friend|aunt|uncle|grandmother|grandfather|"
        r"grandma|grandpa|cousin|niece|nephew))"
    )
    external_patterns = (
        rf"\b{actor} (?:pulled|yanked|removed|extracted|took) (?:out )?(?:one of )?(?:my )?{tooth}(?: out)?\b",
        rf"\b(?:my )?{tooth} (?:was|were) (?:pulled|yanked|removed|extracted|taken) (?:out )?by {actor}\b",
    )
    if any(re.search(pattern, joined) for pattern in external_patterns):
        return "other"

    return ""


def _extract_returned_same_tooth_firm(words: List[str]) -> bool:
    """Detect a narrow, factual same-tooth terminal restoration state."""
    if not _has_teeth(words):
        return False

    joined = " ".join(words)
    return_patterns = (
        r"\b(?:the )?same tooth (?:came|went|was put|was fitted|fit|fitted|returned) back\b",
        r"\b(?:i )?put (?:the )?same tooth back\b",
        r"\b(?:the )?tooth (?:came|went|was put|was fitted|fit|fitted|returned) back into (?:the )?same socket\b",
        r"\b(?:the )?tooth (?:came|went|was put|was fitted|fit|fitted|returned) firmly back\b",
    )
    firm_patterns = (
        r"\bfirm\b",
        r"\bfirmly\b",
        r"\bsecure\b",
        r"\bsecurely\b",
        r"\btight\b",
        r"\bstayed in place\b",
        r"\bfitted perfectly\b",
    )
    return (
        any(re.search(pattern, joined) for pattern in return_patterns)
        and any(re.search(pattern, joined) for pattern in firm_patterns)
    )


def _extract_repetition(words: List[str]) -> bool:
    if not (_has_teeth(words) or _has_gums(words)):
        return False
    joined = " ".join(words)
    return any(pattern in joined for pattern in REPETITION_PATTERNS)


def _extract_subject_class(words: List[str]) -> str:
    if any(token in ARTIFICIAL_TEETH_TERMS for token in words):
        return "artificial"

    for idx, token in enumerate(words):
        if token not in TEETH_TOKENS:
            continue
        if any(value in NON_HUMAN_TERMS for value in words[max(0, idx - 5):idx]):
            return "non_human"

    return "human_or_unspecified"


def extract_teeth_context(dream: str) -> Dict[str, Any]:
    """Extract factual Teeth modifiers without assigning cultural meanings.

    The output is intentionally structural. It does not infer death, illness,
    kinship classes from tooth position, or any other doctrine conclusion.
    """
    words = _tokens(dream)
    owner = _extract_owner(words)
    gum_bleeding = _extract_gum_bleeding(words)
    removal_actor = _extract_removal_actor(words)
    attempt_records = _extract_restoration_attempt_records(dream)
    eligible_attempt_ids = [
        record["attempt_id"]
        for record in attempt_records
        if record["narration_eligibility"]
    ]

    return {
        "has_teeth": _has_teeth(words),
        "has_teeth_cluster": _has_teeth(words) or _has_gums(words),
        "owner": owner["owner"],
        "owner_relationship": owner["owner_relationship"],
        "count": _extract_count(words),
        "pain": _extract_pain(words),
        "positions": _extract_positions(words),
        "loose_or_wobbly": _extract_loose_state(words),
        "broken_or_cracked": _affirmative_local_tooth_state(words, BREAKAGE_TERMS),
        "rotten_or_decayed": _affirmative_local_tooth_state(words, DECAY_TERMS),
        "gold_teeth": _affirmative_local_tooth_state(words, GOLD_TERMS),
        "near_miss_loss": _extract_near_miss_loss(words),
        "hypothetical_loss": _extract_hypothetical_loss(words),
        "gum_bleeding": gum_bleeding,
        "blood_on_tooth": _extract_blood_on_tooth(words),
        "bleeding_physical_cause": _extract_bleeding_physical_cause(words, gum_bleeding),
        "restorative_state": _extract_restorative_state(words),
        "restoration_attempted": bool(eligible_attempt_ids),
        "restoration_attempt_contract_version": RESTORATION_ATTEMPT_CONTRACT_VERSION,
        "restoration_attempt_records": attempt_records,
        "restoration_attempt_contributing_ids": eligible_attempt_ids,
        "replacement_growth": _extract_replacement_growth(words),
        "removal_actor": removal_actor,
        "explicit_pull_removal": bool(removal_actor),
        "returned_same_tooth_firm": _extract_returned_same_tooth_firm(words),
        "repetition": _extract_repetition(words),
        "subject_class": _extract_subject_class(words),
        "context_version": TEETH_CONTEXT_VERSION,
    }
