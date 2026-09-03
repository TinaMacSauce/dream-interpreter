from __future__ import annotations

import re
from copy import deepcopy
from typing import Any, Dict, List, Mapping, Sequence

from app.teeth_registry import (
    EXPECTED_CONTENT_REVISION,
    EXPECTED_DOCTRINE_VERSION,
    EXPECTED_RULES,
    REGISTRY_CONTRACT_VERSION,
)


SOURCE_LAYER = "jts_spiritual_or_editorial_doctrine"
NON_GUARANTEE_POLICY = (
    "tradition_based_non_guaranteed_no_diagnosis_no_culprit_no_causation"
)

BASE_WARNING_RULE_IDS = {
    "TEETH-FALLOUT-OWN",
    "TEETH-FALLOUT-OTHER",
    "TEETH-STATE-LOOSE",
    "TEETH-STATE-BROKEN",
    "TEETH-STATE-ROTTEN",
    "TEETH-OMEN-GUM-BLOOD",
}
MODIFIER_RULE_IDS = {
    "TEETH-FALLOUT-ONE",
    "TEETH-FALLOUT-MULTIPLE",
    "TEETH-PULL-SELF",
    "TEETH-PULL-EXTERNAL",
    "TEETH-MOD-PAIN",
    "TEETH-MOD-PAINLESS",
    "TEETH-MOD-BLOOD",
    "TEETH-MOD-GOLD",
    "TEETH-MOD-REPETITION",
    "TEETH-SUBJECT-EXPLICIT",
}
STRUCTURAL_RULE_IDS = {"TEETH-END-TERMINAL"}

EXPECTED_ACTIVE_RULE_IDS = {
    rule_id for rule_id, status, active in EXPECTED_RULES.values() if active
}
EXPECTED_UNRESOLVED_RULE_IDS = {
    rule_id for rule_id, status, active in EXPECTED_RULES.values() if not active
}


class TeethProvenanceValidationError(ValueError):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def _exact_span(text: str, patterns: Sequence[str]) -> str:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(0).strip(" ,.;:")
    return ""


def _fallback_tooth_span(text: str) -> str:
    return _exact_span(text, (r"\b(?:tooth|teeth|molar|molars)\b",)) or text.strip()


def _owner_id(context: Mapping[str, Any]) -> str:
    if context.get("owner") == "dreamer":
        return "dreamer"
    if context.get("owner") == "other":
        return str(context.get("owner_relationship") or "other_person")
    return "unknown_owner"


def _actor_id(text: str, context: Mapping[str, Any]) -> str:
    if context.get("removal_actor") == "self":
        return "dreamer"
    if context.get("removal_actor") != "other":
        return ""
    span = _exact_span(
        text,
        (
            r"\bmy (mother|father|mom|mum|dad|sister|brother|son|daughter|child|husband|wife|spouse|friend|aunt|uncle|grandmother|grandfather|grandma|grandpa|cousin|niece|nephew)\b",
            r"\b(someone|somebody|stranger|dentist|doctor|man|woman|he|she|they)\b",
        ),
    )
    return re.sub(r"(?i)^my\s+", "", span).lower() if span else "other_actor"


def _rule_span(rule_id: str, text: str, context: Mapping[str, Any]) -> str:
    patterns: Dict[str, Sequence[str]] = {
        "TEETH-FALLOUT-OWN": (
            r"\bmy\s+(?:(?:upper|lower|front|back|broken|cracked|rotten|decayed|gold)\s+)*(?:tooth|teeth|molar|molars)\s+(?:fell|came)\s+out\b",
            r"\bmy\s+(?:tooth|teeth|molar|molars)\b",
        ),
        "TEETH-FALLOUT-OTHER": (
            r"\bmy\s+[a-z]+(?:'s|s')\s+(?:tooth|teeth|molar|molars)\s+(?:fell|came)\s+out\b",
            r"\b(?:his|her|their)\s+(?:tooth|teeth|molar|molars)\s+(?:fell|came)\s+out\b",
        ),
        "TEETH-FALLOUT-ONE": (r"\btooth\b", r"\bmolar\b"),
        "TEETH-FALLOUT-MULTIPLE": (
            r"\b(?:two|three|four|five|six|seven|eight|nine|ten|several|many|multiple|all|both)\s+(?:of\s+)?(?:my\s+|his\s+|her\s+|their\s+)?(?:upper\s+|lower\s+|front\s+|back\s+)?(?:teeth|molars)\b",
            r"\b(?:teeth|molars)\b",
        ),
        "TEETH-MOD-PAIN": (
            r"\bhurt(?:\s+badly)?\b",
            r"\b(?:with pain|painful|aching|ached|toothache)\b",
        ),
        "TEETH-MOD-PAINLESS": (
            r"\b(?:without pain|no pain|did not hurt|didn't hurt|painless)\b",
        ),
        "TEETH-OMEN-GUM-BLOOD": (
            r"\b(?:my\s+|the\s+)?gums?\s+(?:(?:was|were|is|are|started|starts|began)\s+)?(?:bleeding|bled)\b",
            r"\b(?:bleeding|bloody)\s+gums?\b",
        ),
        "TEETH-MOD-BLOOD": (
            r"\bblood\s+(?:was\s+)?(?:on|covering)\s+(?:my\s+|the\s+|a\s+|one\s+)?(?:fallen\s+)?(?:tooth|teeth|molar|molars)\b",
        ),
        "TEETH-PULL-SELF": (
            r"\bi\s+(?:pulled|yanked|removed|extracted|took)\s+(?:out\s+)?(?:one\s+of\s+)?(?:my\s+)?(?:tooth|teeth|molar|molars)(?:\s+out)?\b",
        ),
        "TEETH-PULL-EXTERNAL": (
            r"\b(?:my\s+)?(?:mother|father|mom|mum|dad|sister|brother|son|daughter|child|husband|wife|spouse|friend|aunt|uncle|grandmother|grandfather|grandma|grandpa|cousin|niece|nephew|someone|somebody|stranger|dentist|doctor|man|woman|he|she|they)\s+(?:pulled|yanked|removed|extracted|took)\s+(?:out\s+)?(?:one\s+of\s+)?(?:my\s+)?(?:tooth|teeth|molar|molars)(?:\s+out)?\b",
        ),
        "TEETH-MOD-GOLD": (r"\b(?:gold|golden)\s+(?:tooth|teeth|molar|molars)\b",),
        "TEETH-MOD-REPETITION": (
            r"\b(?:again and again|over and over|kept happening|keeps happening|repeatedly|recurring dream|same dream again)\b",
        ),
        "TEETH-END-TERMINAL": (
            r"\b(?:the\s+)?same\s+tooth\s+returned\s+firmly\s+to\s+(?:the\s+)?same\s+socket\b",
            r"\b(?:the\s+)?same\s+tooth.*?(?:firm|firmly|secure|securely|tight|stayed in place)\b",
        ),
        "TEETH-END-RETURNED-SAME": (
            r"\b(?:the\s+)?same\s+tooth\s+returned\s+firmly\s+to\s+(?:the\s+)?same\s+socket\b",
            r"\b(?:the\s+)?same\s+tooth.*?(?:firm|firmly|secure|securely|tight|stayed in place)\b",
        ),
        "TEETH-POSITION-MAP": (
            r"\b(?:upper|lower|front|back|top|bottom)\s+(?:tooth|teeth|molar|molars)\b",
            r"\b(?:molar|molars)\b",
        ),
        "TEETH-COMBO-STATE-FALLOUT": (
            r"\b(?:broken|cracked|rotten|decayed)\s+(?:tooth|teeth|molar|molars)\s+(?:fell|came)\s+out\b",
        ),
        "TEETH-NONHUMAN": (
            r"\b(?:animal|bird|cat|cow|dog|goat|horse|lion|monkey|pig|snake|tiger|wolf|denture|dentures|implant|implants|prosthetic|prosthetics).*?\b(?:tooth|teeth|molar|molars)\b",
        ),
        "TEETH-REPLACEMENT-GROWTH": (
            r"\b(?:new\s+|healthy\s+|new\s+healthy\s+)?tooth\s+grew(?:\s+back)?\b",
        ),
        "TEETH-SPITTING": (
            r"\b(?:spat|spit)\s+(?:out\s+)?(?:my\s+|the\s+)?(?:tooth|teeth|molar|molars)\b",
        ),
    }
    span = _exact_span(text, patterns.get(rule_id, ()))
    if span:
        return span
    if rule_id in {"TEETH-FALLOUT-OWN", "TEETH-FALLOUT-OTHER"} and context.get("explicit_pull_removal"):
        return _exact_span(text, (r"\bmy\s+(?:tooth|teeth|molar|molars)\b",))
    return _fallback_tooth_span(text)


def _implementation_key(registry: Mapping[str, Any], rule_id: str) -> str:
    for key, rule in (registry.get("rules") or {}).items():
        if str(rule.get("rule_id") or "") == rule_id:
            return str(rule.get("implementation_key") or key)
    return ""


def _rule_record(registry: Mapping[str, Any], rule_id: str) -> Mapping[str, Any]:
    key = _implementation_key(registry, rule_id)
    return (registry.get("rules") or {}).get(key) or {}


def _warning_role(rule_id: str) -> str:
    if rule_id in BASE_WARNING_RULE_IDS:
        return "base_warning"
    if rule_id in MODIFIER_RULE_IDS:
        return "warning_modifier"
    if rule_id in STRUCTURAL_RULE_IDS:
        return "structural_terminal"
    return "unresolved_dependency"


def _reason_codes_for_registry(registry: Mapping[str, Any]) -> List[str]:
    error = str(registry.get("error") or "")
    mapping = {
        "registry_content_revision_mismatch": "REGISTRY_CONTENT_REVISION_MISMATCH",
        "registry_activation_mismatch": "REGISTRY_ACTIVATION_DRIFT",
        "registry_rule_count_mismatch": "REGISTRY_RULE_COUNT_MISMATCH",
        "registry_doctrine_version_mismatch": "REGISTRY_DECISION_MISMATCH",
        "registry_decision_id_mismatch": "REGISTRY_DECISION_MISMATCH",
    }
    for fragment, code in mapping.items():
        if fragment in error:
            return [code]
    return ["UNVERIFIED_REGISTRY_FAIL_CLOSED"]


def build_teeth_rule_provenance(
    *,
    dream: str,
    context: Mapping[str, Any],
    registry: Mapping[str, Any],
    applied_rule_ids: Sequence[str],
    unresolved_rule_ids: Sequence[str],
    active_warning: bool,
    warning_count: str,
) -> Dict[str, Any]:
    """Create auditable rule and warning provenance without adding doctrine."""
    verified = registry.get("verified") is True
    registry_identity = {
        "verified": verified,
        "contract_version": registry.get("contract_version") or REGISTRY_CONTRACT_VERSION,
        "doctrine_version": registry.get("doctrine_version") or EXPECTED_DOCTRINE_VERSION,
        "decision_id": registry.get("decision_id") or EXPECTED_DOCTRINE_VERSION,
        "sheet_revision": registry.get("sheet_revision"),
        "content_revision": registry.get("content_revision"),
        "rule_count": registry.get("rule_count", 0),
        "active_rule_ids": list(registry.get("active_rule_ids") or []),
        "unresolved_rule_ids": list(registry.get("unresolved_rule_ids") or []),
    }
    if not verified:
        return {
            "contract_version": "teeth-rule-provenance-v1",
            "complete": False,
            "registry_identity": registry_identity,
            "registry_gate": {
                "passed": False,
                "reason_codes": _reason_codes_for_registry(registry),
            },
            "rule_bindings": [],
            "candidate_dispositions": [
                {
                    "candidate_id": "jts-teeth-doctrine",
                    "rule_id": "",
                    "disposition": "withheld_registry_unverified",
                    "reason_codes": _reason_codes_for_registry(registry),
                }
            ],
            "warning_provenance": [],
            "release_status": "withheld_registry_unverified",
        }

    owner_id = _owner_id(context)
    actor_id = _actor_id(dream, context)
    entity_ids = [owner_id]
    if actor_id and actor_id not in entity_ids:
        entity_ids.append(actor_id)
    source_event_id = "teeth-event-gum-blood-1" if (
        applied_rule_ids == ["TEETH-OMEN-GUM-BLOOD"]
    ) else "teeth-event-1"

    bindings: List[Dict[str, Any]] = []
    for rule_id in [*applied_rule_ids, *unresolved_rule_ids]:
        rule = _rule_record(registry, rule_id)
        unresolved = rule_id in unresolved_rule_ids
        bindings.append(
            {
                "rule_id": rule_id,
                "registry_implementation_key": _implementation_key(registry, rule_id),
                "status": rule.get("status") or ("UNRESOLVED" if unresolved else "APPROVED"),
                "active": bool(rule.get("active") is True),
                "source_layer": SOURCE_LAYER,
                "source_event_ids": [source_event_id],
                "source_entity_ids": list(entity_ids),
                "source_spans": [_rule_span(rule_id, dream, context)],
                "gate_results": {
                    "registry": "pass",
                    "event": "pass",
                    "owner": "pass" if owner_id != "unknown_owner" else "withheld",
                    "actuality": "pass",
                    "polarity": "pass",
                    "doctrine": "withheld_unresolved" if unresolved else "pass",
                },
                "candidate_disposition": "withheld_unresolved" if unresolved else "applied",
                "warning_role": _warning_role(rule_id),
                "safety_boundary": str(rule.get("safety_boundary") or NON_GUARANTEE_POLICY),
            }
        )

    dispositions: List[Dict[str, Any]] = []
    normalized = re.sub(r"[^a-z0-9]+", " ", dream.lower()).strip()
    if "no tooth was loose" in normalized or "no teeth were loose" in normalized:
        dispositions.append(
            {
                "candidate_id": "negated-loose-state",
                "rule_id": "TEETH-STATE-LOOSE",
                "disposition": "rejected_negated",
                "reason_codes": ["NEGATED_EVENT_INELIGIBLE"],
            }
        )
    if any(value in normalized for value in ("none fell out", "did not fall out", "didn t fall out")):
        dispositions.append(
            {
                "candidate_id": "negated-tooth-loss",
                "rule_id": "TEETH-FALLOUT-OWN",
                "disposition": "rejected_negated",
                "reason_codes": ["NEGATED_EVENT_INELIGIBLE"],
            }
        )
    for binding in bindings:
        dispositions.append(
            {
                "candidate_id": f"candidate-{binding['rule_id'].lower()}",
                "rule_id": binding["rule_id"],
                "disposition": binding["candidate_disposition"],
                "reason_codes": (
                    ["UNRESOLVED_RULE_WITHHELD"]
                    if binding["candidate_disposition"] == "withheld_unresolved"
                    else []
                ),
            }
        )

    base_rule_ids = [rule_id for rule_id in applied_rule_ids if rule_id in BASE_WARNING_RULE_IDS]
    modifier_rule_ids = [rule_id for rule_id in applied_rule_ids if rule_id in MODIFIER_RULE_IDS]
    warning_provenance: List[Dict[str, Any]] = []
    if active_warning and base_rule_ids:
        warning_binding_ids = set(base_rule_ids + modifier_rule_ids)
        warning_bindings = [item for item in bindings if item["rule_id"] in warning_binding_ids]
        warning_provenance.append(
            {
                "warning_id": "teeth-warning-1",
                "source_rule_ids": base_rule_ids,
                "modifier_rule_ids": modifier_rule_ids,
                "source_event_ids": sorted({event for item in warning_bindings for event in item["source_event_ids"]}),
                "source_entity_ids": sorted({entity for item in warning_bindings for entity in item["source_entity_ids"]}),
                "source_spans": [span for item in warning_bindings for span in item["source_spans"]],
                "owner_id": owner_id,
                "warning_count": warning_count or "not_applicable",
                "doctrine_version": registry_identity["doctrine_version"],
                "registry_contract_version": registry_identity["contract_version"],
                "sheet_revision": registry_identity["sheet_revision"],
                "content_revision": registry_identity["content_revision"],
                "certainty_policy": NON_GUARANTEE_POLICY,
                "safety_boundaries": [item["safety_boundary"] for item in warning_bindings],
                "release_status": "released_non_guaranteed",
            }
        )

    if warning_provenance:
        release_status = "released_non_guaranteed"
    elif unresolved_rule_ids:
        release_status = "withheld_unresolved"
    else:
        release_status = "no_warning_released"

    provenance = {
        "contract_version": "teeth-rule-provenance-v1",
        "complete": all(
            item["source_event_ids"] and item["source_entity_ids"] and item["source_spans"]
            for item in bindings
        ),
        "registry_identity": registry_identity,
        "registry_gate": {"passed": True, "reason_codes": []},
        "rule_bindings": bindings,
        "candidate_dispositions": dispositions,
        "warning_provenance": warning_provenance,
        "release_status": release_status,
    }
    validate_teeth_rule_provenance(provenance)
    return provenance


def validate_teeth_rule_provenance(provenance: Mapping[str, Any]) -> None:
    """Reject unsafe registry/rule/warning provenance mutations."""
    identity = provenance.get("registry_identity") or {}
    bindings = list(provenance.get("rule_bindings") or [])
    warnings = list(provenance.get("warning_provenance") or [])
    gate = provenance.get("registry_gate") or {}

    if identity.get("verified") is not True or gate.get("passed") is not True:
        if bindings or warnings or provenance.get("release_status") == "released_non_guaranteed":
            raise TeethProvenanceValidationError("UNVERIFIED_REGISTRY_FAIL_CLOSED")
        return
    if identity.get("content_revision") != EXPECTED_CONTENT_REVISION:
        raise TeethProvenanceValidationError("REGISTRY_CONTENT_REVISION_MISMATCH")
    if identity.get("doctrine_version") != EXPECTED_DOCTRINE_VERSION or identity.get("decision_id") != EXPECTED_DOCTRINE_VERSION:
        raise TeethProvenanceValidationError("REGISTRY_DECISION_MISMATCH")

    active_ids = set(identity.get("active_rule_ids") or [])
    unresolved_ids = set(identity.get("unresolved_rule_ids") or [])
    if active_ids != EXPECTED_ACTIVE_RULE_IDS or unresolved_ids != EXPECTED_UNRESOLVED_RULE_IDS:
        if len(active_ids | unresolved_ids) != len(EXPECTED_RULES):
            raise TeethProvenanceValidationError("REGISTRY_RULE_COUNT_MISMATCH")
        raise TeethProvenanceValidationError("REGISTRY_ACTIVATION_DRIFT")

    known_ids = active_ids | unresolved_ids
    for binding in bindings:
        rule_id = str(binding.get("rule_id") or "")
        if rule_id not in known_ids:
            raise TeethProvenanceValidationError("UNKNOWN_RULE_ID")
        if rule_id in unresolved_ids and (
            binding.get("candidate_disposition") == "applied" or binding.get("active") is True
        ):
            raise TeethProvenanceValidationError("UNRESOLVED_RULE_APPLIED")
        if not binding.get("source_event_ids"):
            raise TeethProvenanceValidationError("RULE_EVENT_PROVENANCE_MISSING")
        spans = binding.get("source_spans") or []
        if not spans or any(not str(span).strip() for span in spans):
            raise TeethProvenanceValidationError("RULE_SPAN_PROVENANCE_MISSING")
        if binding.get("source_layer") != SOURCE_LAYER:
            raise TeethProvenanceValidationError("RULE_SOURCE_LAYER_MISMATCH")

    for warning in warnings:
        sources = set(warning.get("source_rule_ids") or [])
        modifiers = set(warning.get("modifier_rule_ids") or [])
        if not sources or sources & MODIFIER_RULE_IDS or not sources <= BASE_WARNING_RULE_IDS:
            raise TeethProvenanceValidationError("WARNING_BASE_MODIFIER_CONFUSION")
        if sources & unresolved_ids or modifiers & unresolved_ids:
            raise TeethProvenanceValidationError("UNRESOLVED_RULE_APPLIED")
        certainty = str(warning.get("certainty_policy") or "")
        if certainty != NON_GUARANTEE_POLICY:
            raise TeethProvenanceValidationError("REGISTRY_NOT_PREDICTIVE_CERTAINTY")


def mutated_provenance(provenance: Mapping[str, Any]) -> Dict[str, Any]:
    """Return an isolated copy for validation-focused regression tests."""
    return deepcopy(dict(provenance))
