# ============================================================
# Ending rules
# ============================================================

def get_ending_name(row: Dict[str, Any]) -> str:
    return get_rule_name(row, "ending_name", "ending", "name")


def get_ending_outcome_meaning(row: Dict[str, Any]) -> str:
    return row_get(
        row,
        "outcome_meaning",
        "outcome meaning",
        "meaning_modifier",
        "meaning modifier",
        "meaning",
        "effect",
        "effects",
    )


def get_ending_physical_modifier(row: Dict[str, Any]) -> str:
    return row_get(
        row,
        "physical_modifier",
        "physical modifier",
        "physical_effect",
        "physical effects",
        "effect",
        "effects",
    )


def get_ending_action_modifier(row: Dict[str, Any]) -> str:
    return row_get(
        row,
        "action_modifier",
        "action modifier",
        "action",
        "actions",
    )
