# Teeth Doctrine Coverage Audit

Purpose: protect the current Teeth implementation from regressions while the Jamaican True Stories doctrine is expanded. This document intentionally does **not** invent symbol meanings. Meanings remain owned by approved doctrine records.

## Current confirmed implementation behavior

1. **Specific Teeth subject should outrank generic Mouth**
   - When a Teeth/Tooth/Teeth Falling Out base symbol is selected, the broader `mouth` base symbol is removed.
   - This protects subject specificity and prevents container-level doctrine from competing with the actual dream subject.

2. **Specific Teeth Falling Out behavior should suppress generic Falling**
   - If the dream contains a tooth/teeth reference plus a fallout phrase, the behavior detector can canonicalize it to `teeth fell out`.
   - When the specific `Teeth Falling Out` behavior is present, generic `Falling` is suppressed unless the dream separately indicates the dreamer/body also fell.

3. **Teeth must not automatically force a death-omen seal**
   - Teeth-related doctrine can be serious, but the symbol itself must not hard-code a death-omen outcome. Seal/risk must come from the approved rule/arbitration path.

## Regression cases that should remain true

These are extraction/arbitration expectations only. They do not prescribe spiritual meanings.

| Dream wording | Expected extraction/arbitration behavior |
|---|---|
| `My teeth fell out.` | Detect Teeth + specific Teeth Falling Out behavior; do not also emit generic Mouth or generic Falling. |
| `One tooth came out in my hand.` | Detect Tooth/Teeth family + specific Teeth Falling Out behavior if doctrine keywords permit; do not add generic Mouth. |
| `My teeth were falling out while I was falling down stairs.` | Keep Teeth Falling Out **and** generic/body Falling because the dream contains a separate physical fall event. |
| `I looked at my teeth in my mouth.` | Prefer Teeth as the subject. Mouth may be treated as container/context only if a future explicit event-binding rule needs it, not as a competing primary subject. |
| `I dreamed my mouth was locked but my teeth were normal.` | Do not suppress Mouth merely because the word `teeth` appears. Suppression should only occur when a Teeth-family base symbol is actually selected as a meaningful subject. |
| `My teeth did not fall out.` | Negation must not be interpreted as an affirmative Teeth Falling Out event. Add explicit negation protection before production coverage is considered complete. |
| `I thought my teeth would fall out, but they stayed in.` | Hypothetical/anticipated fallout should not be promoted to an occurred Teeth Falling Out event. |
| `My sister's teeth fell out.` | Bind the Teeth event to the sister/relationship subject, not automatically to the dreamer. |
| `My tooth was loose but did not come out.` | Treat `loose` as a state/condition candidate if approved in doctrine; do not convert it to Teeth Falling Out. |
| `My tooth broke.` | Treat `broken` as a state/condition candidate attached to Tooth/Teeth; do not infer Teeth Falling Out unless an actual fallout action is present. |
| `My gums were bleeding around my teeth.` | Keep Bleeding/Gums/Teeth event binding distinct; avoid promoting `bleeding` into an unrelated death or injury doctrine without an approved rule. |

## Highest-priority missing protection

### A. Negation-aware Teeth fallout detection
Current contextual detection checks for Teeth/Tooth plus phrases such as `fell out`, `falling out`, `came out`, and `coming out`. Before Teeth is considered stable, it should explicitly reject negated clauses such as:

- `did not fall out`
- `didn't fall out`
- `never fell out`
- `would not come out`
- `stayed in`

This should be implemented in the deterministic extractor, not delegated to narration.

### B. Event binding for third-party Teeth dreams
The interpreter should distinguish:

- `my teeth fell out`
- `my sister's teeth fell out`
- `a stranger's teeth fell out`

The relationship/person owner should attach to the Teeth event before doctrine synthesis.

### C. Teeth state taxonomy needs doctrine approval
Potential high-frequency state/action variants to review against the existing Jamaican True Stories doctrine before adding meanings:

- loose tooth / loose teeth
- broken or cracked tooth
- chipped tooth
- rotten/decayed tooth
- missing tooth
- pulling/extracting a tooth
- dentist removing a tooth
- tooth pain
- bleeding tooth/gums
- new tooth growing
- artificial/fake/denture teeth
- white/clean teeth
- dirty/yellow teeth

These are **coverage candidates only**, not approved meanings.

## Definition of Teeth cluster “substantially complete”

The Teeth cluster should not be marked complete until all of the following are true:

1. Approved canonical Teeth-family doctrine records are identified and deduplicated.
2. Aliases cover natural singular/plural phrasing and common voice-to-text wording.
3. Specific actions and states are represented only where doctrine has approved them.
4. Negation and hypothetical language do not trigger affirmative events.
5. Teeth-specific events outrank generic container/action collisions such as Mouth and generic Falling.
6. Relationship ownership is bound correctly for the dreamer vs. another person.
7. Ending rules can modify the event without changing the underlying doctrine arbitrarily.
8. Seal/risk remains rule-driven and is not hard-coded from the word `teeth` alone.
9. A regression/golden test set covers the cases above.
10. Narration is validated so it cannot introduce unsupported Teeth meanings or warnings.

## Next implementation action

Add deterministic negation protection for the contextual `Teeth Falling Out` behavior, then add executable regression tests for the positive, separate-body-fall, and negated cases. After that, inspect the doctrine data source to determine which Teeth state/action candidates are already approved before creating any new symbol meanings.
