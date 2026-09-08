from __future__ import annotations

from flask import Blueprint, jsonify

from app.qa_access import (
    get_qa_grant_status,
    public_qa_access_metadata,
    qa_token_from_request,
)
from app.release_info import release_metadata
from app.teeth_doctrine import (
    build_teeth_doctrine_context,
    build_teeth_narration_facts,
)
from app.teeth_registry import get_teeth_registry_snapshot, public_registry_metadata
from app.snake_doctrine import build_snake_doctrine_context, build_snake_narration_facts
from app.snake_registry import get_snake_registry_snapshot, public_snake_registry_metadata


qa_bp = Blueprint("qa", __name__)

TEETH_QA_CONTRACT_VERSION = "teeth-qa-contract-v2"
SNAKE_QA_CONTRACT_VERSION = "snake-qa-contract-v5"


@qa_bp.post("/qa/interpret")
def qa_interpret():
    """Run the real interpreter only when an isolated QA token is active."""
    token = qa_token_from_request()
    grant = get_qa_grant_status(token)
    if grant.get("active") is not True:
        response = jsonify(
            {
                "blocked": True,
                "reason": grant.get("reason") or "missing_token",
                "message": "An active QA access token is required.",
                "access": "blocked",
            }
        )
        response.status_code = 403
        response.headers["Cache-Control"] = "no-store"
        return response
    from app.services.interpreter_service import run_interpretation

    return run_interpretation()


@qa_bp.get("/qa/status")
def qa_status():
    registry = get_teeth_registry_snapshot()
    snake_registry = get_snake_registry_snapshot()
    access = public_qa_access_metadata()
    response = jsonify(
        {
            "service": "dream-interpreter",
            "release": release_metadata(),
            "qa_access": access,
            "doctrine_registry": public_registry_metadata(registry),
            "snake_doctrine_registry": public_snake_registry_metadata(snake_registry),
            "ready": bool(
                registry.get("verified") is True
                and snake_registry.get("verified") is True
                and access.get("configured") is True
                and access.get("storage_ready") is True
            ),
        }
    )
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Robots-Tag"] = "noindex, nofollow"
    return response

# This surface is intentionally bounded to synthetic, founder-approved
# regression inputs. It cannot interpret arbitrary dreams and never touches
# access counters, paid entitlements, AI narration, or customer data.
BASE_TEETH_QA_CASES = (
    ("quantity_one", "My tooth fell out."),
    ("quantity_multiple", "Three of my teeth fell out."),
    ("ownership_other", "My sister's tooth fell out."),
    ("ownership_external_actor", "My sister pulled my tooth out."),
    ("painful_loss", "One of my teeth fell out and it hurt badly."),
    ("painless_loss", "One of my teeth fell out without pain."),
    (
        "blood_after_loss",
        "My tooth fell out and there was blood on the fallen tooth.",
    ),
    (
        "bleeding_gums_with_negations",
        "My gums were bleeding, but no tooth was loose and none fell out.",
    ),
    ("loose_without_loss", "My tooth was loose but did not fall out."),
    ("negated_loss", "My teeth did not fall out."),
    ("hypothetical_loss", "I thought my teeth might fall out."),
    (
        "genuine_terminal_ending",
        "My tooth fell out, but in the end the same tooth fitted firmly "
        "back into the same socket.",
    ),
    ("attempted_ending", "My tooth fell out and I tried to put it back."),
)

ATTEMPT_BINDING_QA_CASES = (
    ("CTX-001-ATTEMPT-BIND-DREAMER-001", "My tooth fell out and I tried to put it back."),
    ("CTX-001-ATTEMPT-BIND-NEGATED-001", "My tooth fell out, but I never tried to put it back."),
    ("CTX-001-ATTEMPT-BIND-HYPOTHETICAL-001", "If my tooth fell out, I would try to put it back."),
    ("CTX-001-ATTEMPT-BIND-QUOTED-001", 'My tooth fell out. My aunt said, "I tried to put it back."'),
    ("CTX-001-ATTEMPT-BIND-WAKING-001", "My tooth fell out. After I woke up, I tried to put it back in my imagination."),
    ("CTX-001-ATTEMPT-BIND-OTHER-OWNER-001", "My sister's tooth fell out and she tried to put it back."),
    ("CTX-001-ATTEMPT-BIND-EXTERNAL-ACTOR-001", "My tooth fell out and my sister tried to put it back."),
    ("CTX-001-ATTEMPT-BIND-MULTI-OWNER-001", "My tooth fell out and I tried to put it back. My sister's tooth fell out and she left it there."),
    ("CTX-001-ATTEMPT-BIND-AMBIGUOUS-TARGET-001", "My tooth and my sister's tooth fell out. I tried to put it back."),
    ("CTX-001-ATTEMPT-BIND-THEN-FIRM-001", "My tooth fell out. I tried to put it back, and then the same tooth fitted firmly back into the same socket."),
    ("CTX-001-ATTEMPT-BIND-THEN-SECOND-LOSS-001", "My left tooth fell out and I tried to put it back. Then another tooth fell out."),
    ("CTX-001-ATTEMPT-BIND-REPORTED-001", "My tooth fell out. My sister told me that she tried to put her tooth back yesterday."),
)

CONDITION_PROVENANCE_QA_CASES = (
    ("CTX-003-COND-PROV-GUMS-NEGATED-001", "My gums were bleeding, but no tooth was loose and none fell out."),
    ("CTX-003-COND-PROV-GUMS-ONLY-001", "My gums were bleeding."),
    ("CTX-003-COND-PROV-GUMS-RETAINED-001", "My gums were bleeding and every tooth stayed firm."),
    ("CTX-003-COND-PROV-LOOSE-NEGATED-LOSS-001", "My tooth was loose but did not fall out."),
    ("CTX-003-COND-PROV-WOBBLY-RETAINED-001", "My tooth was wobbly and stayed in my mouth."),
    ("CTX-003-COND-PROV-TWO-LOOSE-001", "Two of my teeth were loose, but neither fell out."),
    ("CTX-003-COND-PROV-OTHER-OWNER-001", "My sister's tooth was loose but did not fall out."),
    ("CTX-003-COND-PROV-NEGATED-LOOSE-THEN-LOSS-001", "My tooth was not loose, but it fell out."),
    ("CTX-003-COND-PROV-LOOSE-THEN-LOSS-001", "My tooth became loose, then it fell out."),
    ("CTX-003-COND-PROV-QUOTED-001", 'My aunt said, "My tooth is loose." My own tooth stayed firm.'),
    ("CTX-003-COND-PROV-HYPOTHETICAL-001", "If my tooth were loose, I would visit a dentist."),
    ("CTX-003-COND-PROV-MULTI-OWNER-001", "My gums were bleeding. My sister's tooth was loose but did not fall out."),
)

WARNING_CLAIM_PARTITION_QA_CASES = (
    ("CTX-003-CLAIM-PART-MULTI-OWNER-CONDITIONS-001", "My gums were bleeding. My sister's tooth was loose but did not fall out."),
    ("CTX-003-CLAIM-PART-SAME-OWNER-DISTINCT-CONDITIONS-001", "My gums were bleeding and my tooth was loose but did not fall out."),
    ("CTX-003-CLAIM-PART-TWO-OWNER-LOOSE-001", "My tooth was loose. My sister's tooth was loose."),
    ("CTX-003-CLAIM-PART-TWO-OTHER-OWNERS-001", "My sister's tooth was loose. My brother's tooth was loose."),
    ("CTX-003-CLAIM-PART-TWO-LOOSE-ONE-EVENT-001", "Two of my teeth were loose, but neither fell out."),
    ("CTX-003-CLAIM-PART-ONE-LOSS-MULTI-RULE-001", "My tooth fell out."),
    ("CTX-003-CLAIM-PART-LOSS-PLUS-OTHER-LOOSE-001", "My tooth fell out. My sister's tooth was loose but did not fall out."),
    ("CTX-003-CLAIM-PART-GUMS-PLUS-OTHER-LOSS-001", "My gums were bleeding. My sister's tooth fell out."),
    ("CTX-003-CLAIM-PART-LOOSE-THEN-LOSS-001", "My tooth became loose, then it fell out."),
    ("CTX-003-CLAIM-PART-QUOTED-PLUS-GUMS-001", 'My aunt said, "My tooth is loose." My gums were bleeding.'),
    ("CTX-003-CLAIM-PART-HYPOTHETICAL-PLUS-OTHER-001", "If my tooth were loose, I would worry. My sister's tooth was loose."),
    ("CTX-003-CLAIM-PART-NEGATED-PLUS-OTHER-001", "My tooth was not loose. My sister's tooth was loose."),
)

NARRATION_CLAIM_CONSUMPTION_QA_CASES = (
    ("CTX-003-NARR-CONS-MULTI-OWNER-CONDITIONS-001", "My gums were bleeding. My sister's tooth was loose but did not fall out."),
    ("CTX-003-NARR-CONS-SAME-OWNER-DISTINCT-CONDITIONS-001", "My gums were bleeding and my tooth was loose but did not fall out."),
    ("CTX-003-NARR-CONS-TWO-OWNER-LOOSE-001", "My tooth was loose. My sister's tooth was loose."),
    ("CTX-003-NARR-CONS-LOSS-PLUS-OTHER-LOOSE-001", "My tooth fell out. My sister's tooth was loose but did not fall out."),
    ("CTX-003-NARR-CONS-GUMS-PLUS-OTHER-LOSS-001", "My gums were bleeding. My sister's tooth fell out."),
    ("CTX-003-NARR-CONS-QUOTED-PLUS-GUMS-001", 'My aunt said, "My tooth is loose." My gums were bleeding.'),
    ("CTX-003-NARR-CONS-HYPOTHETICAL-PLUS-OTHER-001", "If my tooth were loose, I would worry. My sister's tooth was loose."),
    ("CTX-003-NARR-CONS-NEGATED-PLUS-OTHER-001", "My tooth was not loose. My sister's tooth was loose."),
    ("CTX-003-NARR-CONS-LOOSE-THEN-LOSS-001", "My tooth became loose, then it fell out."),
    ("CTX-003-NARR-CONS-ATTEMPT-THEN-SECOND-LOSS-001", "My tooth fell out and I tried to put it back. Then another tooth fell out."),
    ("CTX-003-NARR-CONS-TERMINAL-RETURN-001", "My tooth fell out, then the same tooth fitted firmly back into the same socket."),
    ("CTX-003-NARR-CONS-ONE-LOSS-MULTI-RULE-001", "My tooth fell out."),
)

TEETH_QA_CASES = (
    BASE_TEETH_QA_CASES
    + ATTEMPT_BINDING_QA_CASES
    + CONDITION_PROVENANCE_QA_CASES
    + WARNING_CLAIM_PARTITION_QA_CASES
    + NARRATION_CLAIM_CONSUMPTION_QA_CASES
)


@qa_bp.get("/qa/teeth-regression")
def teeth_regression_contract():
    registry = get_teeth_registry_snapshot()
    cases = []
    for case_id, dream in TEETH_QA_CASES:
        cases.append(
            {
                "case_id": case_id,
                "dream": dream,
                "doctrine": build_teeth_doctrine_context(dream),
                "narration": build_teeth_narration_facts(dream),
            }
        )

    response = jsonify(
        {
            "contract_version": TEETH_QA_CONTRACT_VERSION,
            "release": release_metadata(),
            "doctrine_registry": public_registry_metadata(
                registry,
                include_rule_ids=True,
            ),
            "case_count": len(cases),
            "cases": cases,
        }
    )
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Robots-Tag"] = "noindex, nofollow"
    return response


SNAKE_QA_CASES = (
    ("base_enemy", "I saw a snake in the dream."),
    ("presence_negated", "I did not see any snake."),
    ("watching", "A snake watched me from the grass."),
    ("attack_not_outcome", "A snake attacked me, but the dream ended before the fight was over."),
    ("retreat", "The snake ran away from me."),
    ("victory", "I fought the snake and killed it."),
    ("found_dead_not_victory", "I found a dead snake beside the road."),
    ("defeat", "The snake defeated me at the end."),
    ("multiple", "Three snakes surrounded me."),
    ("small", "A tiny snake crossed the path."),
    ("large_cobra", "A huge cobra chased me."),
    ("bite", "The snake bit me on the hand."),
    ("attempted_bite", "The snake tried to bite me but did not."),
    ("venom", "The snake bit me and venom entered my arm."),
    ("home", "A snake was inside my house."),
    ("work", "A snake appeared in my office at work."),
    ("transformation", "The snake transformed into a person."),
    ("ownership", "My neighbor owned the snake."),
    ("unfinished", "I was fighting the snake when I woke up."),
    ("color_excluded", "A red snake watched me."),
    ("SNAKE-002-EVENT-WATCH-001", "A snake watched me from the doorway."),
    ("SNAKE-002-EVENT-ATTEMPT-BITE-001", "The snake attacked me and tried to bite my hand, but it never touched me."),
    ("SNAKE-002-EVENT-BITE-DREAMER-001", "The snake bit my ankle and I fell, unable to continue the fight."),
    ("SNAKE-002-TARGET-THIRD-PARTY-001", "The snake ignored me and bit my sister on her wrist."),
    ("SNAKE-003-VENOM-SCOPE-001", "The cobra bit my arm, and I saw venom enter and move through my arm."),
    ("SNAKE-002-PROTECTION-BLOCK-001", "A snake struck at me, but a shield blocked it before it touched me."),
    ("SNAKE-002-MULTI-MIXED-001", "Three snakes surrounded me: a small snake watched, a huge cobra attacked and I killed it, while the third snake ran away."),
    ("SNAKE-002-NEGATION-001", "The snake did not bite me; it only watched me."),
    ("SNAKE-002-HYPOTHETICAL-001", "I thought, if the snake bites me I will lose, but the snake actually ran away."),
    ("SNAKE-002-TARGET-AMBIGUOUS-001", "The snake rushed between my sister and my cousin, then it bit her."),
    ("SNAKE-003-TRANSFORM-SAFETY-001", "The snake changed into my friend and stood beside me."),
    ("SNAKE-002-LOCATION-MULTI-SPHERE-001", "One snake watched me in my kitchen. Later another snake attacked me at work."),
    ("SNAKE-002-COLOR-INVARIANT-BLACK-001", "A black snake attacked me and then ran away."),
    ("SNAKE-002-COLOR-INVARIANT-GREEN-001", "A green snake attacked me and then ran away."),
    ("SNAKE-002-ENDING-ALREADY-DEAD-001", "I discovered a snake already dead beside the road."),
    ("SNAKE-002-ENDING-ESCAPE-001", "The snake chased me through the yard, but I escaped and woke up."),
    ("SNAKE-002-RECURRENCE-UNFINISHED-001", "Again I fought the same snake, but I woke before either of us won."),
    ("SNAKE-002-OWNERSHIP-LOW-001", "My neighbor said the snake was his, but the snake only watched me."),
    ("SNAKE-003-FAITH-SEPARATION-001", "A snake attacked me and I woke before the fight ended. After waking I rejected the bad dream and planned to read Psalm 91 before bed."),
    ("SNAKE-002-TARGET-LINEAGE-SELF-HAND-001", "The snake bit my hand and the fight ended there."),
    ("SNAKE-002-TARGET-LINEAGE-THIRD-PARTY-WRIST-001", "The snake passed me and bit my sister's wrist."),
    ("SNAKE-002-TARGET-LINEAGE-COREFERENCE-IT-001", "My sister held out her hand. The snake bit it."),
    ("SNAKE-002-TARGET-LINEAGE-AMBIGUOUS-001", "My sister and cousin held out their hands. The snake bit one of them."),
    ("SNAKE-002-TARGET-LINEAGE-MULTI-PERSON-001", "One snake bit my hand while a second snake bit my sister's wrist."),
    ("SNAKE-002-TARGET-LINEAGE-SEQUENCE-001", "The snake attacked me, then turned and bit my sister's hand."),
    ("SNAKE-003-TARGET-LINEAGE-VENOM-THIRD-PARTY-001", "The cobra bit my brother's arm and venom moved through his arm."),
    ("SNAKE-002-TARGET-LINEAGE-ATTEMPT-001", "The snake tried to bite my ankle but never touched me."),
    ("SNAKE-002-TARGET-LINEAGE-PROTECTION-001", "The snake tried to bite my child, but I blocked it before contact."),
    ("SNAKE-002-TARGET-LINEAGE-NONPERSON-001", "The snake bit my travel bag and then disappeared."),
    ("SNAKE-002-TARGET-LINEAGE-BITE-THEN-VICTORY-001", "The snake bit my hand, but I kept fighting and killed that same snake at the end."),
    ("SNAKE-002-TARGET-LINEAGE-NEGATED-CORRECTION-001", "The snake did not bite my sister; it bit my hand instead."),
    ("REG-SNAKE-ATTACK-001", "A snake attacked me, but the dream ended before either of us won."),
    ("REG-SNAKE-BITE-ATTEMPT-001", "The snake lunged to bite me but missed."),
    ("REG-SNAKE-BITE-DREAMER-001", "The snake bit my hand."),
    ("REG-SNAKE-CHASE-ESCAPE-001", "A snake chased me, but I escaped and locked the door."),
    ("REG-SNAKE-CHASE-CAPTURE-001", "A snake chased me and wrapped around me, but I woke before it bit me."),
    ("REG-SNAKE-DEFEAT-001", "The snake knocked me down and stood over me when the dream ended."),
    ("REG-SNAKE-HYPOTHETICAL-001", "I wondered what would happen if the snake bit me."),
    ("REG-SNAKE-MULTI-ACTION-001", "Two snakes appeared: one watched me while the other attacked."),
    ("REG-SNAKE-MIXED-ENDINGS-001", "Three snakes came: I killed one, another bit me, and the third ran away."),
    ("REG-SNAKE-NEGATION-001", "The snake did not bite or attack me."),
    ("REG-SNAKE-PROTECT-OTHER-001", "A snake attacked a child, and I killed it before it reached her."),
    ("REG-SNAKE-SIZE-SPECIES-001", "A small garden snake and a huge cobra blocked my path."),
    ("REG-SNAKE-TRANSFORM-001", "The snake turned into my sister."),
    ("REG-SNAKE-TRANSFORM-ACCUSATION-001", "The snake became my coworker, so that proves he is my enemy."),
    ("REG-SNAKE-VENOM-ABSENT-001", "The snake bit me, but the dream never showed or mentioned venom."),
    ("REG-SNAKE-QUOTED-001", "My aunt said, 'A snake bit me,' but I saw no snake."),
    ("SNAKE-002-PARTITION-TWO-DISTINCT-ACTIONS-001", "Two snakes came at me. The first watched from the doorway, while the second attacked my sister."),
    ("SNAKE-002-PARTITION-THREE-MIXED-ENDINGS-001", "Three snakes surrounded me. I killed the first, the second ran away, and the third kept watching when the dream ended."),
    ("SNAKE-002-PARTITION-SAME-SNAKE-SEQUENCE-001", "A snake watched me, then chased me, and finally I killed that same snake."),
    ("SNAKE-002-PARTITION-GROUP-SHARED-ACTION-001", "Three snakes watched me from the fence."),
    ("SNAKE-002-PARTITION-MODIFIER-SCOPE-001", "A small snake watched me while a huge cobra attacked my brother."),
    ("SNAKE-002-PARTITION-LOCATION-SCOPE-001", "One snake watched in my kitchen. Later another snake attacked me at work."),
    ("SNAKE-002-PARTITION-TARGET-SCOPE-001", "One snake bit my sister while another chased me."),
    ("SNAKE-002-PARTITION-PRONOUN-RESOLVED-001", "Two snakes appeared. The first watched me. It then ran away."),
    ("SNAKE-002-PARTITION-PRONOUN-AMBIGUOUS-001", "Two snakes appeared. It attacked my sister."),
    ("SNAKE-002-PARTITION-NEGATED-MEMBER-001", "Two snakes came close. The first did not bite me, but the second bit my sister."),
    ("SNAKE-002-PARTITION-HYPOTHETICAL-MEMBER-001", "Two snakes appeared. If the first bit me I would run, but the second only watched."),
    ("SNAKE-002-PARTITION-RECURRENCE-SAME-ENTITY-001", "The same snake returned from my earlier unfinished dream, watched me, and the fight was still unfinished."),
    ("SNAKE-002-ARBITRATION-ATTACK-UNRESOLVED-001", "A snake attacked me, but the dream ended before either of us won."),
    ("SNAKE-002-ARBITRATION-BITE-THEN-VICTORY-001", "The snake bit my hand, but I kept fighting and killed that same snake at the end."),
    ("SNAKE-002-ARBITRATION-ATTEMPT-THEN-ESCAPE-001", "The snake lunged to bite me but missed. I escaped and locked the door before I woke."),
    ("SNAKE-002-ARBITRATION-CHASE-CAPTURE-WAKE-001", "A snake chased me and wrapped around me, but I woke before it bit me."),
    ("SNAKE-002-ARBITRATION-DEFEAT-KNOCKDOWN-001", "The snake knocked me down and stood over me when the dream ended."),
    ("SNAKE-002-ARBITRATION-RETREAT-VS-DISAPPEAR-001", "Two snakes appeared. The first ran away, while the second simply disappeared."),
    ("SNAKE-002-ARBITRATION-KILL-VS-FOUND-DEAD-001", "I killed one snake, but I only discovered the second snake already dead."),
    ("SNAKE-002-ARBITRATION-INTERMEDIATE-REVERSED-001", "I knocked the snake back and thought I had won, but it rose and bit me when the dream ended."),
    ("SNAKE-002-ARBITRATION-SCENE-BREAK-001", "At home a snake attacked me. In a later scene at work, another snake ran away as the dream ended."),
    ("SNAKE-002-ARBITRATION-QUOTED-ENDING-001", 'My sister said, "The snake killed me," but I saw the snake only watching her when the dream ended.'),
    ("SNAKE-002-ARBITRATION-HYPOTHETICAL-ENDING-001", "If the snake killed me I would lose, but it actually ran away at the end."),
    ("SNAKE-002-ARBITRATION-AMBIGUOUS-KILL-001", "Two snakes stood before me. I killed it at the end."),
    ("EVID-REG-SNAKE-027", "A snake watched my sister from across the room."),
    ("EVID-REG-SNAKE-028", "The snake tried to bite my cousin but never touched her."),
    ("EVID-REG-SNAKE-029", "No snake chased, bit, or attacked me. I only saw a carving of one."),
    ("EVID-REG-SNAKE-030", "A snake was in my bedroom, therefore my partner cursed me."),
    ("EVID-REG-SNAKE-031", "A snake moved through my house but never approached anyone."),
    ("EVID-REG-SNAKE-032", "A snake was hiding in my kitchen."),
    ("EVID-REG-SNAKE-033", "A snake was in my bathroom and did not attack."),
)


@qa_bp.get("/qa/snake-regression")
def snake_regression_contract():
    registry = get_snake_registry_snapshot()
    registry_verified = registry.get("verified") is True
    cases = [
        {
            "case_id": case_id,
            "dream": dream,
            "doctrine": build_snake_doctrine_context(dream),
            "narration": build_snake_narration_facts(dream),
        }
        for case_id, dream in SNAKE_QA_CASES
    ]
    response = jsonify(
        {
            "contract_version": SNAKE_QA_CONTRACT_VERSION,
            "contract_pass": registry_verified,
            "failure_reason": "" if registry_verified else (
                registry.get("error") or "snake_registry_verification_failed"
            ),
            "release": release_metadata(),
            "doctrine_registry": public_snake_registry_metadata(
                registry,
                include_rule_ids=True,
            ),
            "case_count": len(cases),
            "cases": cases,
            "non_billable": True,
            "customer_credits_consumed": False,
        }
    )
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Robots-Tag"] = "noindex, nofollow"
    return response, 200 if registry_verified else 503
