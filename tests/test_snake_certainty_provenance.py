import copy
import hashlib
import json
import unittest

from app.snake_event_graph import extract_snake_event_graph, validate_snake_event_graph


CERTAINTY_ORACLE_DIGESTS = {
    "A snake watched me from the doorway.": "d0082ceb128f45b11d4729d2cb8fb1d6176ead899ec4a87f35fb754931779ae7",
    "Two snakes appeared. It attacked my sister.": "45c310f41a32ed55da2734a2ee74cd5f36e3e1690f6ce7bd29abea2a070e48ec",
    "The snake attacked me and tried to bite my hand, but it never touched me.": "db96db83432e453b4c74c321f17f4d8bb11ae997d892cc4f1b902900df60926b",
    "The snake bit my hand and the fight ended there.": "af13a520b102e5dd67d5c4650a00336257df4b90ba436f4aeab8b31cf1a6c09f",
    "The cobra bit my brother's arm and venom moved through his arm.": "624539710142de20085d65f9432aeef362576b76d6b45f7558db1d0f9b8e554a",
    "One snake watched in my kitchen. Later another snake attacked me at work.": "2e9d9df6f855f1d133f805e012149b5444474151b3054614eb311a8066544930",
    "The snake changed into my friend and stood beside me.": "2522fe139ad5d5efbfe9c155b5739b1e40d84eaca3909e94a9e995849462ab74",
    "A small snake watched me while a huge cobra attacked my brother.": "24abcd603cb66c3688099c9ef80db0ff0e50fbbc882d2cbbef6c6c2f720232ce",
    "Three snakes surrounded me. I killed the first, the second ran away, and the third kept watching when the dream ended.": "d252a4713581a02b4b3f99e8dc917a2dabf0509af420e48432c96016f6ec86a1",
    "Two snakes came close. The first did not bite me, but the second bit my sister.": "477c353b106806a3bcb96f360ba1fb42dca9bcaf261bd73440d89721cd22cc64",
    "Again I fought the same snake, but I woke before either of us won.": "822551a673f140624e609b6742532c97ae2fa96adae0b7678678b9de4e7745eb",
    "A snake attacked me and I woke before the fight ended. After waking I rejected the bad dream and planned to read Psalm 91 before bed.": "44027379fb9901ac371d6c6ce52d0364f9f32bd5116355349cbe0f963572a52b",
    "The snake bit my hand, but I kept fighting and killed that same snake at the end.": "0108cebca12fc7e0580d7b96141be27850ddea59b45cedd4a2fb31bce2afc2f8",
}


class SnakeCertaintyProvenanceTests(unittest.TestCase):
    def graph(self, dream):
        return extract_snake_event_graph(dream)

    def test_all_context_certainty_oracles_match_exactly(self):
        for dream, expected_digest in CERTAINTY_ORACLE_DIGESTS.items():
            graph = self.graph(dream)
            encoded = json.dumps(
                graph["certainty_axis_records"], sort_keys=True,
                separators=(",", ":"),
            ).encode()
            with self.subTest(dream=dream):
                self.assertEqual("snake-certainty-provenance-v1", graph["certainty_contract_version"])
                self.assertEqual(expected_digest, hashlib.sha256(encoded).hexdigest())
                self.assertEqual({"verified": True, "reason_codes": []}, graph["graph_integrity"])

    def assert_rejected(self, dream, mutate, reason):
        graph = copy.deepcopy(self.graph(dream))
        mutate(graph)
        self.assertIn(reason, validate_snake_event_graph(graph)["reason_codes"])

    def test_all_unsafe_certainty_mutations_fail_closed(self):
        watch = "A snake watched me from the doorway."
        ambiguous = "Two snakes appeared. It attacked my sister."
        attempt = "The snake attacked me and tried to bite my hand, but it never touched me."
        venom = "The cobra bit my brother's arm and venom moved through his arm."
        location = "One snake watched in my kitchen. Later another snake attacked me at work."
        transform = "The snake changed into my friend and stood beside me."
        size = "A small snake watched me while a huge cobra attacked my brother."
        mixed = "Three snakes surrounded me. I killed the first, the second ran away, and the third kept watching when the dream ended."
        negated = "Two snakes came close. The first did not bite me, but the second bit my sister."
        recurrence = "Again I fought the same snake, but I woke before either of us won."
        faith = "A snake attacked me and I woke before the fight ended. After waking I rejected the bad dream and planned to read Psalm 91 before bed."

        self.assert_rejected(watch, lambda g: g["certainty_axis_records"][0].update(real_world_certainty="established"), "DOCTRINE_MATCH_NOT_REAL_WORLD_FACT")
        self.assert_rejected(ambiguous, lambda g: g["certainty_axis_records"][0].update(confidence_caps=["doctrine_match_only"]), "AMBIGUOUS_BINDING_CONFIDENCE_CAP")
        self.assert_rejected(attempt, lambda g: g["certainty_axis_records"][0].update(event_completion="completed"), "COMPLETION_AXIS_PROMOTION")
        self.assert_rejected(watch, lambda g: g["certainty_axis_records"][0].update(predictive_certainty="guaranteed"), "TERMINAL_NOT_PREDICTION")
        self.assert_rejected(venom, lambda g: g["certainty_axis_records"][0].update(medical_certainty="established"), "VENOM_NOT_MEDICAL_EVIDENCE")
        self.assert_rejected(location, lambda g: g["certainty_axis_records"][0].update(safety_qualifiers=[]), "LOCATION_NOT_CULPRIT")
        self.assert_rejected(transform, lambda g: g["certainty_axis_records"][0].update(safety_qualifiers=[]), "TRANSFORMED_PERSON_NOT_DEFINITIVE_ENEMY")
        self.assert_rejected(size, lambda g: g["certainty_axis_records"][0].update(safety_qualifiers=[]), "SIZE_NOT_OBJECTIVE_DANGER_PROOF")
        self.assert_rejected(mixed, lambda g: g["certainty_axis_records"][0].update(chain_ids=["chain-snake-2"]), "CERTAINTY_CHAIN_CARDINALITY_LOSS")
        self.assert_rejected(negated, lambda g: g["certainty_axis_records"][0].update(narration_eligibility="eligible_interpretive_only"), "NEGATED_EVENT_NOT_RELEASED")
        self.assert_rejected(recurrence, lambda g: g["certainty_axis_records"][0].update(safety_qualifiers=[]), "RECURRENCE_NOT_GUARANTEED")
        self.assert_rejected(faith, lambda g: g["certainty_axis_records"][0].update(safety_qualifiers=[]), "FAITH_PRACTICE_NOT_GUARANTEED")


if __name__ == "__main__":
    unittest.main()
