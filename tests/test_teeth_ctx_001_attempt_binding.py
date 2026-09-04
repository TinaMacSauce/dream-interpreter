import unittest

from app.teeth_doctrine import (
    build_teeth_doctrine_context,
    build_teeth_narration_facts,
)


CASES = {
    "dreamer": "My tooth fell out and I tried to put it back.",
    "negated": "My tooth fell out, but I never tried to put it back.",
    "hypothetical": "If my tooth fell out, I would try to put it back.",
    "quoted": 'My tooth fell out. My aunt said, "I tried to put it back."',
    "waking": (
        "My tooth fell out. After I woke up, I tried to put it back "
        "in my imagination."
    ),
    "other_owner": "My sister's tooth fell out and she tried to put it back.",
    "external_actor": "My tooth fell out and my sister tried to put it back.",
    "multi_owner": (
        "My tooth fell out and I tried to put it back. "
        "My sister's tooth fell out and she left it there."
    ),
    "ambiguous": (
        "My tooth and my sister's tooth fell out. I tried to put it back."
    ),
    "then_firm": (
        "My tooth fell out. I tried to put it back, and then the same tooth "
        "fitted firmly back into the same socket."
    ),
    "second_loss": (
        "My left tooth fell out and I tried to put it back. "
        "Then another tooth fell out."
    ),
    "reported": (
        "My tooth fell out. My sister told me that she tried to put her tooth "
        "back yesterday."
    ),
}


class TeethAttemptBindingContractTests(unittest.TestCase):
    def doctrine(self, name):
        return build_teeth_doctrine_context(CASES[name])

    def record(self, name):
        records = self.doctrine(name)["restoration_attempt_records"]
        self.assertEqual(1, len(records), name)
        return records[0]

    def test_every_explicit_attempt_has_complete_typed_provenance(self):
        required = {
            "attempt_id",
            "action",
            "actor_id_or_ambiguous",
            "target_tooth_ids_or_ambiguous",
            "owner_id_or_ambiguous",
            "event_chain_id_or_null",
            "scene_id",
            "phase",
            "channel",
            "polarity",
            "modality",
            "actuality",
            "completion",
            "source_span",
            "binding_confidence",
            "narration_eligibility",
            "ineligibility_reasons",
        }
        for name, dream in CASES.items():
            with self.subTest(name=name):
                doctrine = self.doctrine(name)
                self.assertEqual(
                    "restoration-attempt-binding/1.0",
                    doctrine["restoration_attempt_contract_version"],
                )
                record = doctrine["restoration_attempt_records"][0]
                self.assertEqual(required, set(record))
                self.assertEqual("manual_reinsertion_attempt", record["action"])
                self.assertEqual("attempted_only", record["completion"])
                span = record["source_span"]
                self.assertEqual(span["text"], dream[span["start"]:span["end"]])

    def test_actual_attempts_bind_actor_owner_and_target_independently(self):
        expected = {
            "dreamer": ("dreamer", "dreamer", ["tooth-1"]),
            "other_owner": ("sister", "sister", ["sister-tooth-1"]),
            "external_actor": ("sister", "dreamer", ["tooth-1"]),
            "multi_owner": ("dreamer", "dreamer", ["dreamer-tooth-1"]),
            "second_loss": ("dreamer", "dreamer", ["left-tooth-1"]),
        }
        for name, values in expected.items():
            with self.subTest(name=name):
                record = self.record(name)
                self.assertEqual(values[0], record["actor_id_or_ambiguous"])
                self.assertEqual(values[1], record["owner_id_or_ambiguous"])
                self.assertEqual(values[2], record["target_tooth_ids_or_ambiguous"])
                self.assertEqual("actual_attempt", record["actuality"])
                self.assertTrue(record["narration_eligibility"])

    def test_nonactual_attempts_are_retained_but_cannot_release_narration(self):
        expected = {
            "negated": ("dream", "narrative", "negated", "not_actual"),
            "hypothetical": (
                "dream_speech_or_thought",
                "thought_or_hypothetical",
                "conditional_hypothetical",
                "nonactual",
            ),
            "quoted": (
                "dream_speech_or_thought",
                "quoted_speech",
                "quoted",
                "reported_nonactual",
            ),
            "waking": ("waking", "narrative", "imagined", "nonactual"),
            "reported": (
                "dream_speech_or_thought",
                "reported_speech",
                "reported",
                "reported_nonactual",
            ),
        }
        for name, values in expected.items():
            with self.subTest(name=name):
                doctrine = self.doctrine(name)
                record = doctrine["restoration_attempt_records"][0]
                self.assertEqual(values[0], record["phase"])
                self.assertEqual(values[1], record["channel"])
                self.assertEqual(values[2], record["modality"])
                self.assertEqual(values[3], record["actuality"])
                self.assertFalse(record["narration_eligibility"])
                self.assertFalse(doctrine["restoration_attempted"])
                self.assertEqual([], doctrine["narration_consumed_attempt_ids"])
                narration = build_teeth_narration_facts(CASES[name])
                text = " ".join(narration["details"]).lower()
                self.assertNotIn("attempt to put the tooth back", text)

    def test_hypothetical_loss_does_not_release_a_warning(self):
        doctrine = self.doctrine("hypothetical")
        self.assertFalse(doctrine["active_fallout"])
        self.assertFalse(doctrine["active_warning"])
        self.assertEqual([], doctrine["applied_rule_ids"])

    def test_quote_report_waking_and_ambiguity_do_not_cross_bind(self):
        quoted = self.record("quoted")
        self.assertEqual("aunt", quoted["actor_id_or_ambiguous"])
        self.assertEqual("ambiguous", quoted["owner_id_or_ambiguous"])
        self.assertEqual("ambiguous", quoted["target_tooth_ids_or_ambiguous"])

        reported = self.record("reported")
        self.assertEqual("sister", reported["actor_id_or_ambiguous"])
        self.assertEqual("sister", reported["owner_id_or_ambiguous"])
        self.assertEqual(["sister-tooth-1"], reported["target_tooth_ids_or_ambiguous"])
        self.assertIsNone(reported["event_chain_id_or_null"])

        ambiguous = self.record("ambiguous")
        self.assertEqual("low", ambiguous["binding_confidence"])
        self.assertEqual("ambiguous", ambiguous["owner_id_or_ambiguous"])
        self.assertFalse(ambiguous["narration_eligibility"])

    def test_attempt_history_survives_a_real_terminal_return(self):
        doctrine = self.doctrine("then_firm")
        record = doctrine["restoration_attempt_records"][0]
        self.assertTrue(record["narration_eligibility"])
        self.assertEqual("attempted_only", record["completion"])
        self.assertTrue(doctrine["ending_precedence"])
        self.assertEqual("same_tooth_returned_firm", doctrine["terminal_ending"])
        self.assertEqual(["TEETH-END-TERMINAL"], doctrine["applied_rule_ids"])
        self.assertFalse(doctrine["restoration_attempted"])
        self.assertEqual([], doctrine["narration_consumed_attempt_ids"])

    def test_eligible_attempt_narration_names_its_consumed_record(self):
        doctrine = self.doctrine("dreamer")
        self.assertTrue(doctrine["restoration_attempted"])
        self.assertEqual(["attempt-1"], doctrine["restoration_attempt_contributing_ids"])
        self.assertEqual(["attempt-1"], doctrine["narration_consumed_attempt_ids"])
        self.assertFalse(doctrine["ending_precedence"])


if __name__ == "__main__":
    unittest.main()
