import unittest

from app.rules import detect_rule_hits


BROKEN_ROW = {
    "state_name": "Broken",
    "keywords": "broken, cracked, chipped",
    "priority": "10",
    "active": "yes",
}

LOOSE_ROW = {
    "state_name": "Loose",
    "keywords": "loose, wobbly",
    "priority": "10",
    "active": "yes",
}

BLEEDING_ROW = {
    "state_name": "Bleeding",
    "keywords": "bleeding, blood",
    "priority": "10",
    "active": "yes",
}


class TeethStateNegationRegressionTests(unittest.TestCase):
    def _state_names(self, dream: str):
        hits = detect_rule_hits(
            dream=dream,
            rows=[BROKEN_ROW, LOOSE_ROW, BLEEDING_ROW],
            kind="state",
            max_hits=5,
        )
        return [hit["name"] for hit in hits]

    def test_not_broken_does_not_activate_broken_state(self):
        self.assertNotIn("Broken", self._state_names("My tooth was not broken."))

    def test_not_loose_does_not_activate_loose_state(self):
        self.assertNotIn("Loose", self._state_names("My tooth was not loose."))

    def test_not_bleeding_does_not_activate_bleeding_state(self):
        self.assertNotIn("Bleeding", self._state_names("My gums were not bleeding."))

    def test_never_bled_does_not_activate_blood_keyword(self):
        self.assertNotIn("Bleeding", self._state_names("There was never blood around my teeth."))

    def test_negated_broken_then_real_cracked_still_detects_broken(self):
        dream = "My front tooth was not broken, but a back tooth was cracked."
        self.assertIn("Broken", self._state_names(dream))

    def test_negated_bleeding_then_real_blood_still_detects_bleeding(self):
        dream = "My gums were not bleeding at first, but later there was blood around my teeth."
        self.assertIn("Bleeding", self._state_names(dream))

    def test_affirmative_states_still_detect(self):
        names = self._state_names("My tooth was wobbly, another was chipped, and my gums were bleeding.")
        self.assertIn("Loose", names)
        self.assertIn("Broken", names)
        self.assertIn("Bleeding", names)


if __name__ == "__main__":
    unittest.main()
