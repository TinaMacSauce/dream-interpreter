import unittest

from app.rules import detect_rule_hits


TEETH_FALLING_ROW = {
    "behavior_name": "Teeth Falling Out",
    "keywords": "teeth fell out, teeth falling out, fall out, falling out, fell out, came out, coming out",
    "active": "yes",
    "priority": "10",
}


class TeethNegationRegressionTests(unittest.TestCase):
    def _names(self, dream: str):
        hits = detect_rule_hits(
            dream=dream,
            rows=[TEETH_FALLING_ROW],
            kind="behavior",
            max_hits=5,
        )
        return {str(hit.get("name", "")).lower() for hit in hits}

    def test_affirmative_teeth_falling_still_matches(self):
        self.assertIn("teeth falling out", self._names("My teeth fell out."))

    def test_did_not_fall_out_does_not_match(self):
        self.assertNotIn("teeth falling out", self._names("My teeth did not fall out."))

    def test_never_fell_out_does_not_match(self):
        self.assertNotIn("teeth falling out", self._names("My teeth never fell out."))

    def test_didnt_fall_out_does_not_match_after_normalization(self):
        self.assertNotIn("teeth falling out", self._names("My teeth didn't fall out."))

    def test_no_teeth_fell_out_does_not_match(self):
        self.assertNotIn("teeth falling out", self._names("No teeth fell out."))


if __name__ == "__main__":
    unittest.main()
