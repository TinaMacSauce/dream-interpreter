from pathlib import Path
import unittest


ADMIN_ROUTE_PATH = (
    Path(__file__).resolve().parents[1]
    / "app"
    / "routes"
    / "admin.py"
)


class QaAccessContractTests(unittest.TestCase):
    def test_qa_grant_is_admin_protected_and_bounded(self):
        source = ADMIN_ROUTE_PATH.read_text(encoding="utf-8")

        route = '@admin_bp.route("/admin/qa-grant", methods=["POST", "OPTIONS"])'
        self.assertIn(route, source)

        start = source.index(route)
        qa_slice = source[start:]

        self.assertIn("auth_fail = require_admin()", qa_slice)
        self.assertIn('QA_EMAIL_DOMAIN = "qa.jamaicantruestories.com"', source)
        self.assertIn("maximum=QA_MAX_USES", qa_slice)
        self.assertIn("maximum=QA_MAX_HOURS", qa_slice)
        self.assertIn("mark_dream_pack_purchase(", qa_slice)
        self.assertIn("set_buyer_session(email)", qa_slice)
        self.assertIn('"access_type": "temporary_qa"', qa_slice)

    def test_qa_grant_has_hard_caps(self):
        source = ADMIN_ROUTE_PATH.read_text(encoding="utf-8")

        self.assertIn("QA_MAX_USES = 50", source)
        self.assertIn("QA_MAX_HOURS = 6", source)
        self.assertIn("QA_DEFAULT_USES = 25", source)
        self.assertIn("QA_DEFAULT_HOURS = 2", source)


if __name__ == "__main__":
    unittest.main()
