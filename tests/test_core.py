import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError

from novelty_extraction.benchmark import evaluate, train
from novelty_extraction.controller import BaselinePolicy, LearnedPolicy, fit_ridge
from novelty_extraction.engine import Budget, Costs, Engine
from novelty_extraction.features import FEATURE_NAMES, features
from novelty_extraction.investigators import DeepSeekInvestigator, EvidenceInvestigator
from novelty_extraction.memory import Memory
from novelty_extraction.types import Assessment, Claim, Experience, Status


def event(identifier="e1", value="yes", refs=("measurement",)):
    return Experience(identifier, f"The answer to question K is {value}", "source", Claim("K", value), refs)


def verified():
    return Assessment(Status.VERIFIED, "Checked measurement", ("measurement",))


class MemoryTests(unittest.TestCase):
    def setUp(self):
        self.memory = Memory(pending_limit=2)
        self.addCleanup(self.memory.close)

    def test_hypothesis_cannot_answer(self):
        self.memory.remember(event(), Assessment(Status.HYPOTHESIS, "Unconfirmed"))
        self.assertIsNone(self.memory.answer("K"))

    def test_verified_revision_preserves_history(self):
        self.memory.remember(event(), verified())
        self.memory.remember(event("e2", "no"), verified())
        self.assertEqual(self.memory.answer("K"), "no")
        self.assertEqual([r["status"] for r in self.memory.records()], ["superseded", "verified"])

    def test_unverified_revision_cannot_replace_verified(self):
        self.memory.remember(event(), verified())
        self.memory.remember(event("e2", "no"), Assessment(Status.HYPOTHESIS, "Unknown"))
        self.assertEqual(self.memory.answer("K"), "yes")

    def test_idempotency_and_identity_collision(self):
        self.assertTrue(self.memory.remember(event(), verified()))
        self.assertFalse(self.memory.remember(event(), verified()))
        with self.assertRaises(ValueError):
            self.memory.remember(event(value="no"), verified())

    def test_hypothesis_can_be_promoted(self):
        self.memory.remember(event(), Assessment(Status.HYPOTHESIS, "Unknown"))
        self.assertTrue(self.memory.remember(event(), verified()))
        self.assertEqual(self.memory.answer("K"), "yes")

    def test_pending_is_bounded_and_removed_on_write(self):
        for i in range(3):
            self.memory.defer(event(str(i)))
        self.assertEqual([e.event_id for e in self.memory.deferred()], ["1", "2"])
        self.memory.remember(event("2"), verified())
        self.assertEqual([e.event_id for e in self.memory.deferred()], ["1"])

    def test_persistence_and_clone_isolation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memory.sqlite"
            with Memory(path) as persistent:
                persistent.remember(event(), verified())
                with persistent.clone() as clone:
                    clone.remember(event("e2", "no"), verified())
                self.assertEqual(persistent.answer("K"), "yes")
            with Memory(path) as reopened:
                self.assertEqual(reopened.answer("K"), "yes")

    def test_rejected_claim_is_not_written(self):
        self.assertFalse(self.memory.remember(event(), Assessment(Status.REJECTED, "Disproved")))
        self.assertEqual(self.memory.records(), [])


class ControlTests(unittest.TestCase):
    def test_budget_never_exceeded(self):
        for limit in (0, 0.1, 1.3, 6.49, 6.5, 20, 100):
            with self.subTest(limit=limit), Memory() as memory:
                budget = Budget(limit)
                engine = Engine(memory, EvidenceInvestigator({"measurement": Claim("K", "yes")}),
                                BaselinePolicy("all"), budget, {"K"})
                for i in range(30):
                    engine.process(event(str(i)))
                self.assertLessEqual(budget.spent, limit + 1e-9)
                self.assertAlmostEqual(sum(budget.ledger.values()), budget.spent)

    def test_cost_validation(self):
        for value in (-1, math.nan, math.inf):
            with self.assertRaises(ValueError):
                Budget(value)
            with self.assertRaises(ValueError):
                Costs(write=value)
        with self.assertRaises(ValueError):
            Budget(1).charge("bad", -1)

    def test_observable_features_distinguish_conflict(self):
        with Memory() as memory:
            memory.remember(event(), verified())
            x = features(event("e2", "no"), memory.retrieve(event()), {"K"})
            self.assertEqual(len(x), len(FEATURE_NAMES))
            self.assertEqual(x[3:5], [1, 1])
            self.assertTrue(all(math.isfinite(v) for v in x))

    def test_failed_investigation_is_charged_and_deferred(self):
        class Failure:
            def investigate(self, *args):
                raise RuntimeError("Unavailable")
        with Memory() as memory:
            budget = Budget(20)
            engine = Engine(memory, Failure(), BaselinePolicy("all"), budget, {"K"})
            with self.assertRaises(RuntimeError):
                engine.process(event())
            self.assertEqual(budget.ledger["investigate"], 5)
            self.assertEqual(len(memory.deferred()), 1)
            self.assertEqual(memory.records(), [])
            audit = json.loads(memory.db.execute("SELECT payload FROM audit").fetchone()[0])
            self.assertEqual(audit["decision"], "investigation_error")

    def test_zero_selection_still_pays_scan_cost(self):
        with Memory() as memory:
            budget = Budget(20)
            engine = Engine(memory, EvidenceInvestigator({}), BaselinePolicy("random", probability=0), budget, {"K"})
            self.assertEqual(engine.process(event())["decision"], "deferred")
            self.assertAlmostEqual(budget.spent, 1.3)


class EvidenceTests(unittest.TestCase):
    def test_unknown_evidence_is_not_verification(self):
        result = EvidenceInvestigator({}).investigate(event(), [])
        self.assertEqual(result.status, Status.HYPOTHESIS)

    def test_conflicting_registry_measurements_remain_unresolved(self):
        investigator = EvidenceInvestigator({"a": Claim("K", "yes"), "b": Claim("K", "no")})
        self.assertEqual(investigator.investigate(event(refs=("a", "b")), []).status, Status.HYPOTHESIS)

    def test_wrong_claim_is_rejected(self):
        investigator = EvidenceInvestigator({"measurement": Claim("K", "yes")})
        self.assertEqual(investigator.investigate(event(value="no"), []).status, Status.REJECTED)

    def test_verification_needs_evidence(self):
        with self.assertRaises(ValueError):
            Assessment(Status.VERIFIED, "Trust me")

    def test_schema_rejects_string_evidence_list(self):
        data = event().to_dict()
        data["evidence_refs"] = "measurement"
        with self.assertRaises(ValueError):
            Experience.from_dict(data)

    @patch("novelty_extraction.investigators.urlopen")
    def test_model_self_confidence_never_becomes_verified(self, mock_open):
        mock_open.return_value.__enter__.return_value.read.return_value = json.dumps({
            "choices": [{"message": {"content": "Absolutely verified and correct!"}}],
            "usage": {"total_tokens": 12},
        }).encode()
        investigator = DeepSeekInvestigator(api_key="test-only")
        result = investigator.investigate(event(), [])
        self.assertEqual(result.status, Status.HYPOTHESIS)
        self.assertEqual(investigator.usage[0]["total_tokens"], 12)
        request = mock_open.call_args.args[0]
        self.assertEqual(json.loads(request.data)["model"], "deepseek-flash")

    @patch("novelty_extraction.investigators.urlopen")
    def test_http_error_is_sanitized(self, mock_open):
        mock_open.side_effect = HTTPError("https://endpoint", 401, "sensitive detail", {}, None)
        with self.assertRaisesRegex(RuntimeError, "HTTP 401") as caught:
            DeepSeekInvestigator(api_key="test-secret").investigate(event(), [])
        self.assertNotIn("test-secret", str(caught.exception))

    @patch("novelty_extraction.investigators.urlopen")
    def test_malformed_api_result_is_an_error(self, mock_open):
        mock_open.return_value.__enter__.return_value.read.return_value = b'{"choices": []}'
        with self.assertRaises(RuntimeError):
            DeepSeekInvestigator(api_key="test-only").investigate(event(), [])


class LearningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.policy, cls.metadata = train(range(2), count=40)

    def test_fit_can_learn_a_simple_signal(self):
        weights = fit_ridge([[1, 0], [1, 1], [1, 2], [1, 3]], [0, 2, 4, 6], ridge=0.001)
        self.assertAlmostEqual(weights[1], 2, places=2)

    def test_controller_roundtrip_and_schema_check(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "controller.json"
            self.policy.save(path)
            self.assertEqual(self.policy, LearnedPolicy.load(path))
            data = json.loads(path.read_text())
            data["features"] = ["wrong"]
            path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                LearnedPolicy.load(path)

    def test_evaluation_is_deterministic_and_budgeted(self):
        report = evaluate(self.policy, [100, 101], count=30, budget_limit=40)
        self.assertEqual(report, evaluate(self.policy, [100, 101], count=30, budget_limit=40))
        self.assertTrue(set(self.metadata["seeds"]).isdisjoint(report["evaluation_seeds"]))
        self.assertEqual(len(report["summary"]), 5)
        for run in report["runs"]:
            self.assertLessEqual(run["spent"], 40 + 1e-9)
            self.assertEqual(run["wrong_answers"], 0)

    def test_invalid_training_data_rejected(self):
        for rows, targets in (([], []), ([[1], [1, 2]], [1, 2]), ([[math.nan]], [1])):
            with self.assertRaises(ValueError):
                fit_ridge(rows, targets)


if __name__ == "__main__":
    unittest.main()
