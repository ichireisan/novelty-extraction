import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class CLITests(unittest.TestCase):
    def test_offline_example(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([
                sys.executable, "-m", "novelty_extraction", "run",
                "--events", "examples/events.jsonl", "--goals", "examples/goals.json",
                "--evidence", "examples/evidence.json", "--memory", str(Path(directory) / "memory.sqlite"),
            ], cwd=ROOT, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            rows = [json.loads(line) for line in result.stdout.splitlines()]
            self.assertEqual([row["decision"] for row in rows[:-1]], ["rejected", "stored", "unresolved"])

    def test_learned_policy_requires_controller(self):
        result = subprocess.run([
            sys.executable, "-m", "novelty_extraction", "run", "--events", "examples/events.jsonl",
            "--goals", "examples/goals.json", "--policy", "learned",
        ], cwd=ROOT, text=True, capture_output=True)
        self.assertEqual(result.returncode, 1)
        self.assertIn("requires --controller", result.stderr)
