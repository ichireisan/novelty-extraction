"""Two fitted linear value heads and transparent comparison policies.

These regressions are a small runnable baseline, not the proposed neural encoder head.
"""

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

from .features import FEATURE_NAMES, RETENTION_NAMES, retention_features
from .types import Assessment, Status


def fit_ridge(rows: list[list[float]], targets: list[float], ridge: float = 0.1) -> list[float]:
    if not rows or len(rows) != len(targets) or ridge <= 0:
        raise ValueError("Need nonempty paired training data and positive regularization")
    n = len(rows[0])
    if not n or any(len(row) != n for row in rows) or not all(
        math.isfinite(v) for row in rows for v in row
    ) or not all(math.isfinite(v) for v in targets):
        raise ValueError("Training data must be rectangular and finite")
    # Solve (X'X + ridge I)w = X'y using partial-pivot Gaussian elimination.
    matrix = [[sum(row[i] * row[j] for row in rows) + (ridge if i == j else 0.0)
               for j in range(n)] + [sum(row[i] * y for row, y in zip(rows, targets))]
              for i in range(n)]
    for i in range(n):
        pivot = max(range(i, n), key=lambda j: abs(matrix[j][i]))
        matrix[i], matrix[pivot] = matrix[pivot], matrix[i]
        divisor = matrix[i][i]
        if abs(divisor) < 1e-12:
            raise ValueError("Ill-conditioned training matrix")
        matrix[i] = [v / divisor for v in matrix[i]]
        for j in range(n):
            if j != i:
                factor = matrix[j][i]
                matrix[j] = [a - factor * b for a, b in zip(matrix[j], matrix[i])]
    return [row[-1] for row in matrix]


def predict(weights: list[float], x: list[float]) -> float:
    if len(weights) != len(x) or not all(math.isfinite(v) for v in x):
        raise ValueError("Feature shape or value mismatch")
    return sum(w * v for w, v in zip(weights, x))


@dataclass
class LearnedPolicy:
    investigation_weights: list[float]
    retention_weights: list[float]
    cost_weight: float = 0.05

    def __post_init__(self):
        if len(self.investigation_weights) != len(FEATURE_NAMES) or len(self.retention_weights) != len(RETENTION_NAMES):
            raise ValueError("Controller feature schema mismatch")
        if not all(math.isfinite(v) for v in self.investigation_weights + self.retention_weights):
            raise ValueError("Weights must be finite")
        if not math.isfinite(self.cost_weight) or self.cost_weight < 0:
            raise ValueError("cost_weight must be finite and nonnegative")

    def investigate(self, x: list[float], cost: float) -> bool:
        return predict(self.investigation_weights, x) > self.cost_weight * cost

    def retain(self, x: list[float], assessment: Assessment, cost: float) -> bool:
        return assessment.status != Status.REJECTED and predict(
            self.retention_weights, retention_features(x, assessment)
        ) > self.cost_weight * cost

    def save(self, path: str | Path):
        Path(path).write_text(json.dumps({
            "schema_version": 1, "features": FEATURE_NAMES, "retention_features": RETENTION_NAMES,
            "investigation_weights": self.investigation_weights,
            "retention_weights": self.retention_weights, "cost_weight": self.cost_weight,
        }, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "LearnedPolicy":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("schema_version") != 1 or data.get("features") != list(FEATURE_NAMES) or data.get("retention_features") != list(RETENTION_NAMES):
            raise ValueError("Unsupported controller schema")
        return cls(data["investigation_weights"], data["retention_weights"], data["cost_weight"])


class BaselinePolicy:
    def __init__(self, mode: str, seed: int = 0, probability: float = 0.3):
        if mode not in {"all", "random", "surprise", "deduplicate"}:
            raise ValueError("Unknown baseline")
        if not 0 <= probability <= 1:
            raise ValueError("probability must be between zero and one")
        self.mode, self.rng, self.probability = mode, random.Random(seed), probability

    def investigate(self, x: list[float], cost: float) -> bool:
        if self.mode == "random":
            return self.rng.random() < self.probability
        if self.mode == "surprise":
            # Lexical distance baseline, not calibrated LM surprisal.
            return x[7] > 0.65
        if self.mode == "deduplicate":
            return not x[3] or bool(x[4])
        return True

    def retain(self, x: list[float], assessment: Assessment, cost: float) -> bool:
        return assessment.status == Status.VERIFIED
