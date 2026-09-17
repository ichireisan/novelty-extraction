"""Budgeted control loop. Costs are abstract units, not provider billing estimates."""

import math
from collections import defaultdict
from dataclasses import dataclass, field

from .features import features
from .memory import Memory
from .types import Experience, Investigator, Status


@dataclass
class Costs:
    scan: float = 1.0
    retrieve: float = 0.2
    control: float = 0.1
    investigate: float = 5.0
    write: float = 0.2

    def __post_init__(self):
        if not all(math.isfinite(v) and v >= 0 for v in vars(self).values()):
            raise ValueError("Costs must be finite and nonnegative")


@dataclass
class Budget:
    limit: float
    spent: float = field(default=0.0, init=False)
    ledger: dict = field(default_factory=lambda: defaultdict(float), init=False)

    def __post_init__(self):
        if not math.isfinite(self.limit) or self.limit < 0:
            raise ValueError("Budget must be finite and nonnegative")

    def can_afford(self, amount: float) -> bool:
        if not math.isfinite(amount) or amount < 0:
            raise ValueError("Charge must be finite and nonnegative")
        return self.spent + amount <= self.limit + 1e-9

    def charge(self, stage: str, amount: float):
        if not self.can_afford(amount):
            raise ValueError("Budget exceeded")
        self.spent += amount
        self.ledger[stage] += amount


class Engine:
    def __init__(self, memory: Memory, investigator: Investigator, policy, budget: Budget,
                 goal_keys: set[str], costs: Costs | None = None):
        self.memory, self.investigator, self.policy = memory, investigator, policy
        self.budget, self.goal_keys, self.costs = budget, goal_keys, costs or Costs()

    def process(self, event: Experience) -> dict:
        c = self.costs
        result = {"event_id": event.event_id, "investigated": False, "stored": False}
        if not self.budget.can_afford(c.scan + c.retrieve + c.control):
            return {**result, "decision": "budget_exhausted", "spent": self.budget.spent}
        self.budget.charge("scan", c.scan)
        memories = self.memory.retrieve(event)
        self.budget.charge("retrieve", c.retrieve)
        x = features(event, memories, self.goal_keys)
        investigate = self.policy.investigate(x, c.investigate + c.write)
        self.budget.charge("control", c.control)
        if not investigate or not self.budget.can_afford(c.investigate + c.write):
            self.memory.defer(event)
            result["decision"] = "deferred" if not investigate else "deferred_budget"
        else:
            # Charge attempted work even if the provider fails. Do not turn a failure into evidence.
            self.budget.charge("investigate", c.investigate)
            result["investigated"] = True
            try:
                assessment = self.investigator.investigate(event, memories)
            except Exception as exc:
                self.memory.defer(event)
                self.memory.log({**result, "decision": "investigation_error", "error_type": type(exc).__name__,
                                 "spent": self.budget.spent})
                raise
            result.update(status=assessment.status.value, reason=assessment.reason)
            if self.policy.retain(x, assessment, c.write):
                self.budget.charge("write", c.write)
                result["stored"] = self.memory.remember(event, assessment)
                result["decision"] = "stored" if result["stored"] else "unchanged"
            elif assessment.status == Status.HYPOTHESIS:
                self.memory.defer(event)
                result["decision"] = "unresolved"
            else:
                result["decision"] = "rejected" if assessment.status == Status.REJECTED else "not_retained"
        result["spent"] = self.budget.spent
        self.memory.log(result)
        return result
