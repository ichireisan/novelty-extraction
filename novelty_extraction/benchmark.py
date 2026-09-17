"""Synthetic closed-world benchmark. No DeepSeek inference or real-world novelty claims."""

import json
import random
import statistics
from pathlib import Path

from .controller import BaselinePolicy, LearnedPolicy, fit_ridge
from .engine import Budget, Costs, Engine
from .features import features, retention_features
from .investigators import EvidenceInvestigator
from .memory import Memory
from .types import Assessment, Claim, Experience, Status


def scenario(seed: int, count: int = 160):
    rng = random.Random(seed)
    truth = {f"device-{seed}-{i}/mode": rng.choice(["safe", "fast", "manual"]) for i in range(24)}
    keys = list(truth)
    goals = set(keys[:16])
    evidence = {f"measurement-{seed}-{i}": Claim(key, truth[key]) for i, key in enumerate(keys)}
    events = []
    for i in range(count):
        if rng.random() < 0.18:
            text = " ".join(str(rng.getrandbits(40)) for _ in range(12))
            events.append(Experience(f"{seed}-noise-{i}", text, "noise"))
            continue
        index = rng.randrange(len(keys))
        key = keys[index]
        value = truth[key] if rng.random() < 0.70 else rng.choice(
            [v for v in ["safe", "fast", "manual"] if v != truth[key]]
        )
        refs = (f"measurement-{seed}-{index}",) if rng.random() < 0.70 else ()
        text = rng.choice([
            f"The recorded mode for {key} is {value}.",
            f"A report says {key} operates in {value} mode.",
            f"Observation: {key} = {value}.",
        ])
        events.append(Experience(f"{seed}-event-{i}", text, f"source-{rng.randrange(6)}", Claim(key, value), refs))
    return truth, goals, evidence, events


def initialize(memory: Memory, truth: dict[str, str]):
    for i, (key, value) in enumerate(list(truth.items())[:4]):
        event = Experience(f"initial-{i}", f"Initial measurement: {key} = {value}", "initial", Claim(key, value))
        memory.remember(event, Assessment(Status.VERIFIED, "Initial fixture measurement", (f"initial:{i}",)))


def score(memory: Memory, truth: dict[str, str], goals: set[str]) -> int:
    return sum(memory.answer(key) == truth[key] for key in goals)


def train(seeds=range(8), count: int = 160) -> tuple[LearnedPolicy, dict]:
    """Paired memory ablations on training-only worlds, with randomized state updates.

    Target = number of additional goal questions answered correctly after storing a
    reviewed claim. Both branches are evaluated on the same goal set. This is a myopic
    value target and does not model compositional or delayed rewards.
    """
    seeds = list(seeds)
    xs, ys, rx, ry = [], [], [], []
    positive = 0
    for seed in seeds:
        truth, goals, evidence, events = scenario(seed, count)
        investigator = EvidenceInvestigator(evidence)
        rng = random.Random(seed + 50_000)
        with Memory() as memory:
            initialize(memory, truth)
            for event in events:
                retrieved = memory.retrieve(event)
                x = features(event, retrieved, goals)
                before = score(memory, truth, goals)
                assessment = investigator.investigate(event, retrieved)
                with memory.clone() as counterfactual:
                    if assessment.status == Status.VERIFIED:
                        counterfactual.remember(event, assessment)
                    gain = score(counterfactual, truth, goals) - before
                xs.append(x)
                ys.append(float(gain))
                rx.append(retention_features(x, assessment))
                ry.append(float(gain))
                positive += gain > 0
                if rng.random() < 0.35 and assessment.status == Status.VERIFIED:
                    memory.remember(event, assessment)
    policy = LearnedPolicy(fit_ridge(xs, ys), fit_ridge(rx, ry))
    return policy, {"seeds": seeds, "examples": len(xs), "positive_gain_examples": positive,
                    "method": "paired synthetic memory ablations; frozen observable features"}


def evaluate(learned: LearnedPolicy, seeds=range(100, 110), budget_limit: float = 180.0,
             count: int = 160) -> dict:
    seeds = list(seeds)
    if not seeds or count <= 0:
        raise ValueError("Need evaluation seeds and a positive event count")
    runs = []
    for seed in seeds:
        truth, goals, evidence, events = scenario(seed, count)
        for mode in ("all", "random", "surprise", "deduplicate", "learned"):
            with Memory() as memory:
                initialize(memory, truth)
                initial = score(memory, truth, goals)
                budget = Budget(budget_limit)
                policy = learned if mode == "learned" else BaselinePolicy(mode, seed)
                engine = Engine(memory, EvidenceInvestigator(evidence), policy, budget, goals)
                decisions = []
                for event in events:
                    decision = engine.process(event)
                    if decision["decision"] == "budget_exhausted":
                        break
                    decisions.append(decision)
                correct = score(memory, truth, goals)
                answers = {key: memory.answer(key) for key in goals}
                runs.append({
                    "seed": seed, "policy": mode, "initial_correct": initial,
                    "correct": correct, "questions": len(goals), "accuracy": correct / len(goals),
                    "wrong_answers": sum(v is not None and v != truth[k] for k, v in answers.items()),
                    "processed": len(decisions), "investigated": sum(d["investigated"] for d in decisions),
                    "writes": sum(d["stored"] for d in decisions), "spent": round(budget.spent, 6),
                    "budget": budget_limit, "cost_ledger": dict(budget.ledger),
                    "gain_per_cost": (correct - initial) / budget.spent if budget.spent else 0.0,
                })
    summaries = []
    for mode in ("all", "random", "surprise", "deduplicate", "learned"):
        subset = [r for r in runs if r["policy"] == mode]
        summaries.append({"policy": mode, **{
            name: statistics.mean(r[name] for r in subset)
            for name in ("accuracy", "wrong_answers", "processed", "investigated", "spent", "gain_per_cost")
        }, "accuracy_stddev": statistics.stdev(r["accuracy"] for r in subset) if len(subset) > 1 else 0.0})
    return {
        "schema_version": 1, "benchmark": "synthetic fact acquisition v1", "evaluation_seeds": seeds,
        "cost_units": "simulated; not wall time, tokens, FLOPs, or dollars", "costs": vars(Costs()),
        "limitations": ["structured claims and supplied goal keys", "perfect fixture verification",
                        "myopic gains", "no DeepSeek run", "no compositional reasoning or distribution-shift test",
                        "training cost excluded from deployment budget; must be amortized in real use"],
        "summary": summaries, "runs": runs,
    }


def demo(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    policy, training = train()
    policy.save(output / "controller.json")
    report = evaluate(policy)
    report["training"] = training
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
