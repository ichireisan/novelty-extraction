"""Offline by default; live calls require the explicit --investigator deepseek switch."""

import argparse
import json
import sys
from pathlib import Path

from .benchmark import demo, evaluate, train
from .controller import BaselinePolicy, LearnedPolicy
from .engine import Budget, Engine
from .investigators import DeepSeekInvestigator, EvidenceInvestigator
from .memory import Memory
from .types import Claim, Experience


def print_report(report: dict):
    print("Synthetic benchmark — abstract cost units; no model inference")
    print(f"{'policy':<14} {'accuracy':>9} {'stddev':>9} {'events':>8} {'reviews':>8} {'cost':>8}")
    for row in report["summary"]:
        print(f"{row['policy']:<14} {row['accuracy']:>9.3f} {row['accuracy_stddev']:>9.3f} "
              f"{row['processed']:>8.1f} {row['investigated']:>8.1f} {row['spent']:>8.1f}")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Experimental budgeted selective learning")
    sub = root.add_subparsers(dest="command", required=True)
    d = sub.add_parser("demo", help="Train and evaluate offline on disjoint synthetic worlds")
    d.add_argument("--output", type=Path, default=Path("runs/demo"))
    t = sub.add_parser("train", help="Fit the two linear heads on synthetic paired outcomes")
    t.add_argument("--output", type=Path, default=Path("runs/controller.json"))
    t.add_argument("--seeds", type=int, nargs="+", default=list(range(8)))
    b = sub.add_parser("benchmark", help="Evaluate a saved controller")
    b.add_argument("--controller", type=Path, required=True)
    b.add_argument("--seeds", type=int, nargs="+", default=list(range(100, 110)))
    b.add_argument("--budget", type=float, default=180.0)
    b.add_argument("--output", type=Path, default=Path("runs/benchmark.json"))
    r = sub.add_parser("run", help="Process JSONL experiences into a persistent memory")
    r.add_argument("--events", type=Path, required=True)
    r.add_argument("--goals", type=Path, required=True, help="JSON array of relevant claim keys")
    r.add_argument("--memory", type=Path, default=Path("runs/memory.sqlite"))
    r.add_argument("--budget", type=float, default=100.0)
    r.add_argument("--policy", choices=["all", "random", "surprise", "deduplicate", "learned"], default="all")
    r.add_argument("--controller", type=Path)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--investigator", choices=["evidence", "deepseek"], default="evidence")
    r.add_argument("--evidence", type=Path, help="Explicitly trusted measurement registry; empty if omitted")
    r.add_argument("--model", default="deepseek-flash")
    r.add_argument("--base-url", default="https://api.deepseek.com")
    return root


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "demo":
            report = demo(args.output)
            print_report(report)
            print(f"Artifacts: {args.output.resolve()}")
        elif args.command == "train":
            model, metadata = train(args.seeds)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            model.save(args.output)
            print(json.dumps(metadata))
        elif args.command == "benchmark":
            report = evaluate(LearnedPolicy.load(args.controller), args.seeds, args.budget)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            print_report(report)
        else:
            run_stream(args)
    except (ValueError, RuntimeError, OSError, TypeError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def run_stream(args):
    if args.policy == "learned" and args.controller is None:
        raise ValueError("--policy learned requires --controller")
    policy = LearnedPolicy.load(args.controller) if args.policy == "learned" else BaselinePolicy(args.policy, args.seed)
    goals = json.loads(args.goals.read_text(encoding="utf-8"))
    if not isinstance(goals, list) or not all(isinstance(key, str) for key in goals):
        raise ValueError("Goals must be a JSON array of claim keys")
    if args.investigator == "deepseek":
        investigator = DeepSeekInvestigator(model=args.model, base_url=args.base_url)
    else:
        data = json.loads(args.evidence.read_text(encoding="utf-8")) if args.evidence else {}
        investigator = EvidenceInvestigator({ref: Claim(**claim) for ref, claim in data.items()})
    args.memory.parent.mkdir(parents=True, exist_ok=True)
    budget = Budget(args.budget)
    with Memory(args.memory) as memory, args.events.open(encoding="utf-8") as stream:
        engine = Engine(memory, investigator, policy, budget, set(goals))
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                event = Experience.from_dict(json.loads(line))
            except (ValueError, TypeError) as exc:
                raise ValueError(f"Invalid experience on line {line_number}: {exc}") from exc
            decision = engine.process(event)
            print(json.dumps(decision))
            if decision["decision"] == "budget_exhausted":
                break
    summary = {"spent": budget.spent, "cost_ledger": dict(budget.ledger)}
    if isinstance(investigator, DeepSeekInvestigator):
        summary["provider_usage"] = investigator.usage
    print(json.dumps(summary))
