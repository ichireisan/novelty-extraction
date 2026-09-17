"""Cheap observable features; no access to benchmark answers or verification results."""

import math
import re
from collections.abc import Sequence

from .types import Assessment, Experience, Status

FEATURE_NAMES = (
    "bias", "has_claim", "goal_relevant", "known_verified", "conflicts_with_verified",
    "has_evidence_reference", "new_source", "lexical_novelty", "log_length",
)
RETENTION_NAMES = FEATURE_NAMES + ("verified", "hypothesis", "rejected")


def features(event: Experience, memories: Sequence[dict], goal_keys: set[str]) -> list[float]:
    words = set(re.findall(r"\w+", event.text.lower()))
    similarities = []
    for item in memories:
        other = set(re.findall(r"\w+", item["text"].lower()))
        similarities.append(len(words & other) / max(1, len(words | other)))
    matching = [m for m in memories if event.claim and m["claim_key"] == event.claim.key
                and m["status"] == "verified"]
    return [
        1.0,
        float(event.claim is not None),
        float(event.claim is not None and event.claim.key in goal_keys),
        float(bool(matching)),
        float(bool(matching) and any(m["value"] != event.claim.value for m in matching)),
        float(bool(event.evidence_refs)),
        float(not any(m["source"] == event.source for m in memories)),
        1.0 - max(similarities, default=0.0),
        min(math.log1p(len(event.text.split())) / 10.0, 1.0),
    ]


def retention_features(x: list[float], assessment: Assessment) -> list[float]:
    return x + [float(assessment.status == s) for s in Status]
