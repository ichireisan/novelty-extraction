"""Small, provider-independent data contracts."""

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Protocol, Sequence


class Status(str, Enum):
    VERIFIED = "verified"
    HYPOTHESIS = "hypothesis"
    REJECTED = "rejected"


@dataclass(frozen=True)
class Claim:
    # Include applicability conditions in the key, e.g. "service:S/retry/timeout".
    key: str
    value: str

    def __post_init__(self):
        if not all(isinstance(x, str) and x.strip() for x in (self.key, self.value)):
            raise ValueError("Claim key and value must be nonempty strings")


@dataclass(frozen=True)
class Experience:
    event_id: str
    text: str
    source: str
    claim: Claim | None = None
    evidence_refs: tuple[str, ...] = ()

    def __post_init__(self):
        if not all(isinstance(x, str) and x.strip() for x in (self.event_id, self.text, self.source)):
            raise ValueError("event_id, text, and source must be nonempty strings")
        if not isinstance(self.evidence_refs, tuple) or not all(
            isinstance(x, str) and x.strip() for x in self.evidence_refs
        ):
            raise ValueError("evidence_refs must be a tuple of nonempty strings")
        if self.claim is not None and not isinstance(self.claim, Claim):
            raise ValueError("claim must be a Claim or None")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Experience":
        allowed = {"event_id", "text", "source", "claim", "evidence_refs"}
        if not isinstance(data, dict) or set(data) - allowed:
            raise ValueError("Invalid experience fields")
        refs = data.get("evidence_refs", [])
        if not isinstance(refs, (list, tuple)):
            raise ValueError("evidence_refs must be a list")
        try:
            claim = Claim(**data["claim"]) if data.get("claim") is not None else None
            return cls(data["event_id"], data["text"], data["source"], claim, tuple(refs))
        except (KeyError, TypeError) as exc:
            raise ValueError("Invalid experience schema") from exc


@dataclass(frozen=True)
class Assessment:
    status: Status
    reason: str
    evidence_refs: tuple[str, ...] = ()

    def __post_init__(self):
        if not isinstance(self.status, Status):
            raise ValueError("Assessment status must be a Status")
        if self.status == Status.VERIFIED and not self.evidence_refs:
            raise ValueError("Verification requires evidence references")


class Investigator(Protocol):
    def investigate(self, event: Experience, memories: Sequence[dict]) -> Assessment:
        """VERIFIED must mean checked by this adapter's documented verifier."""
        ...
