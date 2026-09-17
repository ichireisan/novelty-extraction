# Extending the reference implementation

## Replace the investigator

Implement the `Investigator` protocol:

```python
from novelty_extraction.types import Assessment, Status

class TestInvestigator:
    def investigate(self, event, memories):
        # Replace with an actual, bounded test of the event's scoped claim.
        return Assessment(Status.HYPOTHESIS, "A domain test is still needed")
```

Only return `Status.VERIFIED` when the configured evidence mechanism establishes the claim under its declared assumptions. Include evidence references in the assessment. The interface itself cannot determine whether a verifier is honest or competent.

## Replace the selection policy

Implement:

```text
investigate(features, estimated_investigation_and_write_cost) -> bool
retain(features, assessment, estimated_write_cost) -> bool
```

`LearnedPolicy` is the small linear reference. A neural head, contextual bandit, or policy operating on raw representations will need to extend the feature interface. Preserve the distinction between pre-investigation and post-investigation information.

## Use the loop in Python

```python
from novelty_extraction.controller import BaselinePolicy
from novelty_extraction.engine import Budget, Engine
from novelty_extraction.investigators import EvidenceInvestigator
from novelty_extraction.memory import Memory
from novelty_extraction.types import Claim, Experience

with Memory() as memory:
    verifier = EvidenceInvestigator({"test:1": Claim("service:S/retry", "requires-key")})
    engine = Engine(memory, verifier, BaselinePolicy("all"), Budget(30), {"service:S/retry"})
    result = engine.process(Experience(
        "event:1", "Retrying service S requires a key", "test-runner",
        Claim("service:S/retry", "requires-key"), ("test:1",),
    ))
    print(result)
    print(memory.answer("service:S/retry"))
```

## Memory and retrieval

`Memory.retrieve` prioritizes exact claim-key matches and then recent entries. It is intentionally a baseline. Replace it with a semantic index or hybrid retriever while keeping evidence and scope intact. Memory methods accept paths to persistent SQLite databases; the caller creates parent directories when using the Python API directly.

`Memory.deferred()` returns unresolved experiences. A caller can reconsider them under a new budget or after receiving related evidence. Automatic replay scheduling and complementary-clue discovery are not implemented. The default pending buffer holds 128 events; persistent beliefs and the audit are not capacity-limited.

`Memory.answer` returns only a verified value. A newer verified conflicting value supersedes older verified entries, preserving their records. A hypothesis cannot overwrite a verified answer. This policy needs domain-aware conflict resolution for scientific evidence or time-dependent facts.

## Claims and provenance

The caller supplies one optional structured claim per experience. If a document contains several claims, split it upstream while retaining a shared document reference and relevant context. Automatic extraction is an extension, not hidden inside the prototype.

Event IDs must be stable and unique. Evidence references must resolve to actual records in the configured verifier. The built-in `new_source` feature only checks retrieved source strings; it does not establish statistical independence between sources.

## Costs and failure handling

Customize `Costs` and `Budget` for simulations. For a real monetary cap, implement admission control using provider pricing and worst-case token limits, then reconcile measured usage. The current fixed investigation cost cannot enforce such a cap.

Investigator exceptions are charged, audited by error type, and surfaced. The event is deferred for explicit retry. No automatic provider retries are made. Concurrent or multi-process writes, schema migrations, and transactional coupling of remote calls to local storage are outside this prototype's scope.
