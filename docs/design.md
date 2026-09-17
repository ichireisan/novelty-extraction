# Research design

## Objective

For memory state M, experience x, future task distribution Q, and action a, estimate:

```text
ΔV(a | M, x, Q) = E[U(agent after a, future tasks) − U(agent without a, future tasks)]
net_value(a) = ΔV(a) − λ C(a)
```

Keep the downstream reasoning budget comparable when measuring capability. Include all additional selection, investigation, retrieval, maintenance, and learning costs when making an efficiency claim.

This is an operational research objective grounded in value-of-information and metareasoning ideas. It is not a newly proven universal measure of novelty.

## Two decisions

**Investigation:** Is another retrieval, reasoning pass, experiment, or tool call likely to improve future outcomes enough to pay for itself?

**Retention:** After investigating, does the supported change deserve persistent storage? An unresolved hypothesis can be stored as such without becoming a verified belief.

The reference policy fits separate linear regressions. Investigation features are available before review. Retention features additionally include the review status. Both target the paired improvement in the synthetic task score. Neither target includes resource cost; cost is applied at selection time.

The current value estimate is myopic: it measures the immediate improvement in a fixed question set after a proposed memory write. A later research version should estimate delayed return and effects on future search, experiments, and computation.

## Observable signals

The prototype exposes: presence of a claim; relevance to supplied goal keys; an existing verified answer; contradiction with that answer; evidence-reference availability; source novelty within retrieved entries; lexical novelty; and text length. A bias term completes the feature vector.

The absence of an evidence reference is not proof that a claim is false. A new source is not proof of independent evidence. A lexical mismatch is not semantic novelty. These are explicitly limited predictors whose utility is evaluated on outcomes.

The proposed model-level implementation substitutes learned representations from the encoder for most hand-designed features. Explicit provenance and resource-budget features should remain available to the controller.

## Memory semantics

- `verified`: accepted by the configured verifier, under its documented assumptions.
- `hypothesis`: a candidate statement that must not supply a verified answer.
- `superseded`: a previously verified value replaced by a later verified value for the same scoped key.
- Rejected claims are recorded in the decision audit rather than inserted as beliefs.
- Unselected and unresolved experiences enter a bounded FIFO buffer. The prototype does not automatically replay it.

Automatic supersession assumes the verifier establishes that the new value should govern the same scoped key. Real domains need timestamps, applicability conditions, and conflict resolution beyond this simple rule. Two contradictory papers should not be resolved merely by arrival order.

## Learning versus persistence

Three processes are distinct:

1. Adding context changes a response within a model invocation.
2. Updating persistent memory changes what future invocations can retrieve.
3. Training changes model parameters.

The demo performs (2) and trains the small controller offline. It does not train DeepSeek or change a foundation model's weights. Adapter consolidation, replay, and continual learning are research extensions.

## Biological inspiration

The motivating analogy is selective allocation of attention and memory in biological learners. Hippocampal mismatch detection and reward-prediction-error studies motivate useful distinctions, but the repository makes no claim that these modules reproduce specific brain circuits or that biology implements the objective above. See [references](references.md).
