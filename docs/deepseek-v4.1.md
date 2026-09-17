# Proposed DeepSeek-V4.1-Flash integration

Status: **design proposal, not a working internal patch**. Sources were consulted on 2026-09-17. See the official [model card](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash), [configuration](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/config.json), and [inference implementation](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/inference/model.py).

## Relevant published architecture

DeepSeek describes 20 causal encoder layers followed by 20 decoder layers. Decoder global KV is projected from final encoder states. Its reported active parameter counts are 8B during prefill and 16B during decoding. MoE layers use one shared expert and six selected from 384 routed experts. CSA2 shares/reuses sparse-attention infrastructure. Engram provides learned token-based conditional memory. The configuration specifies width 5120 and four hyper-connection streams.

These properties offer candidate attachment points, not evidence that hidden states already encode a calibrated novelty score. Parameter counts do not directly determine wall-clock savings. The readable reference forward loops through the backbone; an efficient production integration must preserve CED-specific serving optimizations rather than assuming that adding a hook automatically obtains the advertised prefill behavior.

## Model-level proposal

At each event boundary:

1. Read the encoder's residual representation before it is discarded or projected into a more compressed cache.
2. Combine hyper-connection streams with a trained readout, then pool the event span. Test boundary pooling against token pooling; qualifiers can be lost by naïve averaging.
3. Project the result to a compact query and retrieve relevant persistent memories. Train the query projection for retrieval; raw hidden-state distance is not automatically semantic similarity.
4. Feed the event representation, retrieved memory representation, current goal, source metadata, and remaining budget to a small investigation head.
5. Schedule an optional reasoning pass or tool call. Preserve normal input processing and causal cache invariants.
6. Feed the review result to a separate retention head. Store supported changes and evidence through the memory interface.

Illustrative shapes, **not tested DeepSeek APIs**:

```text
encoder residual: [batch, tokens, streams=4, width=5120]
trained stream readout -> [batch, tokens, 5120]
event pooling and projection -> [batch, events, 256]
retrieved memory projection -> [batch, events, 256]
concat(event, memory, goal, provenance, budget) -> value head
value head -> predicted gain for each allowed action
```

A prototype should freeze the backbone and train only the readout, retrieval projection, and heads. Export just event summaries from the device; transferring all intermediate states to the CPU can erase any savings. Compare actual latency, throughput, and memory usage against the same serving stack without the controller.

## Training the readout

Collect paired rollouts from identical agent states. One receives ordinary processing; another receives a specified additional computation. Use independently scored outcomes and held-out subsequent tasks to estimate benefit. For retention labels, compare identical agents with and without the proposed memory write.

Train a regression or action-value head on those outcomes. Calibrate on separate validation data. Retain randomized exploration and log selection propensities if learning later from selectively observed deployment outcomes. Do not train the selector on future answers or on the verifier result before it chooses to investigate.

For prediction-error features, predictions must be formed before the outcome is revealed. Computing surprise from a representation that has already seen its own target creates leakage. Full-vocabulary probabilities also require computation; they are not necessarily available at an encoder-only attachment point.

## Existing components are not substitutes

- **MoE routing** selects experts. Router scores are not automatically uncertainty estimates or estimates of the value of another computation. Changing six active experts to a larger number requires training and evaluation.
- **CSA2 indexing** selects context for attention. Attention importance and value of long-term retention are different objectives.
- **Engram** uses learned n-gram lookups. It is not an append-only episodic database into which arbitrary new claims can be inserted.
- **KV caching** preserves inference state; it is not a provenance-aware belief store.
- **Speculative decoding** accelerates token generation; acceptance or confidence scores do not certify factual truth.

## API-level implementation included here

`DeepSeekInvestigator` sends the selected observation and retrieved entries through Chat Completions. It returns an unresolved assessment, never a verification. The current wrapper does not expose encoder states, numeric reasoning-effort controls, internal routing, or incremental cache surgery.

This adapter is tested with mocked HTTP responses only. A contributor should run opt-in compatibility tests against the actual service and record model identifier, date, request options, costs, and outcomes before claiming live support. Authentication uses `DEEPSEEK_API_KEY`; no credentials are stored in source.

The synthetic controller is not pretrained for natural-language streams. Its saved weights are a demonstration artifact, not a recommended production DeepSeek controller.
