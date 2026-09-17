# Evaluation protocol

## Included benchmark

`benchmark.py` generates independent synthetic worlds. Each world has 24 device-mode facts, 16 known goal keys, and an initial memory containing four verified facts. Observations include correct claims, wrong claims, repetitions/paraphrases, irrelevant claims, and meaningless numeric noise. Some claims cite a measurement reference.

The investigator alone can access the measurement registry. An observation can cite a reference without revealing the measurement's answer. The selector receives only the event, retrieved memory, and goal keys. The evaluator alone scores answers against the world state. Device names and values are regenerated across seeds.

The fixture verifier is perfect for the supplied measurements. This is why the reference report has no wrong verified answers. It says nothing about a real model's calibration or hallucination rate.

## Training

For each training experience:

1. Capture pre-investigation features.
2. Run the fixture investigator.
3. Clone the current memory to create a counterfactual branch.
4. Insert a verified result in that branch and measure the change in number of correctly answered goal questions.
5. Fit the investigation head on pre-review features and the retention head on features including review status.
6. Randomly incorporate some verified results into the continuing training state so examples cover both known and unknown claims.

Default training seeds: 0–7. Default evaluation seeds: 100–109. The default hyperparameters were fixed before the reference run; no benchmark-driven search is included. The CLI permits custom seeds, so experiment authors must preserve disjoint training, validation, and test partitions themselves.

There is no generalization claim across task families. The training and test generators share the same distribution and supplied goal vocabulary structure.

## Deployment costs

Every processed event pays:

| Stage | Abstract units |
|---|---:|
| Scan | 1.0 |
| Retrieve | 0.2 |
| Controller | 0.1 |
| Investigation, when selected | 5.0 |
| Memory-write attempt, when selected | 0.2 |

The same rates and total budget apply to every policy. Selection can process more events by avoiding investigations. The common controller rate does not establish that the actual implementations have equal runtime. Deferred-buffer maintenance and audit overhead are represented only implicitly in the fixed rates, not individually measured. Offline training is excluded from deployment budgets and must be amortized before claiming end-to-end efficiency.

The default budget is 180 units and the stream has 160 events. Processing ends when there is not enough budget for the next scan/retrieval/control cycle. An investigation reserves sufficient room for a possible write. Failed investigations are charged and the observation is deferred; the run surfaces the error.

## Baselines

- `all`: investigate each event while affordable.
- `random`: independently investigate with probability 0.3 using a fixed seed.
- `surprise`: investigate if lexical novelty exceeds 0.65. The CLI name is shorthand; this is Jaccard distance from retrieved text, not LM surprisal.
- `deduplicate`: investigate if no verified answer is retrieved for the claim or its value conflicts.
- `learned`: use fitted gain estimates and a fixed resource price of 0.05.

All baseline retention policies retain verified results only. The learned policy has a separate retention estimate. A useful future ablation is to hold retention constant to isolate selection gains, then hold selection constant to isolate retention gains.

## Reproduce and vary

```bash
python3 -m novelty_extraction demo --output runs/demo
python3 -m novelty_extraction train --seeds 0 1 2 3 4 5 6 7 --output runs/controller.json
python3 -m novelty_extraction benchmark \
  --controller runs/controller.json --seeds 100 101 102 103 104 105 106 107 108 109 \
  --budget 180 --output runs/benchmark.json
```

`benchmarks/reference-report.json` contains all per-world results and cost ledgers, not just aggregate winners. Results are deterministic under the supplied implementation, subject to ordinary floating-point differences.

## Needed next benchmarks

1. Complementary clues, such as two observations that jointly identify an answer but individually have no value.
2. Verification noise, misleading sources, source dependence, and confident but incorrect model assessments.
3. Changing facts, scope changes, and conflicting evidence without arrival-order resolution.
4. Unannounced future tasks and shifts in relevance.
5. Real tool tasks with executable scoring and equal total token, latency, or monetary budgets.
6. Break-even experiments including selector training, retrieval, buffering, memory growth, and model adaptation.

Report false belief adoption, missed useful experiences, retention volume, delayed benefit, and cost alongside task accuracy. Do not use a model's self-rated novelty as the sole endpoint.
