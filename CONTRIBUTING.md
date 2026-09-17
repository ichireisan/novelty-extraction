# Contributing

Contributions are welcome: implementations, reproducible experiments, counterexamples, literature corrections, and clearer explanations.

## Start locally

```bash
python3 -m unittest discover -s tests -v
python3 -m novelty_extraction demo --output runs/demo
```

The core uses Python's standard library. Keep optional model integrations isolated so the demo remains usable without model weights, GPUs, or credentials. Follow the existing type annotations and small interfaces.

## Propose a focused change

For a research extension, state the hypothesis, baseline, evaluation distribution, cost accounting, and a result that would falsify the hypothesis. A negative result with a reproducible experiment is useful.

For a bug, provide a minimal synthetic example and expected behavior. Add tests when changing budget enforcement, provenance, learning targets, verification, or memory updates. Network tests must be mocked by default; make paid live tests explicit and opt-in.

## Report evidence precisely

- Distinguish code that exists from an architecture proposal.
- Keep training, validation, and evaluation worlds separate.
- Publish per-seed results, hyperparameters, and costs, including failures.
- Do not describe lexical distance as semantic novelty or model confidence as verification.
- Do not claim an internal DeepSeek integration without a working patch and measured results.
- Check rights and licenses before adding datasets, model code, or weights.
- Do not contribute private conversations, credentials, or proprietary data.

## Pull requests

Keep the change scoped, explain its behavior, and include validation commands. Update the relevant limitation when a contribution actually removes it. Unless explicitly stated otherwise and agreed by maintainers, contributions are made under the repository's MIT license.

Be respectful, critique claims with evidence, and make room for contributors with different backgrounds. Maintainers may close abusive or unrelated discussions.
