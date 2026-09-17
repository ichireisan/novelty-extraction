# Foundations and sources

This project combines established ideas; it does not claim that value of information, memory selection, or useful information originated here.

## Information and computation

- Tishby, Pereira, and Bialek. **The Information Bottleneck Method.** Relevant information is defined relative to a target. [Paper](https://arxiv.org/abs/physics/0004057).
- Xu, Zhao, Song, Stewart, and Ermon. **A Theory of Usable Information Under Computational Constraints.** Predictive information depends on the observer's capabilities. [Paper](https://arxiv.org/abs/2002.10689).
- Callaway et al. **Learning to Select Computations.** Learned allocation of computation under metareasoning objectives. [Paper](https://arxiv.org/abs/1711.06892).
- Schmidhuber. **Driven by Compression Progress.** Learning progress distinguishes discovering learnable structure from persistent unpredictability. [Paper](https://arxiv.org/abs/0812.4360).
- Grünwald and Vitányi. **Shannon Information and Kolmogorov Complexity.** Distinguishes conditional complexity and shared algorithmic information. [Paper](https://arxiv.org/abs/cs/0410002).

## Biological motivation

- Kumaran and Maguire. **An Unexpected Sequence of Events: Mismatch Detection in the Human Hippocampus.** [Primary study](https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.0040424).
- Schultz, Dayan, and Montague. **A Neural Substrate of Prediction and Reward.** [Author-hosted abstract and paper](https://www.gatsby.ucl.ac.uk/~dayan/papers/sdm97.html).

These studies motivate questions and distinctions. They do not validate the repository's particular architecture as a biological model.

## DeepSeek sources

Consulted 2026-09-17:

- [Official V4.1-Flash announcement](https://www.deepseek.com/en/news/deepseek-v4-1-flash/).
- [Official model card](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash).
- [Released configuration](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/config.json).
- [Reference inference code](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/inference/model.py).

Only links and brief architectural descriptions are included. No model weights or upstream source code are copied into this project. For reproducible future model experiments, record the exact upstream revision and serving-engine version used.
