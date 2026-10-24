# Sentiment Classification: Representations and Adaptation

How much does the representation matter, and what changes when the language model is fine-tuned? This study compares four sentiment classifiers: a one-hot log-linear model, a classifier over frozen DistilRoBERTa token embeddings, a BiLSTM over those embeddings, and end-to-end Transformer fine-tuning.

Joint coursework by **Roee Tekoah and Yona Gan**, with experiments on binary sentiment from the Stanford Sentiment Treebank. The [eight-page study](docs/sentiment-study.pdf) covers representations, training behavior, negation and rare-word subsets, parameter counts, and computational cost.

## Recorded results

| Model | Test accuracy | Trainable parameters |
|---|---:|---:|
| One-hot log-linear | 82.85% | 16,272 |
| Frozen-embedding log-linear | 82.43% | 769 |
| BiLSTM | 84.93% | 696,201 |
| Fine-tuned Transformer | 88.05% | 82,119,169 |

These are recorded seed-42 results on 962 test sentences. The first three models train on 73,570 full-sentence/phrase examples; fine-tuning uses 2,500 full sentences. This is an exploratory comparison with different training budgets, rather than a controlled estimate of architecture alone. Negation and rare-word subsets contain only 62 and 50 examples. [Methods and limitations](docs/methods.md) · [machine-readable run record](results/recorded-run.json).

## Run

Python 3.11 or later:

```console
python -m pip install -r requirements.txt
python -m pytest -q
python code/main.py --model onehot --data-path /path/to/stanfordSentimentTreebank
```

The synthetic tests run without downloading a model or corpus. Optional integrations require separately acquired data or model files. Other model choices are `transformer-average`, `lstm`, and `transformer`; Transformer routes download `distilroberta-base` when it is not cached. Training writes local figures and result files under `figs/`.

The dataset, weights, embedding caches, and assignment handout are not bundled. [Reproduction and data format](docs/reproduction.md) · [attribution and provenance](docs/provenance.md).
