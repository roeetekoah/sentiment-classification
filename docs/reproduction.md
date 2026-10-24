# Reproduction

Install `requirements.txt` and run commands from the repository root. Obtain the Stanford Sentiment Treebank separately. The preserved course loader expects `SOStr.txt`, `STree.txt`, `dictionary.txt`, and `sentiment_labels.txt` in the directory passed to `--data-path`.

```console
python code/main.py --help
python code/main.py --model onehot --data-path /path/to/stanfordSentimentTreebank
python code/main.py --model transformer-average --data-path /path/to/stanfordSentimentTreebank
python code/main.py --model lstm --data-path /path/to/stanfordSentimentTreebank
python code/main.py --model transformer --data-path /path/to/stanfordSentimentTreebank
```

Each runner uses seed 42. A GPU is useful for fine-tuning. Third-party dependencies and pretrained weights are obtained separately under their own terms. Exact GPU numerical repetition can depend on hardware and library versions.

`python -m pytest -q` checks model contracts with synthetic tensors and a mock tokenizer. To enable the external integration checks, set `SENTIMENT_INTEGRATION=1` after acquiring the model and set `SENTIMENT_DATA` to the treebank directory. The tests explicitly skip unavailable integrations.

For the optional forward-only cost experiment, run `python code/benchmark.py --device cuda` (or `--device cpu`). See the recorded benchmark's method note before interpreting its timings.
