# Methods and limits

Sentiment values at most 0.4 are negative; values at least 0.6 are positive; neutral examples are discarded. The supplied treebank loader produces 7,693 full training sentences, 962 validation sentences, and 962 test sentences. Phrase augmentation supplies 73,570 training examples for the first three models. Fine-tuning samples 2,500 full training sentences with seed 42 and runs for two epochs.

The frozen representation is the pretrained token-embedding matrix, rather than a contextual encoder forward pass. The averaged classifier is order-insensitive; the BiLSTM processes length-52 sequences, including trailing zero padding. Fine-tuning uses the Transformer encoder and a binary classification head. A single final test pass supplies saved probabilities reused for subset analysis.

The one-hot vocabulary follows the supplied corpus-wide vocabulary construction. No repeated-seed uncertainty estimate is available. The subset analyses are descriptive and small. Different augmentation and training budgets prevent a clean architecture-only causal comparison. The retained study explains the full procedures and limitations.

Recorded forward timings use synthetic preprocessed inputs and a batch of 64 on an RTX 4060 Ti. They exclude tokenization, data transfer, preprocessing, loading, and training. They are amortized throughput measurements, not single-sentence latency. The benchmark does not evaluate trained sentiment quality.

The source's report-verification script was rerun during release and confirmed canonical file hashes, saved predictions, metrics, subset scores, configuration, and report constants. Training was not rerun for publication. Public JSON preserves histories and metadata; private source pickles and the dataset remain outside this repository.
