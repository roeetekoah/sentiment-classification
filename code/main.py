"""
Exercise 5 – Sentiment Analysis
=============================================
Fill in all functions/methods marked with  # TODO
Do NOT change function signatures unless you document the change in your README.
"""

import os
import pickle
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# Optional import: the one-hot pipeline, the model classes' shapes, and all the pure
# helpers work without HuggingFace, so importing `main` (and collecting the test suite)
# must not require `transformers`. Transformer-only entry points call _require_transformers().
try:
    from transformers import AutoTokenizer, AutoModel
except ImportError:                                   # pragma: no cover
    AutoTokenizer = AutoModel = None


def _require_transformers():
    if AutoTokenizer is None or AutoModel is None:
        raise ImportError("This feature requires the 'transformers' package "
                          "(pip install transformers).")


import data_loader
from data_loader import get_negated_polarity_examples, get_rare_words_examples

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

SEQ_LEN = 52

ONEHOT_AVERAGE       = "onehot_average"
TRANSFORMER_AVERAGE  = "transformer_average"
TRANSFORMER_SEQUENCE = "transformer_sequence"

TRAIN = "train"
VAL   = "val"
TEST  = "test"

TRANSFORMER_MODEL_NAME = "distilroberta-base"


# ──────────────────────────────────────────────────────────────────────────────
# Device helper  (provided)
# ──────────────────────────────────────────────────────────────────────────────

def get_available_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=42):
    """Seed Python, NumPy and PyTorch (CPU+CUDA) for repeatability. Note this gives
    repeatability, not bit-exact determinism across hardware/cuDNN versions."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ──────────────────────────────────────────────────────────────────────────────
# Pickle helpers  (provided)
# ──────────────────────────────────────────────────────────────────────────────

def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


# ──────────────────────────────────────────────────────────────────────────────
# Section 7 – Transformer embeddings loader
# ──────────────────────────────────────────────────────────────────────────────

def load_transformer_embeddings(cache_path="transformer_emb_cache.pkl"):
    """
    Loads distilroberta-base and extracts the token embedding matrix.
    Caches the matrix locally to avoid reloading.

    Returns
    -------
    tokenizer        : HuggingFace tokenizer
    embedding_matrix : np.ndarray of shape (vocab_size, hidden_dim)
    embedding_dim    : int
    """
    _require_transformers()
    # The tokenizer is small and comes from the local HF cache, so it is always
    # re-created; only the 50265x768 matrix is worth pickling.
    tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_MODEL_NAME)

    if os.path.exists(cache_path):
        embedding_matrix = load_pickle(cache_path)
        return tokenizer, embedding_matrix, embedding_matrix.shape[1]

    model = AutoModel.from_pretrained(TRANSFORMER_MODEL_NAME)
    embedding_matrix = (model.embeddings.word_embeddings.weight
                        .detach().cpu().numpy().astype(np.float32))
    save_pickle(embedding_matrix, cache_path)
    del model  # ~330MB; no reason to keep the backbone alive for a look-up table

    return tokenizer, embedding_matrix, embedding_matrix.shape[1]


# ──────────────────────────────────────────────────────────────────────────────
# One-hot helpers  (Section 6)
# ──────────────────────────────────────────────────────────────────────────────

def get_word_to_ind(words_list):
    """
    Returns a dict mapping each word in words_list to a unique integer index.
    :param words_list: list of words
    :return: dict {word: index}
    """
    return {word: ind for ind, word in enumerate(dict.fromkeys(words_list))}


def get_one_hot(size, ind):
    """
    Returns a one-hot numpy vector of length `size` with 1 at position `ind`.
    :param size: length of the vector
    :param ind: index of the 1 entry
    :return: np.ndarray of shape (size,)
    """
    vector = np.zeros(size)
    vector[ind] = 1

    return vector

def average_one_hots(sent, word_to_ind):
    """
    Returns the average one-hot embedding for the tokens in `sent`.
    Words not in word_to_ind are ignored (not added to the average).
    My implementation is O(T + V) and avoids summing huge sparse vectors
    :param sent: Sentence object (sent.text is a list of strings)
    :param word_to_ind: dict {word: index}
    :return: np.ndarray of shape (vocab_size,)
    """
    # Which indices are known word; e.g., are in the supplied word_to_ind.
    rec_inds = np.fromiter(
        (word_to_ind[word] for word in sent.text if word in word_to_ind),
        dtype=np.intp
    )

    # Safety to avoid division by 0
    if rec_inds.size == 0:
        return np.zeros(len(word_to_ind), dtype=float)

    counts = np.bincount(rec_inds, minlength=len(word_to_ind))

    return counts / rec_inds.size

# ──────────────────────────────────────────────────────────────────────────────
# Transformer-embedding helpers  (Sections 7 & 8)
# ──────────────────────────────────────────────────────────────────────────────

def get_transformer_average(sent, tokenizer, embedding_matrix, embedding_dim):
    """
    Tokenizes `sent` and returns the average of its Transformer token embeddings.
    Special tokens (e.g. <s>, </s>) are excluded (add_special_tokens=False).
    A token id outside the embedding matrix's row range contributes a ZERO vector and is
    still counted in the denominator (per the handout), so the result is the sum of the
    in-range token embeddings divided by the total number of tokens. For a tokenizer and
    matrix from the same checkpoint every id is in range, so this equals the plain mean.

    :param sent:             Sentence object
    :param tokenizer:        HuggingFace tokenizer
    :param embedding_matrix: np.ndarray (vocab_size, embedding_dim)
    :param embedding_dim:    int
    :return: np.ndarray of shape (embedding_dim,)
    """
    # Follows the handout literally: keep one conceptual vector per token; a token whose
    # id is out of the embedding matrix's range contributes a ZERO vector (not skipped),
    # and we divide by the TOTAL token count. Since the tokenizer and matrix come from the
    # same checkpoint every id is in range, so this equals the mean of the valid rows --
    # but it is the literal specification and is covered by a unit test.
    token_ids = np.asarray(
        tokenizer.encode(" ".join(sent.text), add_special_tokens=False),
        dtype=np.int64,
    )
    if token_ids.size == 0:
        return np.zeros(embedding_dim, dtype=embedding_matrix.dtype)

    vecs = np.zeros((token_ids.size, embedding_dim), dtype=embedding_matrix.dtype)
    in_range = (token_ids >= 0) & (token_ids < embedding_matrix.shape[0])
    vecs[in_range] = embedding_matrix[token_ids[in_range]]   # out-of-range rows stay zero
    return vecs.mean(axis=0)                                  # divide by TOTAL token count


def sentence_to_embedding(sent, tokenizer, embedding_matrix, seq_len, embedding_dim):
    """
    Maps `sent` to a fixed-size matrix of Transformer token embeddings.
    - Sentences longer than seq_len are truncated (keep the first seq_len tokens).
    - Sentences shorter than seq_len are padded with zero-vectors.
    Special tokens are excluded from the encoding.

    :param sent:             Sentence object
    :param tokenizer:        HuggingFace tokenizer
    :param embedding_matrix: np.ndarray (vocab_size, embedding_dim)
    :param seq_len:          int – target sequence length
    :param embedding_dim:    int
    :return: np.ndarray of shape (seq_len, embedding_dim)
    """
    token_ids = np.asarray(
        tokenizer.encode(
            " ".join(sent.text),
            add_special_tokens=False,
        )[:seq_len],
        dtype=np.int64,          # explicit integer dtype for token ids (matches the average helper)
    )

    result = np.zeros(
        (seq_len, embedding_dim),
        dtype=embedding_matrix.dtype,
    )

    valid = (
            (token_ids >= 0)
            & (token_ids < embedding_matrix.shape[0])
    )

    result[np.flatnonzero(valid)] = embedding_matrix[token_ids[valid]]

    return result


# ──────────────────────────────────────────────────────────────────────────────
# Dataset & DataManager  (provided)
# ──────────────────────────────────────────────────────────────────────────────

class OnlineDataset(Dataset):
    """PyTorch Dataset that converts sentences on-the-fly."""

    def __init__(self, sent_data, sent_func, sent_func_kwargs):
        self.data = sent_data
        self.sent_func = sent_func
        self.sent_func_kwargs = sent_func_kwargs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sent  = self.data[idx]
        emb   = self.sent_func(sent, **self.sent_func_kwargs)
        label = sent.sentiment_class
        return torch.tensor(emb, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)


class DataManager:
    """
    Handles all data management.
    data_type: one of ONEHOT_AVERAGE, TRANSFORMER_AVERAGE, TRANSFORMER_SEQUENCE
    """

    def __init__(self, data_type=ONEHOT_AVERAGE, use_sub_phrases=True,
                 dataset_path="stanfordSentimentTreebank", batch_size=64,
                 embedding_dim=None, tokenizer=None, embedding_matrix=None,
                 max_train_samples=None):

        self.sentiment_dataset = data_loader.SentimentTreeBank(dataset_path, split_words=True)

        self.sentences = {}
        if use_sub_phrases:
            self.sentences[TRAIN] = self.sentiment_dataset.get_train_set_phrases()
        else:
            self.sentences[TRAIN] = self.sentiment_dataset.get_train_set()
        self.sentences[VAL]  = self.sentiment_dataset.get_validation_set()
        self.sentences[TEST] = self.sentiment_dataset.get_test_set()

        # API deviation (documented in README): the handout (2.4, 9) refers to a
        # max_train_samples argument that the provided skeleton does not define.
        # Draws a fixed-seed random subset of the training set for reproducibility.
        if max_train_samples is not None and max_train_samples < len(self.sentences[TRAIN]):
            self.sentences[TRAIN] = random.Random(42).sample(
                self.sentences[TRAIN], max_train_samples)

        words_list = list(self.sentiment_dataset.get_word_counts().keys())

        if data_type == ONEHOT_AVERAGE:
            self.sent_func = average_one_hots
            self.sent_func_kwargs = {"word_to_ind": get_word_to_ind(words_list)}

        elif data_type == TRANSFORMER_AVERAGE:
            if tokenizer is None or embedding_matrix is None:
                raise ValueError("tokenizer and embedding_matrix are required for this data_type")
            self.sent_func = get_transformer_average
            self.sent_func_kwargs = {
                "tokenizer":        tokenizer,
                "embedding_matrix": embedding_matrix,
                "embedding_dim":    embedding_dim,
            }

        elif data_type == TRANSFORMER_SEQUENCE:
            if tokenizer is None or embedding_matrix is None:
                raise ValueError("tokenizer and embedding_matrix are required for this data_type")
            self.sent_func = sentence_to_embedding
            self.sent_func_kwargs = {
                "tokenizer":        tokenizer,
                "embedding_matrix": embedding_matrix,
                "seq_len":          SEQ_LEN,
                "embedding_dim":    embedding_dim,
            }
        else:
            raise ValueError(f"Unknown data_type: {data_type}")

        self.torch_datasets  = {k: OnlineDataset(sents, self.sent_func, self.sent_func_kwargs)
                                 for k, sents in self.sentences.items()}
        self.torch_iterators = {k: DataLoader(ds, batch_size=batch_size, shuffle=(k == TRAIN))
                                 for k, ds in self.torch_datasets.items()}

    def get_torch_iterator(self, data_subset=TRAIN):
        return self.torch_iterators[data_subset]

    def get_labels(self, data_subset=TRAIN):
        return np.array([s.sentiment_class for s in self.sentences[data_subset]])

    def get_input_shape(self):
        return self.torch_datasets[TRAIN][0][0].shape


# ──────────────────────────────────────────────────────────────────────────────
# Models
# ──────────────────────────────────────────────────────────────────────────────

class LogLinear(nn.Module):
    """
    Log-linear model for sentiment analysis.
    Used for both the one-hot model (Section 6) and the Transformer-embedding model (Section 7).
    forward() must return LOGITS (no sigmoid).
    predict() applies sigmoid and returns probabilities.
    """

    def __init__(self, embedding_dim):
        super().__init__()
        
        self.linear = nn.Linear(embedding_dim, 1)

    def forward(self, x):
        return self.linear(x).squeeze(-1)

    def predict(self, x):
        return torch.sigmoid(self(x))   # go through nn.Module.__call__ (runs hooks)


class LSTM(nn.Module):
    """
    Bidirectional LSTM for sentiment analysis (Section 8).
    Input  : (batch, seq_len, embedding_dim)
    Output : logits of shape (batch,)
    forward() must return LOGITS (no sigmoid).
    predict() applies sigmoid and returns probabilities.
    """

    def __init__(self, embedding_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            bidirectional=True,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(2 * hidden_dim, 1)

    def forward(self, text):
        # h_n: (num_layers * num_directions, batch, hidden_dim)
        _, (h_n, _) = self.lstm(text)
        last_fwd = h_n[-2]     # final layer, forward direction  -> (batch, hidden)
        last_bwd = h_n[-1]     # final layer, backward direction -> (batch, hidden)
        h = torch.cat((last_fwd, last_bwd), dim=1)   # (batch, 2*hidden)
        h = self.dropout(h)
        return self.linear(h).squeeze(-1)            # (batch,)

    def predict(self, text):
        return torch.sigmoid(self(text))   # via nn.Module.__call__


# ──────────────────────────────────────────────────────────────────────────────
# Training utilities
# ──────────────────────────────────────────────────────────────────────────────
# ── Training hyperparameters ──────────────────────────
# Log-linear models (Sections 6 & 7)
LOGLINEAR_LR           = 0.01
LOGLINEAR_N_EPOCHS     = 10
LOGLINEAR_WEIGHT_DECAY = 0.0
BATCH_SIZE             = 64      # shared by all three DataManager-based models

# bi-LSTM (Section 8)
LSTM_LR           = 0.001
LSTM_N_EPOCHS     = 4
LSTM_WEIGHT_DECAY = 0.0001
LSTM_DROPOUT      = 0.5
LSTM_HIDDEN_DIM   = 100

# Fine-tuned Transformer (Section 9)
TRANSFORMER_LR            = 1e-5
TRANSFORMER_N_EPOCHS      = 2
TRANSFORMER_WEIGHT_DECAY  = 0.0
TRANSFORMER_TRAIN_SAMPLES = 2500

def binary_accuracy(preds, y):
    """
    Returns the fraction of correctly classified examples.

    preds: predicted PROBABILITIES in [0,1], i.e. post-sigmoid; thresholded at 0.5.
           (The original docstring said "or logits - round at 0.5", which is wrong:
           sigmoid(0) == 0.5, so the logit threshold is 0.0, not 0.5. Pass probabilities.)
    y:     true labels {0., 1.}

    Accepts torch Tensors or numpy arrays. Both are flattened to 1-D first, so a
    (B, 1) prediction cannot silently broadcast against a (B,) label into a (B, B)
    matrix of nonsense.
    :return: float
    """
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()

    if isinstance(y, torch.Tensor):
        y = y.detach().cpu().numpy()

    preds = np.asarray(preds).reshape(-1)
    y = np.asarray(y).reshape(-1)

    if preds.shape != y.shape:
        raise ValueError(f"binary_accuracy: preds {preds.shape} vs y {y.shape}")

    return float(np.mean((preds >= 0.5) == (y >= 0.5)))


def train_epoch(model, data_iterator, optimizer, criterion):
    """
    Runs one full pass over the training data.
    :param model:         a LogLinear or LSTM model
    :param data_iterator: PyTorch DataLoader for the training set
    :param optimizer:     torch optimizer
    :param criterion:     loss criterion (BCEWithLogitsLoss)
    :return: (avg_loss, avg_accuracy) over the epoch
    """
    device = get_available_device()
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for x, y in data_iterator:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)

        logits = model(x)
        loss = criterion(logits, y)

        loss.backward()
        optimizer.step()

        # Classification decisions:
        predictions = logits >= 0

        batch_size = y.numel()

        total_loss += loss.item() * batch_size
        total_correct += (
                predictions == y.bool()
        ).sum().item()

        total_examples += batch_size

    avg_loss = total_loss / total_examples
    avg_accuracy = total_correct / total_examples

    return avg_loss, avg_accuracy

def evaluate(model, data_iterator, criterion):
    """
    Evaluates the model WITHOUT updating parameters.
    Use torch.no_grad() to avoid computing unnecessary gradients.
    :param model:         a LogLinear or LSTM model
    :param data_iterator: PyTorch DataLoader (val or test)
    :param criterion:     loss criterion
    :return: (avg_loss, avg_accuracy)
    """
    device = get_available_device()
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    with torch.inference_mode():
        for x, y in data_iterator:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = criterion(logits, y)

            # sigmoid(logit) >= 0.5 is equivalent to logit >= 0
            predictions = logits >= 0

            batch_size = x.shape[0]
            total_loss += loss.item() * batch_size
            total_correct += (
                    predictions == y.bool()
            ).sum().item()
            total_examples += batch_size

    average_loss = total_loss / total_examples
    accuracy = total_correct / total_examples

    return average_loss, accuracy

def get_predictions_for_data(model, data_iter):
    """
    Collects all model predictions (probabilities) over a data iterator.
    :param model:     a LogLinear or LSTM model
    :param data_iter: PyTorch DataLoader
    :return: np.ndarray of shape (n_examples,)
    """
    device = get_available_device()
    model.eval()

    batches = []
    with torch.inference_mode():
        for x, _ in data_iter:
            x = x.to(device)
            probs = model.predict(x)                 # (batch,) probabilities in [0,1]
            batches.append(probs.detach().cpu().numpy())

    return np.concatenate(batches).reshape(-1)       # (n_examples,)

def train_model(model, data_manager, n_epochs, lr, weight_decay=0.0):
    """
    Full training loop using Adam optimizer and BCEWithLogitsLoss.
    Records per-epoch train/validation loss and accuracy.

    :param model:        LogLinear or LSTM instance
    :param data_manager: DataManager instance
    :param n_epochs:     number of epochs
    :param lr:           learning rate
    :param weight_decay: L2 regularization coefficient
    :return: dict with keys train_loss, train_acc, val_loss, val_acc, each a list of
             length n_epochs. Following the provided skeleton, train_loss/train_acc are the
             running averages over the epoch's optimization pass (as returned by
             train_epoch), while val_loss/val_acc are a post-epoch evaluation. This is the
             conventional training-curve convention: the train curve shows the within-epoch
             learning trajectory, so a model that saturates early (underfitting) or whose
             validation later diverges (overfitting) is visible.
    """
    device = get_available_device()
    model.to(device)   # train_epoch/evaluate move the DATA to device; the MODEL is moved here

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, n_epochs + 1):
        train_loss, train_acc = train_epoch(
            model, data_manager.get_torch_iterator(TRAIN), optimizer, criterion)
        val_loss, val_acc = evaluate(
            model, data_manager.get_torch_iterator(VAL), criterion)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"epoch {epoch:2d}/{n_epochs} | "
              f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
              f"val loss {val_loss:.4f} acc {val_acc:.4f}")

    return history


# ──────────────────────────────────────────────────────────────────────────────
# Plotting helper  (provided – feel free to modify)
# ──────────────────────────────────────────────────────────────────────────────

def plot_and_save(history, title_prefix, save_prefix):
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure()
    plt.plot(epochs, history["train_loss"], label="Train")
    plt.plot(epochs, history["val_loss"],   label="Validation")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title(f"{title_prefix} – Loss")
    plt.legend(); plt.tight_layout()
    plt.savefig(f"{save_prefix}_loss.png")
    plt.close()

    plt.figure()
    plt.plot(epochs, history["train_acc"], label="Train")
    plt.plot(epochs, history["val_acc"],   label="Validation")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy")
    plt.title(f"{title_prefix} – Accuracy")
    plt.legend(); plt.tight_layout()
    plt.savefig(f"{save_prefix}_acc.png")
    plt.close()


def evaluate_special_subsets(model, data_manager, *, test_probs=None, test_labels=None,
                             test_sentences=None, subset_indices=None):
    """
    Prints AND returns accuracy on the negated-polarity and rare-words subsets.
    (Modified from the provided version to also return the values, so the report
    can consume them; behaviour is otherwise identical.)

    The optional keyword arguments let an experiment reuse predictions and labels
    collected during its single test-set pass. Omitting them preserves the original
    ``evaluate_special_subsets(model, data_manager)`` behaviour.
    :return: dict {"negated_polarity": (acc, n), "rare_words": (acc, n)}
    """
    dataset = data_manager.sentiment_dataset
    test_sents = (data_manager.sentences[TEST]
                  if test_sentences is None else test_sentences)

    if subset_indices is None:
        subset_indices = {
            "negated_polarity": get_negated_polarity_examples(test_sents),
            "rare_words": get_rare_words_examples(test_sents, dataset),
        }

    if test_probs is None:
        test_probs = get_predictions_for_data(
            model, data_manager.get_torch_iterator(TEST))
    if test_labels is None:
        test_labels = data_manager.get_labels(TEST)

    all_preds = np.asarray(test_probs).reshape(-1)
    all_labels = np.asarray(test_labels).reshape(-1)

    results = {}
    for key, name in [("negated_polarity", "Negated polarity"),
                      ("rare_words", "Rare words")]:
        indices = subset_indices[key]
        if len(indices) == 0:
            print(f"  {name}: no examples found")
            results[key] = (float("nan"), 0)
            continue
        acc = float(np.mean(
            (all_preds[indices] >= 0.5).astype(float) == all_labels[indices]
        ))
        print(f"  {name} accuracy ({len(indices)} examples): {acc:.4f}")
        results[key] = (acc, len(indices))
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Shared experiment wrap-up: test evaluation, plots, subsets, results file
# ──────────────────────────────────────────────────────────────────────────────

def _test_metrics_single_pass(model, data_iter, criterion):
    """
    One forward pass over the test set that returns everything the report needs:
    (probs, labels, loss, accuracy). Loss (BCE-with-logits), accuracy (logit >= 0),
    and the saved probabilities (sigmoid) are all derived from the SAME logits, so they
    cannot disagree and the test set is traversed once instead of twice. Numeric
    definitions are identical to `evaluate` + `get_predictions_for_data`.
    """
    device = get_available_device()
    model.eval()
    probs_batches, label_batches = [], []
    total_loss, total_examples = 0.0, 0
    with torch.inference_mode():
        for x, y in data_iter:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            total_loss += criterion(logits, y).item() * x.shape[0]
            total_examples += x.shape[0]
            probs_batches.append(torch.sigmoid(logits).detach().cpu().numpy())
            label_batches.append(y.detach().cpu().numpy())

    probs = np.concatenate(probs_batches).reshape(-1)
    labels = np.concatenate(label_batches).reshape(-1)
    loss = total_loss / total_examples
    acc = float(((probs >= 0.5) == (labels >= 0.5)).mean())
    return probs, labels, loss, acc


def _run_metadata(model, seed, hyperparams):
    """Provenance stamped into every results pickle: seed, hyperparameters, trainable
    parameter count, device, and library versions -- so a saved run is reproducible and
    self-describing without re-reading the training code."""
    meta = {
        "seed":            seed,
        "hyperparams":     dict(hyperparams or {}),
        "trainable_params": int(sum(p.numel() for p in model.parameters()
                                    if p.requires_grad)),
        "device":          str(get_available_device()),
        "torch_version":   torch.__version__,
        "numpy_version":   np.__version__,
    }
    if AutoModel is not None:                      # transformers is optional at import time
        import transformers
        meta["transformers_version"] = transformers.__version__
    return meta


def _finish_experiment(model, data_manager, history, title, save_prefix,
                       seed=None, hyperparams=None):
    """
    Everything every experiment reports after training: test loss/accuracy,
    the two special-subset accuracies, the two plots, and a results pickle
    (so the report can be built without re-running training).
    """
    os.makedirs(os.path.dirname(save_prefix) or ".", exist_ok=True)
    plot_and_save(history, title, save_prefix)

    criterion = nn.BCEWithLogitsLoss()
    # Single pass: loss, accuracy, and the saved per-example probabilities all come from
    # the same forward pass (the TEST loader is not shuffled, so order is dataset order).
    test_probs, test_labels, test_loss, test_acc = _test_metrics_single_pass(
        model, data_manager.get_torch_iterator(TEST), criterion)
    print(f"TEST | loss {test_loss:.4f} acc {test_acc:.4f}")
    test_sentences = data_manager.sentences[TEST]
    subset_indices = {
        "negated_polarity": get_negated_polarity_examples(test_sentences),
        "rare_words": get_rare_words_examples(
            test_sentences, data_manager.sentiment_dataset),
    }
    subsets = evaluate_special_subsets(
        model, data_manager,
        test_probs=test_probs,
        test_labels=test_labels,
        test_sentences=test_sentences,
        subset_indices=subset_indices,
    )

    results = {
        "history":     history,
        "test_loss":   test_loss,
        "test_acc":    test_acc,
        "subsets":     subsets,
        "test_probs":  test_probs,
        "test_labels": test_labels,
        "meta":        _run_metadata(model, seed, hyperparams),
    }
    save_pickle(results, f"{save_prefix}_results.pkl")
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Section 6 – Log-linear with one-hot
# ──────────────────────────────────────────────────────────────────────────────

def train_log_linear_with_one_hot(dataset_path="stanfordSentimentTreebank", seed=42):
    """
    Creates all objects needed for training/evaluation of the one-hot log-linear model
    and runs the training process.
    Hyperparameters: lr=0.01, n_epochs=10, batch_size=64
    (Documented signature change: added `seed` with the required default of 42.)
    """
    set_seed(seed)
    dm = DataManager(data_type=ONEHOT_AVERAGE, batch_size=BATCH_SIZE,
                     dataset_path=dataset_path)
    model = LogLinear(embedding_dim=dm.get_input_shape()[0])

    history = train_model(model, dm, n_epochs=LOGLINEAR_N_EPOCHS,
                          lr=LOGLINEAR_LR, weight_decay=LOGLINEAR_WEIGHT_DECAY)
    return _finish_experiment(model, dm, history,
                              "Log-linear (one-hot)", "figs/loglinear_onehot",
                              seed=seed,
                              hyperparams={"lr": LOGLINEAR_LR, "n_epochs": LOGLINEAR_N_EPOCHS,
                                           "batch_size": BATCH_SIZE,
                                           "weight_decay": LOGLINEAR_WEIGHT_DECAY})


# ──────────────────────────────────────────────────────────────────────────────
# Section 7 – Log-linear with Transformer embeddings
# ──────────────────────────────────────────────────────────────────────────────

def train_log_linear_with_transformer(dataset_path="stanfordSentimentTreebank", seed=42):
    """
    Creates all objects needed for training/evaluation of the Transformer-embedding
    log-linear model and runs the training process.
    Hyperparameters: lr=0.01, n_epochs=10, batch_size=64
    (Documented signature change: added `seed` with the required default of 42.)
    """
    set_seed(seed)
    tokenizer, embedding_matrix, embedding_dim = load_transformer_embeddings()
    dm = DataManager(data_type=TRANSFORMER_AVERAGE, batch_size=BATCH_SIZE,
                     dataset_path=dataset_path, embedding_dim=embedding_dim,
                     tokenizer=tokenizer, embedding_matrix=embedding_matrix)
    model = LogLinear(embedding_dim=embedding_dim)

    history = train_model(model, dm, n_epochs=LOGLINEAR_N_EPOCHS,
                          lr=LOGLINEAR_LR, weight_decay=LOGLINEAR_WEIGHT_DECAY)
    return _finish_experiment(model, dm, history,
                              "Log-linear (Transformer embeddings)",
                              "figs/loglinear_transformer",
                              seed=seed,
                              hyperparams={"lr": LOGLINEAR_LR, "n_epochs": LOGLINEAR_N_EPOCHS,
                                           "batch_size": BATCH_SIZE,
                                           "weight_decay": LOGLINEAR_WEIGHT_DECAY})


# ──────────────────────────────────────────────────────────────────────────────
# Section 8 – Bi-LSTM with Transformer embeddings
# ──────────────────────────────────────────────────────────────────────────────

def train_lstm_with_transformer(dataset_path="stanfordSentimentTreebank", seed=42):
    """
    Creates all objects needed for training/evaluation of the bi-LSTM model
    and runs the training process.
    Hyperparameters: lr=0.001, weight_decay=0.0001, dropout=0.5, n_epochs=4, batch_size=64
    (Documented signature change: added `seed` with the required default of 42.)
    """
    set_seed(seed)
    tokenizer, embedding_matrix, embedding_dim = load_transformer_embeddings()
    dm = DataManager(data_type=TRANSFORMER_SEQUENCE, batch_size=BATCH_SIZE,
                     dataset_path=dataset_path, embedding_dim=embedding_dim,
                     tokenizer=tokenizer, embedding_matrix=embedding_matrix)
    model = LSTM(embedding_dim=embedding_dim, hidden_dim=LSTM_HIDDEN_DIM,
                 n_layers=1, dropout=LSTM_DROPOUT)

    history = train_model(model, dm, n_epochs=LSTM_N_EPOCHS,
                          lr=LSTM_LR, weight_decay=LSTM_WEIGHT_DECAY)
    return _finish_experiment(model, dm, history,
                              "bi-LSTM (Transformer embeddings)", "figs/lstm",
                              seed=seed,
                              hyperparams={"lr": LSTM_LR, "n_epochs": LSTM_N_EPOCHS,
                                           "batch_size": BATCH_SIZE, "dropout": LSTM_DROPOUT,
                                           "hidden_dim": LSTM_HIDDEN_DIM,
                                           "weight_decay": LSTM_WEIGHT_DECAY})


# ══════════════════════════════════════════════════════════════════════════════
# Section 9 – Fine-Tuning a Transformer Model
# ══════════════════════════════════════════════════════════════════════════════

class TransformerSentimentDataset(Dataset):
    """
    Encodes each sentence with the Transformer tokenizer on the fly.
    __getitem__ should return (input_ids, attention_mask, label).
    """

    MAX_LENGTH = 64

    def __init__(self, sentences, tokenizer, max_length=None):
        self.sentences  = sentences
        self.tokenizer  = tokenizer
        self.max_length = max_length or self.MAX_LENGTH

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sent = self.sentences[idx]
        enc = self.tokenizer(
            " ".join(sent.text),
            truncation=True,
            max_length=self.max_length,
            padding="max_length",   # fixed length -> default collate can stack batches
        )
        # NOTE: unlike Sections 7-8, special tokens are KEPT here -- the classifier
        # reads the final hidden state at position 0, which IS the <s> ([CLS]) token.
        return (torch.tensor(enc["input_ids"], dtype=torch.long),
                torch.tensor(enc["attention_mask"], dtype=torch.long),
                torch.tensor(sent.sentiment_class, dtype=torch.float32))


class TransformerSentimentClassifier(nn.Module):
    """
    distilroberta-base base model (no HF classification head) +
    a single Linear layer on top of the [CLS] token's final hidden state.
    forward() returns logits (no sigmoid).
    predict() returns probabilities (after sigmoid).
    """

    def __init__(self, model_name=TRANSFORMER_MODEL_NAME):
        super().__init__()
        _require_transformers()
        self.transformer = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.transformer.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        out = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        cls_hidden = out.last_hidden_state[:, 0]     # <s> is RoBERTa's [CLS], position 0
        return self.classifier(cls_hidden).squeeze(-1)

    def predict(self, input_ids, attention_mask):
        return torch.sigmoid(self(input_ids, attention_mask))   # via nn.Module.__call__


def _transformer_pass(model, loader, criterion, device, optimizer=None):
    """
    One pass over `loader`. With an optimizer: a training epoch; without: evaluation.
    The Section-6 train_epoch/evaluate cannot be reused here because batches are
    (input_ids, attention_mask, label) 3-tuples, not (embedding, label) pairs.
    :return: (avg_loss, avg_accuracy)
    """
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss, total_correct, total_examples = 0.0, 0, 0
    with torch.enable_grad() if training else torch.no_grad():
        for input_ids, attention_mask, y in loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            y = y.to(device)

            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, y)
            if training:
                loss.backward()
                optimizer.step()

            n = y.numel()
            total_loss += loss.item() * n
            total_correct += ((logits >= 0) == y.bool()).sum().item()
            total_examples += n

    return total_loss / total_examples, total_correct / total_examples


def _transformer_test_single_pass(model, loader, criterion, device):
    """One pass over the (unshuffled) test loader returning (probs, labels, loss, acc);
    loss, accuracy and probabilities share the same logits (no second traversal)."""
    model.eval()
    probs_batches, label_batches = [], []
    total_loss, total_examples = 0.0, 0
    with torch.inference_mode():
        for input_ids, attention_mask, y in loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            y = y.to(device)
            logits = model(input_ids, attention_mask)
            total_loss += criterion(logits, y).item() * y.numel()
            total_examples += y.numel()
            probs_batches.append(torch.sigmoid(logits).detach().cpu().numpy())
            label_batches.append(y.detach().cpu().numpy())
    probs = np.concatenate(probs_batches).reshape(-1)
    labels = np.concatenate(label_batches).reshape(-1)
    return probs, labels, total_loss / total_examples, \
        float(((probs >= 0.5) == (labels >= 0.5)).mean())


def train_transformer(dataset_path="stanfordSentimentTreebank",
                      batch_size=64, n_epochs=2, lr=1e-5, weight_decay=0.0, seed=42):
    """
    Full training procedure for the fine-tuned Transformer.
    Use BCEWithLogitsLoss and Adam. No HuggingFace Trainer.
    Report train/val loss and accuracy per epoch, test loss/accuracy,
    and accuracy on the two special subsets.
    (Documented signature change: added `seed` with the required default of 42.)
    """
    _require_transformers()
    set_seed(seed)
    device = get_available_device()
    tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_MODEL_NAME)

    # Full sentences only (no sub-phrases), subsampled to 2500 via DataManager's
    # max_train_samples, per the handout. The DataManager's one-hot machinery is
    # lazy and never touched; we only use its sentence splits here.
    dm = DataManager(data_type=ONEHOT_AVERAGE, use_sub_phrases=False,
                     dataset_path=dataset_path, batch_size=batch_size,
                     max_train_samples=TRANSFORMER_TRAIN_SAMPLES)

    loaders = {
        split: DataLoader(
            TransformerSentimentDataset(dm.sentences[split], tokenizer),
            batch_size=batch_size, shuffle=(split == TRAIN))
        for split in (TRAIN, VAL, TEST)
    }
    model = TransformerSentimentClassifier().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    # Same convention as train_model: train_loss/train_acc are the running average over the
    # epoch's optimization pass; val is a post-epoch evaluation.
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    for epoch in range(1, n_epochs + 1):
        train_loss, train_acc = _transformer_pass(model, loaders[TRAIN], criterion,
                                                  device, optimizer)
        val_loss, val_acc = _transformer_pass(model, loaders[VAL], criterion, device)
        for k, v in zip(("train_loss", "train_acc", "val_loss", "val_acc"),
                        (train_loss, train_acc, val_loss, val_acc)):
            history[k].append(v)
        print(f"epoch {epoch:2d}/{n_epochs} | "
              f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
              f"val loss {val_loss:.4f} acc {val_acc:.4f}")

    os.makedirs("figs", exist_ok=True)
    plot_and_save(history, "Fine-tuned Transformer", "figs/transformer")

    # Single pass over the test set: loss, accuracy and per-example probabilities all
    # come from the same logits (the test loader preserves dataset order; shuffle=False).
    all_preds, all_labels, test_loss, test_acc = _transformer_test_single_pass(
        model, loaders[TEST], criterion, device)
    print(f"TEST | loss {test_loss:.4f} acc {test_acc:.4f}")

    # Special subsets: indices are into the test list; reuse the stored one-pass outputs.
    test_sents = dm.sentences[TEST]
    subset_indices = {
        "negated_polarity": get_negated_polarity_examples(test_sents),
        "rare_words": get_rare_words_examples(test_sents, dm.sentiment_dataset),
    }
    subsets = evaluate_special_subsets(
        model, dm,
        test_probs=all_preds,
        test_labels=all_labels,
        test_sentences=test_sents,
        subset_indices=subset_indices,
    )

    # Per-example test predictions + labels (dataset order; test loader unshuffled).
    # These are exactly the arrays returned by `_transformer_test_single_pass` above.
    results = {"history": history, "test_loss": test_loss,
               "test_acc": test_acc, "subsets": subsets,
               "test_probs": all_preds, "test_labels": all_labels,
               "meta": _run_metadata(model, seed,
                                     {"lr": lr, "n_epochs": n_epochs,
                                      "batch_size": batch_size,
                                      "weight_decay": weight_decay,
                                      "train_samples": TRANSFORMER_TRAIN_SAMPLES})}
    save_pickle(results, "figs/transformer_results.pkl")
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Entry point (kept at the very bottom so every Section-9 definition above is
# in scope; also required on Windows for DataLoader with num_workers > 0)
# ──────────────────────────────────────────────────────────────────────────────

def main():
    import argparse
    runners = {
        "onehot":              train_log_linear_with_one_hot,
        "transformer-average": train_log_linear_with_transformer,
        "lstm":                train_lstm_with_transformer,
        "transformer":         train_transformer,
    }
    p = argparse.ArgumentParser(description="Ex5 sentiment models")
    p.add_argument("--model", choices=list(runners) + ["all"], default="all",
                   help="which model to train (default: all)")
    p.add_argument("--data-path", default="stanfordSentimentTreebank",
                   help="path to the extracted treebank (default: ./stanfordSentimentTreebank)")
    args = p.parse_args()
    todo = list(runners) if args.model == "all" else [args.model]
    for name in todo:
        print(f"\n=== {name} ===", flush=True)
        runners[name](dataset_path=args.data_path)


if __name__ == "__main__":
    main()
