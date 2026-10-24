"""
Comprehensive test suite for Exercise 5, written against the SKELETON's documented
contracts (signatures + docstrings), not against any particular implementation.

Design: the logic of every function is tested offline and deterministically using
  * a MockTokenizer / small fake embedding matrix   (transformer helpers, §9 dataset)
  * synthetic tensors and a MockDataManager          (training utilities, train_model)
so the core suite needs neither HuggingFace nor the treebank. Tests that genuinely
require the real tokenizer/model or the dataset are grouped at the end and skip
automatically when those are unavailable.

Run it yourself:
    python -m pytest test_ex5.py -v
    python test_ex5.py
"""
import os

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import main as m

DATA = os.environ.get("SENTIMENT_DATA", "stanfordSentimentTreebank")


# ════════════════════════════════════════════════════════════════════════════════
# Fixtures, stubs and mocks (keep the logic tests independent of HF / the dataset)
# ════════════════════════════════════════════════════════════════════════════════

def Sent(words, label=1.0):
    """Minimal stand-in for data_loader.Sentence: exposes .text and .sentiment_class."""
    return type("S", (), {"text": list(words), "sentiment_class": float(label)})()


class MockTokenizer:
    """Deterministic tokenizer with a tiny fixed vocab. Content ids are 5..; special
    tokens are ids 1 (cls) and 2 (sep). Supports the two calling conventions the code
    uses: .encode(text, add_special_tokens=...) and tokenizer(text, ...)."""
    cls_token_id = 1
    sep_token_id = 2
    _vocab = {"good": 5, "bad": 6, "movie": 7, "not": 8, "the": 9, "was": 10}

    def _content_ids(self, text):
        return [self._vocab.get(w, 4) for w in text.split()]   # 4 = generic/unknown-in-vocab

    def encode(self, text, add_special_tokens=True):
        ids = self._content_ids(text)
        return ([self.cls_token_id] + ids + [self.sep_token_id]) if add_special_tokens else ids

    def __call__(self, text, truncation=False, max_length=16, padding=None):
        ids = [self.cls_token_id] + self._content_ids(text) + [self.sep_token_id]
        if truncation:
            ids = ids[:max_length]
        mask = [1] * len(ids)
        if padding == "max_length":
            pad = max_length - len(ids)
            ids = ids + [0] * pad
            mask = mask + [0] * pad
        return {"input_ids": ids, "attention_mask": mask}


@pytest.fixture
def mock_emb():
    """(MockTokenizer, matrix, dim). Row i of the matrix is the constant vector i, so a
    token's contribution is trivially checkable. vocab_size 11 covers ids 0..10."""
    dim = 4
    mat = np.stack([np.full(dim, i, dtype=np.float32) for i in range(11)])
    return MockTokenizer(), mat, dim


def synthetic_vec_loader(n=48, dim=16, seed=0, shuffle=True, batch=16):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, dim, generator=g)
    y = (torch.rand(n, generator=g) > 0.5).float()
    return DataLoader(TensorDataset(X, y), batch_size=batch, shuffle=shuffle)


def synthetic_seq_loader(n=24, seq=None, dim=8, seed=0, batch=8):
    seq = seq or m.SEQ_LEN
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, seq, dim, generator=g)
    y = (torch.rand(n, generator=g) > 0.5).float()
    return DataLoader(TensorDataset(X, y), batch_size=batch, shuffle=False)


class MockDataManager:
    """Provides just the interface train_model uses: get_torch_iterator(split) and
    torch_datasets[split]. Backed by synthetic vectors, so train_model is tested with no
    dependency on the real data."""
    def __init__(self, dim=16, n=48):
        g = torch.Generator().manual_seed(1)
        X = torch.randn(n, dim, generator=g)
        y = (torch.rand(n, generator=g) > 0.5).float()
        ds = TensorDataset(X, y)
        self.torch_datasets = {m.TRAIN: ds, m.VAL: ds}
        self._it = {m.TRAIN: DataLoader(ds, batch_size=16, shuffle=True),
                    m.VAL: DataLoader(ds, batch_size=16, shuffle=False)}
        self.dim = dim

    def get_torch_iterator(self, split=m.TRAIN):
        return self._it[split]


# ════════════════════════════════════════════════════════════════════════════════
# 1.  One-hot helpers
# ════════════════════════════════════════════════════════════════════════════════

def test_get_word_to_ind_unique_indices():
    d = m.get_word_to_ind(["a", "b", "a", "c", "b", "a"])
    assert set(d) == {"a", "b", "c"}
    assert len(set(d.values())) == len(d)              # indices unique
    assert all(0 <= i < len(d) for i in d.values())    # usable to index a |V|-vector


def test_get_word_to_ind_empty_and_single():
    assert m.get_word_to_ind([]) == {}
    assert list(m.get_word_to_ind(["solo"]).values()) == [0]


def test_get_one_hot_basic_and_boundaries():
    for size, ind in [(5, 0), (5, 2), (5, 4), (1, 0)]:
        v = m.get_one_hot(size, ind)
        assert v.shape == (size,)
        assert v[ind] == 1 and v.sum() == 1


def test_average_one_hots_normalised_frequency_and_unknowns():
    w2i = {"good": 0, "bad": 1, "movie": 2}
    v = m.average_one_hots(Sent(["good", "good", "movie", "zzz"]), w2i)   # zzz unknown
    assert v.shape == (3,)
    assert np.allclose(v, [2 / 3, 0, 1 / 3]) and np.isclose(v.sum(), 1)


def test_average_one_hots_order_invariant():
    w2i = {"good": 0, "bad": 1, "not": 2}
    a = m.average_one_hots(Sent(["good", "not", "bad"]), w2i)
    b = m.average_one_hots(Sent(["bad", "good", "not"]), w2i)
    assert np.allclose(a, b)                            # bag of words


def test_average_one_hots_all_unknown_is_zero_no_crash():
    v = m.average_one_hots(Sent(["zzz", "qqq"]), {"a": 0, "b": 1})
    assert v.shape == (2,) and v.sum() == 0            # must not divide by zero


# ════════════════════════════════════════════════════════════════════════════════
# 2.  Transformer-embedding helpers (mocked -> offline)
# ════════════════════════════════════════════════════════════════════════════════

def test_transformer_average_shape_and_mean(mock_emb):
    tok, mat, dim = mock_emb
    # "good movie" -> content ids [5,7]; rows are constant 5 and 7 -> mean 6
    v = m.get_transformer_average(Sent(["good", "movie"]), tok, mat, dim)
    assert v.shape == (dim,) and np.allclose(v, 6.0)


def test_transformer_average_excludes_special_tokens(mock_emb):
    tok, mat, dim = mock_emb
    # If specials (ids 1,2 -> rows 1,2) were included, the mean would be pulled toward them.
    v = m.get_transformer_average(Sent(["good", "movie"]), tok, mat, dim)
    assert np.allclose(v, 6.0)                          # == mean of content rows only


def test_transformer_average_out_of_range_id(mock_emb):
    tok, mat, dim = mock_emb
    small = mat[:6]                                     # valid ids 0..5; id 7 now out of range
    # "good movie" -> ids [5, 7]; row 5 valid, id 7 out of range.
    v = m.get_transformer_average(Sent(["good", "movie"]), tok, small, dim)
    assert v.shape == (dim,) and np.isfinite(v).all()
    # handout: out-of-range contributes a zero vector, counted in the denominator -> 5/2
    assert np.allclose(v, 2.5)


def test_transformer_average_empty(mock_emb):
    tok, mat, dim = mock_emb
    v = m.get_transformer_average(Sent([""]), tok, mat, dim)
    assert v.shape == (dim,) and np.isfinite(v).all()


def test_sentence_to_embedding_pads_short(mock_emb):
    tok, mat, dim = mock_emb
    seq = m.sentence_to_embedding(Sent(["good", "movie"]), tok, mat, 6, dim)
    assert seq.shape == (6, dim)
    assert np.allclose(seq[0], 5) and np.allclose(seq[1], 7)   # first two = token rows
    assert np.all(seq[2:] == 0)                                # rest zero-padded


def test_sentence_to_embedding_truncates_long(mock_emb):
    tok, mat, dim = mock_emb
    words = ["good", "bad", "movie", "not"]             # 4 content tokens
    seq = m.sentence_to_embedding(Sent(words), tok, mat, 2, dim)
    assert seq.shape == (2, dim)
    assert np.allclose(seq[0], 5) and np.allclose(seq[1], 6)   # first two only
    assert not np.any(np.all(seq == 0, axis=1))                # no padding rows


def test_sentence_to_embedding_exact_length(mock_emb):
    tok, mat, dim = mock_emb
    seq = m.sentence_to_embedding(Sent(["good", "bad"]), tok, mat, 2, dim)
    assert seq.shape == (2, dim) and not np.any(np.all(seq == 0, axis=1))


# ════════════════════════════════════════════════════════════════════════════════
# 3.  Models
# ════════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("batch", [1, 8])
def test_loglinear_forward_shape_and_predict(batch):
    model = m.LogLinear(50)
    logits = model(torch.randn(batch, 50))
    assert logits.shape == (batch,)
    p = model.predict(torch.randn(batch, 50))
    assert p.shape == (batch,) and (0 <= p).all() and (p <= 1).all()


def test_loglinear_predict_is_sigmoid_of_forward():
    model = m.LogLinear(30)
    x = torch.randn(5, 30)
    assert torch.allclose(model.predict(x), torch.sigmoid(model(x)), atol=1e-6)


def test_loglinear_param_count_is_d_plus_1():
    assert sum(p.numel() for p in m.LogLinear(768).parameters()) == 769


@pytest.mark.parametrize("batch", [1, 8])
def test_lstm_forward_shape(batch):
    model = m.LSTM(32, 16, 1, 0.5)
    assert model(torch.randn(batch, m.SEQ_LEN, 32)).shape == (batch,)


def test_lstm_is_bidirectional_and_concatenates():
    model = m.LSTM(32, 16, 1, 0.5)
    names = [n for n, _ in model.named_parameters()]
    assert any("reverse" in n for n in names)          # bidirectional
    assert model.linear.in_features == 32              # 2 * hidden(16)


def test_lstm_predict_is_sigmoid_and_dropout_gated():
    model = m.LSTM(32, 16, 1, 0.5)
    x = torch.randn(4, m.SEQ_LEN, 32)
    model.eval()
    assert torch.allclose(model.predict(x), torch.sigmoid(model(x)), atol=1e-6)
    assert torch.allclose(model(x), model(x))          # dropout OFF in eval
    model.train()
    assert not torch.allclose(model(x), model(x))      # dropout ON in train


# ════════════════════════════════════════════════════════════════════════════════
# 4.  binary_accuracy
# ════════════════════════════════════════════════════════════════════════════════

def test_binary_accuracy_all_right_wrong_half():
    y = np.array([1., 0., 1., 0.])
    assert m.binary_accuracy(np.array([0.9, 0.1, 0.8, 0.2]), y) == 1.0
    assert m.binary_accuracy(np.array([0.1, 0.9, 0.2, 0.8]), y) == 0.0
    assert m.binary_accuracy(np.array([0.9, 0.9, 0.9, 0.9]), y) == 0.5


def test_binary_accuracy_threshold_at_half():
    # 0.5 -> positive; 0.6 -> positive; 0.4 -> negative
    assert m.binary_accuracy(np.array([0.5, 0.6, 0.4]), np.array([1., 1., 0.])) == 1.0


def test_binary_accuracy_returns_float():
    assert isinstance(m.binary_accuracy(np.array([0.9]), np.array([1.])), float)


def test_binary_accuracy_no_silent_broadcast():
    assert m.binary_accuracy(torch.full((4, 1), 0.9), torch.tensor([1., 1., 1., 0.])) == 0.75


# ════════════════════════════════════════════════════════════════════════════════
# 5.  Training utilities (synthetic data -> offline)
# ════════════════════════════════════════════════════════════════════════════════

def test_train_epoch_returns_floats_and_updates_params():
    # train_epoch/evaluate move the DATA to device and assume the MODEL is already there
    # (train_model moves it); replicate that here.
    model = m.LogLinear(16).to(m.get_available_device())
    before = [p.clone() for p in model.parameters()]
    loss, acc = m.train_epoch(model, synthetic_vec_loader(dim=16),
                              torch.optim.Adam(model.parameters(), lr=0.05),
                              nn.BCEWithLogitsLoss())
    assert isinstance(loss, float) and isinstance(acc, float)
    assert np.isfinite(loss) and 0.0 <= acc <= 1.0
    assert any(not torch.allclose(a, b) for a, b in zip(before, model.parameters()))


def test_evaluate_does_not_update_params_and_is_eval_mode():
    model = m.LSTM(8, 8, 1, 0.5).to(m.get_available_device())   # has dropout
    loader = synthetic_seq_loader(dim=8)
    crit = nn.BCEWithLogitsLoss()
    before = [p.clone() for p in model.parameters()]
    l1, a1 = m.evaluate(model, loader, crit)
    l2, a2 = m.evaluate(model, loader, crit)
    assert all(torch.allclose(a, b) for a, b in zip(before, model.parameters()))  # no update
    assert (l1, a1) == (l2, a2)                         # deterministic -> eval mode (dropout off)
    assert np.isfinite(l1) and 0.0 <= a1 <= 1.0


def test_get_predictions_for_data_shape_range_and_order():
    dev = m.get_available_device()
    model = m.LogLinear(16).to(dev)
    loader = synthetic_vec_loader(n=40, dim=16, shuffle=False)
    p = m.get_predictions_for_data(model, loader)
    assert isinstance(p, np.ndarray) and p.shape == (40,)
    assert (0 <= p).all() and (p <= 1).all()
    # order preserved: predictions equal predict() on the same (unshuffled) inputs
    X = torch.cat([x for x, _ in loader]).to(dev)
    assert np.allclose(p, model.predict(X).detach().cpu().numpy(), atol=1e-6)


def test_single_pass_test_metrics_match_two_pass():
    """The one-pass test evaluation used in _finish_experiment must return exactly the
    loss/accuracy of `evaluate` and the probabilities of `get_predictions_for_data`
    (guards against the refactor that collapsed two test traversals into one)."""
    dev = m.get_available_device()
    model = m.LogLinear(16).to(dev)
    crit = nn.BCEWithLogitsLoss()
    l_loader = synthetic_vec_loader(n=40, dim=16, shuffle=False)
    p_loader = synthetic_vec_loader(n=40, dim=16, shuffle=False)
    s_loader = synthetic_vec_loader(n=40, dim=16, shuffle=False)
    exp_loss, exp_acc = m.evaluate(model, l_loader, crit)
    exp_probs = m.get_predictions_for_data(model, p_loader)
    probs, labels, loss, acc = m._test_metrics_single_pass(model, s_loader, crit)
    assert probs.shape == (40,) and labels.shape == (40,)
    assert np.allclose(probs, exp_probs, atol=1e-6)
    assert abs(loss - exp_loss) < 1e-5 and abs(acc - exp_acc) < 1e-9


def test_finish_experiment_traverses_test_loader_once(monkeypatch, tmp_path):
    """The shared final path must reuse its stored test probabilities and labels for
    subset metrics instead of iterating over the TEST loader a second time."""
    x = torch.tensor([[1., 0.], [0., 1.], [-1., 0.], [0., -1.]])
    y = torch.tensor([1., 1., 0., 0.])

    class CountingLoader:
        def __init__(self):
            self.traversals = 0

        def __iter__(self):
            self.traversals += 1
            return iter([(x[:2], y[:2]), (x[2:], y[2:])])

    class FinishDataManager:
        def __init__(self):
            self.test_loader = CountingLoader()
            self.label_reads = 0
            self.sentiment_dataset = object()
            self.sentences = {
                m.TEST: [Sent([f"word-{i}"], label) for i, label in enumerate(y)]
            }

        def get_torch_iterator(self, split=m.TRAIN):
            assert split == m.TEST
            return self.test_loader

        def get_labels(self, split=m.TRAIN):
            assert split == m.TEST
            self.label_reads += 1
            return y.numpy()

    dm = FinishDataManager()
    model = m.LogLinear(2).to(m.get_available_device())
    with torch.no_grad():
        model.linear.weight.copy_(
            torch.tensor([[2., -2.]], device=m.get_available_device()))
        model.linear.bias.zero_()

    monkeypatch.setattr(m, "plot_and_save", lambda *args, **kwargs: None)
    monkeypatch.setattr(m, "save_pickle", lambda *args, **kwargs: None)
    monkeypatch.setattr(m, "get_negated_polarity_examples", lambda sents: [0, 2])
    monkeypatch.setattr(
        m, "get_rare_words_examples", lambda sents, dataset: [1, 3])

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    results = m._finish_experiment(
        model, dm, history, "test", str(tmp_path / "finish"),
        seed=42, hyperparams={},
    )

    expected_logits = torch.tensor([2., -2., -2., 2.])
    expected_probs = torch.sigmoid(expected_logits).numpy()
    expected_loss = nn.BCEWithLogitsLoss()(expected_logits, y).item()

    assert dm.test_loader.traversals == 1
    assert dm.label_reads == 0
    assert np.allclose(results["test_probs"], expected_probs, atol=1e-6)
    assert np.array_equal(results["test_labels"], y.numpy())
    assert results["test_loss"] == pytest.approx(expected_loss)
    assert results["test_acc"] == 0.5
    assert results["subsets"]["negated_polarity"] == (1.0, 2)
    assert results["subsets"]["rare_words"] == (0.0, 2)


# ════════════════════════════════════════════════════════════════════════════════
# 6.  train_model  (the full training flow, mocked data manager)
# ════════════════════════════════════════════════════════════════════════════════

def test_train_model_history_keys_and_length():
    dm = MockDataManager(dim=16)
    h = m.train_model(m.LogLinear(dm.dim), dm, n_epochs=3, lr=0.05)
    for key in ("train_loss", "train_acc", "val_loss", "val_acc"):   # skeleton-required keys
        assert key in h and len(h[key]) == 3
        assert all(np.isfinite(v) for v in h[key])


def test_train_model_reduces_training_loss():
    dm = MockDataManager(dim=16)
    h = m.train_model(m.LogLinear(dm.dim), dm, n_epochs=8, lr=0.1)
    assert h["train_loss"][-1] < h["train_loss"][0]


def test_train_model_records_train_epoch_online_average(monkeypatch):
    """Skeleton contract: train_model records exactly what train_epoch returns -- the
    running average over the epoch's optimization pass -- as train_loss/train_acc, not a
    separate post-epoch re-evaluation of the train set."""
    calls = {"n": 0}

    def fake_train_epoch(model, data_iter, optimizer, criterion):
        calls["n"] += 1
        return 0.4242, 0.7777                       # sentinel (online loss, acc)

    monkeypatch.setattr(m, "train_epoch", fake_train_epoch)
    dm = MockDataManager(dim=16)
    h = m.train_model(m.LogLinear(dm.dim), dm, n_epochs=3, lr=0.05)
    assert calls["n"] == 3                          # one online pass per epoch
    assert h["train_loss"] == [0.4242, 0.4242, 0.4242]
    assert h["train_acc"] == [0.7777, 0.7777, 0.7777]
    assert "train_acc_online" not in h              # no separate online/post split anymore


# ════════════════════════════════════════════════════════════════════════════════
# 7.  Section 9 dataset (mocked tokenizer -> offline)
# ════════════════════════════════════════════════════════════════════════════════

def test_transformer_dataset_returns_typed_triple_and_keeps_specials():
    tok = MockTokenizer()
    ds = m.TransformerSentimentDataset([Sent(["good", "movie"], 1.0)], tok, max_length=8)
    ids, mask, label = ds[0]
    assert ids.dtype == torch.long and mask.dtype == torch.long
    assert ids.shape == (8,) and mask.shape == (8,) and label.dtype == torch.float32
    assert ids[0].item() == tok.cls_token_id            # §9 KEEPS special tokens
    assert label.item() == 1.0


def test_transformer_dataset_len():
    ds = m.TransformerSentimentDataset([Sent(["a"]), Sent(["b"])], MockTokenizer())
    assert len(ds) == 2


def test_transformer_dataset_attention_mask_contents():
    """Real and special tokens get mask 1; padded positions get mask 0, and the number
    of 1s equals the real token count (2 content + 2 special here)."""
    tok = MockTokenizer()
    ds = m.TransformerSentimentDataset([Sent(["good", "movie"])], tok, max_length=10)
    ids, mask, _ = ds[0]
    n_real = 4                                          # <cls> good movie <sep>
    assert mask[:n_real].tolist() == [1, 1, 1, 1]
    assert mask[n_real:].sum().item() == 0             # padded positions are 0
    assert int(mask.sum().item()) == n_real
    assert (ids[mask == 0] == 0).all()                 # padded ids are the pad id (0)


def test_transformer_average_empty_mock_is_zero_no_nan(mock_emb):
    """A sentence that tokenizes to no tokens yields a zero vector, not NaN."""
    tok, mat, dim = mock_emb
    v = m.get_transformer_average(Sent([""]), tok, mat, dim)
    assert v.shape == (dim,) and np.all(v == 0) and np.isfinite(v).all()


def test_transformer_classifier_shape_and_backbone(real_emb):
    """(B,L) ids/mask -> (B,) logits (incl. batch 1); the backbone is trainable."""
    clf = m.TransformerSentimentClassifier()
    for B in (1, 3):
        ids = torch.randint(0, 100, (B, 8))
        mask = torch.ones(B, 8, dtype=torch.long)
        assert clf(ids, mask).shape == (B,)
        p = clf.predict(ids, mask)
        assert p.shape == (B,) and (0 <= p).all() and (p <= 1).all()
    assert clf.classifier.out_features == 1
    assert all(p.requires_grad for p in clf.transformer.parameters())   # backbone trainable


# ════════════════════════════════════════════════════════════════════════════════
# 8.  Provided helpers
# ════════════════════════════════════════════════════════════════════════════════

def test_pickle_roundtrip(tmp_path):
    p = tmp_path / "x.pkl"
    obj = {"a": np.arange(3), "b": [1, 2]}
    m.save_pickle(obj, str(p))
    back = m.load_pickle(str(p))
    assert np.array_equal(back["a"], obj["a"]) and back["b"] == obj["b"]


def test_get_available_device():
    assert isinstance(m.get_available_device(), torch.device)


def test_binary_accuracy_matches_logit_threshold():
    """The loops threshold logits at 0; predict()+binary_accuracy threshold probabilities
    at 0.5. Since sigmoid(0)=0.5 they must agree."""
    logits = torch.tensor([2.0, -1.0, 0.5, -3.0])
    y = torch.tensor([1., 0., 1., 1.])
    inline = ((logits >= 0) == y.bool()).float().mean().item()
    assert inline == m.binary_accuracy(torch.sigmoid(logits), y)


# ════════════════════════════════════════════════════════════════════════════════
# 9.  Real-tokenizer / dataset integration (skips if unavailable)
# ════════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def real_emb():
    if os.environ.get("SENTIMENT_INTEGRATION") != "1":
        pytest.skip("Set SENTIMENT_INTEGRATION=1 to use an independently acquired model")
    try:
        return m.load_transformer_embeddings("transformer_emb_cache.pkl")
    except Exception as e:                              # pragma: no cover
        pytest.skip(f"real transformer embeddings unavailable: {e}")


def test_real_transformer_average_consistency(real_emb):
    """With the real tokenizer/matrix, get_transformer_average equals the mean of the
    non-pad rows of sentence_to_embedding when nothing is truncated."""
    tok, mat, dim = real_emb
    s = Sent(["a", "very", "good", "movie"])
    avg = m.get_transformer_average(s, tok, mat, dim)
    seq = m.sentence_to_embedding(s, tok, mat, m.SEQ_LEN, dim)
    n = int((seq != 0).any(1).sum())
    assert np.allclose(seq[:n].mean(0), avg, atol=1e-5)


def test_real_dataset_special_tokens_kept(real_emb):
    tok, _, _ = real_emb
    ds = m.TransformerSentimentDataset([Sent(["good", "film"])], tok, max_length=16)
    ids, _, _ = ds[0]
    assert ids[0].item() == tok.cls_token_id


@pytest.mark.skipif(not os.path.isdir(DATA), reason="treebank folder not found")
def test_full_onehot_pipeline_end_to_end():
    """DataManager -> LogLinear -> train_model -> get_predictions_for_data ->
    binary_accuracy, on a small slice, exercising the whole flow together."""
    m.set_seed(0)
    dm = m.DataManager(data_type=m.ONEHOT_AVERAGE, batch_size=32, dataset_path=DATA)
    for k in (m.TRAIN, m.VAL, m.TEST):
        dm.sentences[k] = dm.sentences[k][:96]
        dm.torch_datasets[k] = m.OnlineDataset(dm.sentences[k], dm.sent_func, dm.sent_func_kwargs)
        dm.torch_iterators[k] = DataLoader(dm.torch_datasets[k], batch_size=32,
                                           shuffle=(k == m.TRAIN))
    model = m.LogLinear(dm.get_input_shape()[0])
    h = m.train_model(model, dm, n_epochs=3, lr=0.01)
    assert len(h["val_acc"]) == 3
    preds = m.get_predictions_for_data(model, dm.get_torch_iterator(m.TEST))
    labels = dm.get_labels(m.TEST)
    acc = m.binary_accuracy(preds, labels)
    assert 0.0 <= acc <= 1.0 and preds.shape == labels.shape


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
