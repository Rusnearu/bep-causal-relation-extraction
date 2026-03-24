# Naive Bayes 3-Class Baseline — Full Explanation

## 2. Do we train from scratch or from the 19-class model?

**Completely from scratch, totally separate.**

The 19-class model (in `notebooks/reproduction/`) is a reproduction experiment —
it exists only to verify that our code gets ~56% macro F1, matching the original
paper. It is a dead end after that.

The 3-class model starts fresh:
- Different label set (3 classes, not 19).
- Different training data (same sentences, but all non-causal labels collapsed to `Other`).
- Different evaluation metric.
- Different code path (`src/shared/` instead of `src/naive_bayes/`).

There is no transfer of weights, no fine-tuning. Naive Bayes has no concept of
pre-trained weights anyway — it just counts word frequencies.

---

**F1 for a single class** = harmonic mean of precision and recall:
```
Precision = TP / (TP + FP)   — of all examples we called class X, how many were really X?
Recall    = TP / (TP + FN)   — of all real class X examples, how many did we find?
F1        = 2 × (P × R) / (P + R)
```

**Macro** = compute F1 separately for CE(e1,e2) and CE(e2,e1), then average them
with equal weight. This means the model must be good at **both** directions.

**Why exclude Other?**
- `Other` is 87.5% of the data. If you include it in the average, a model that just
  predicts "Other" for everything gets very high F1 on Other, which swamps the metric.
- We care specifically about whether the model can detect causality and its direction.
  Other is not interesting to us.
- This mirrors SemEval tradition exactly.

#### Secondary metrics also computed:

```python
macro_f1_all3 = f1_score(y_true, y_pred, labels=LABEL_ORDER, average='macro')
micro_f1      = f1_score(y_true, y_pred, labels=LABEL_ORDER, average='micro')
acc           = accuracy_score(y_true, y_pred)
cm            = confusion_matrix(y_true, y_pred, labels=LABEL_ORDER)
```

- **Macro F1 all 3**: same calculation but includes Other. Always higher than the
  primary metric because Other is easy.
- **Micro F1**: weighted by support (number of examples per class). Dominated by
  the majority class, approximately equals accuracy here. Not the main metric.
- **Accuracy**: overall % correct. Misleading with imbalanced classes (a model
  predicting "Other" always gets ~87.5% accuracy).
- **Confusion matrix**: 3×3 grid, rows = true labels, columns = predicted. The
  off-diagonal cells on the top-left 2×2 sub-matrix (CE(e1,e2) vs CE(e2,e1)) are
  particularly revealing — they show how often the model confuses the direction.

---

For all three models, `src/shared/data_loader.py` returns the same raw sentence
dicts with `<e1>/<e2>` tags intact. Each model's own code (`src/naive_bayes/features.py`,
`src/r_bert/features.py`, `src/sdp_lstm/features.py`) applies its own preprocessing
on top.

All three models then call the same `src/shared/evaluation.py` `evaluate()` function
with the same metric, so results are directly comparable.

---

## Data flow summary

```
TRAIN_FILE.TXT
    → load_semeval_train()               [shared/data_loader.py]
        → _parse_semeval_blocks()        [shared/data_loader.py]
        → _map_to_3class() ×8000         [shared/data_loader.py]
        → 8000 example dicts + labels    (3-class: CE(e1,e2), CE(e2,e1), Other)
    → extract_local_context() ×8000      [naive_bayes/features.py]
    → 8000 feature strings               e.g. "the chronic inflammation in the helicobacter"
    → CountVectorizer.fit_transform()    [sklearn, inside Pipeline]
    → sparse matrix (8000 × 48053)
    → MultinomialNB.fit()                [sklearn]
    → trained model

TEST_FILE_FULL.TXT
    → load_semeval_test_with_labels()    [shared/data_loader.py]
        → _parse_semeval_blocks()        [shared/data_loader.py]
        → _map_to_3class() ×2717         [shared/data_loader.py]
        → 2717 example dicts + labels    (3-class)
    → extract_local_context() ×2717      [naive_bayes/features.py]
    → 2717 feature strings
    → CountVectorizer.transform()        (uses vocabulary from training)
    → sparse matrix (2717 × 48053)
    → MultinomialNB.predict()
    → 2717 predicted labels

y_true + y_pred
    → evaluate()                         [shared/evaluation.py]
    → primary: macro F1 over CE(e1,e2) and CE(e2,e1) only
    → per-class P / R / F1, overall metrics, confusion matrix
    → report.txt + metrics.json + confusion_matrix.png  → results/naive_bayes/semeval2010/
```
