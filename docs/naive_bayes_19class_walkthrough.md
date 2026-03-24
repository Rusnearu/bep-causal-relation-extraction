## Why the official Perl scorer?

The paper's score is computed by the **official SemEval scorer**, not by sklearn.
The scorer has specific rules:
- Evaluates on exactly the 9 directed relations (18 classes), NOT including `Other`.
- Computes macro-F1 as a simple unweighted average of the 18 per-class F1 scores.

If we used sklearn's `f1_score(average='macro')`, we'd get a slightly different
number because sklearn would include `Other` in the average. The Perl scorer is
the authoritative tool.

---

## Data flow summary

```
TRAIN_FILE.TXT
    → load_train_data()                     [naive_bayes/data_loader.py]
        → load_semeval_train()              [shared/data_loader.py]
            → _parse_semeval_blocks()       [shared/data_loader.py]
            → 8000 example dicts + labels   (19-class, label_mode='full')
        → extract_local_context() ×8000     [naive_bayes/features.py]
        → 8000 feature strings              e.g. "an arrayed configuration of antenna elements ."
    → CountVectorizer.fit_transform()       [sklearn, inside Pipeline]
    → sparse matrix (8000 × 48053)
    → MultinomialNB.fit()                   [sklearn]
    → trained model

TEST_FILE.TXT
    → load_test_data()                      [naive_bayes/data_loader.py]
        → _parse_semeval_lines()            [naive_bayes/data_loader.py]
        → 2717 example dicts                (no labels)
        → extract_local_context() ×2717     [naive_bayes/features.py]
        → 2717 feature strings
    → CountVectorizer.transform()           (uses vocabulary from training)
    → sparse matrix (2717 × 48053)
    → MultinomialNB.predict()
    → 2717 predicted labels

TEST_FILE_FULL.TXT
    → save_key_for_scorer()                 [naive_bayes/data_loader.py]
    → answer_key.txt

predictions + answer_key.txt
    → semeval2010_task8_format_checker.pl   (validates format)
    → semeval2010_task8_scorer-v1.2.pl      (computes macro F1 ≈ 56%)
    → scorer_output.txt
```
