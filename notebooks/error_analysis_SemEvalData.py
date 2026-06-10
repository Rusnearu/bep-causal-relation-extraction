"""
Error analysis for the 3-class causal relation extraction on SemEval dataset.

Loads saved predictions from all three models (Naive Bayes, R-BERT, C-GCN)
and the gold test labels, then calcualtes:

  1. Error counts
  2. Error overlap across models
  3. Sentences all three models get wrong
  4. Direction confusion  
  5. Explicit causal marker analysis
  6. Sentence length analysis
  7. Unique failures — sentences only ONE model gets wrong

Run from the repo root:
    python notebooks/error_analysis_SemEvalData.py
"""

import os
import re
import sys
from collections import Counter, defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from shared.data_loader import load_semeval_test_with_labels

TEST_FILE   = os.path.join(ROOT, "data", "semeval2010", "raw", "TEST_FILE_FULL.TXT")
RESULTS_DIR = os.path.join(ROOT, "results")

LABELS   = ["Cause-Effect(e1,e2)", "Cause-Effect(e2,e1)", "Other"]
CAUSAL   = {"Cause-Effect(e1,e2)", "Cause-Effect(e2,e1)"}
OPPOSITE = {"Cause-Effect(e1,e2)": "Cause-Effect(e2,e1)",
            "Cause-Effect(e2,e1)": "Cause-Effect(e1,e2)"}

# Causal connective words 
EXPLICIT_MARKERS = [
    "because", "cause", "caused", "causes", "causing",
    "due to", "as a result", "result in", "results in", "resulted in",
    "lead to", "leads to", "led to", "leading to",
    "therefore", "thus", "hence", "consequently", "so that",
    "trigger", "triggers", "triggered", "produce", "produces", "produced",
    "bring about", "give rise to", "stem from", "owing to",
]

# Load data
def load_predictions(model_name: str):
    path = os.path.join(RESULTS_DIR, model_name, "semeval2010", "predictions.txt")
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f]


test_examples = load_semeval_test_with_labels(TEST_FILE, label_mode="3class")
nb_preds    = load_predictions("naive_bayes")
rbert_preds = load_predictions("r_bert")
cgcn_preds  = load_predictions("cgcn")

assert len(test_examples) == len(nb_preds) == len(rbert_preds) == len(cgcn_preds), \
    "Prediction file length does not match test set size."

# Build combined records
def strip_tags(sentence: str) -> str:
    """Remove <e1>, </e1>, <e2>, </e2> tags."""
    return re.sub(r"</?e[12]>", "", sentence)


records = []
for i, ex in enumerate(test_examples):
    true = ex["label"]
    nb   = nb_preds[i]
    rb   = rbert_preds[i]
    cg   = cgcn_preds[i]

    def dir_confused(pred):
        return true in CAUSAL and pred == OPPOSITE.get(true)

    raw_sentence = strip_tags(ex["sentence"]).lower()
    has_marker   = any(m in raw_sentence for m in EXPLICIT_MARKERS)
    token_count  = len(ex["sentence"].split())

    records.append({
        "id":       ex["id"],
        "sentence": ex["sentence"],
        "e1":       ex["e1"],
        "e2":       ex["e2"],
        "true":     true,
        "nb":       nb,
        "rbert":    rb,
        "cgcn":     cg,
        "nb_wrong":    nb != true,
        "rbert_wrong": rb != true,
        "cgcn_wrong":  cg != true,
        "all_wrong":   nb != true and rb != true and cg != true,
        "nb_dir_confused":    dir_confused(nb),
        "rbert_dir_confused": dir_confused(rb),
        "cgcn_dir_confused":  dir_confused(cg),
        "has_marker": has_marker,
        "length":     token_count,
    })

causal_records = [r for r in records if r["true"] in CAUSAL]
other_records  = [r for r in records if r["true"] == "Other"]

# Helper
SEP  = "=" * 70
SEP2 = "-" * 70


def section(title):
    print(f"\n{SEP}\n{title}\n{SEP}")


def subsection(title):
    print(f"\n{SEP2}\n{title}\n{SEP2}")


def pct(n, total):
    return f"{n}/{total} ({100*n/total:.1f}%)" if total else "0/0"


def show_examples(recs, n=10, show_preds=True):
    for r in recs[:n]:
        print(f"  [{r['id']}] true={r['true']}")
        print(f"         sent: {r['sentence']}")
        print(f"         e1={r['e1']!r}  e2={r['e2']!r}")
        if show_preds:
            print(f"         NB={r['nb']}  R-BERT={r['rbert']}  C-GCN={r['cgcn']}")
        print()


# SECTION 1 — Overall error counts
section("1. OVERALL ERROR COUNTS")

total = len(records)
for name, key in [("Naive Bayes", "nb_wrong"), ("R-BERT", "rbert_wrong"), ("C-GCN", "cgcn_wrong")]:
    n_wrong = sum(1 for r in records if r[key])
    print(f"  {name:15s}  errors: {pct(n_wrong, total)}")

print()
for label in LABELS:
    subset = [r for r in records if r["true"] == label]
    n = len(subset)
    print(f"  True label: {label}  (n={n})")
    for name, key in [("NB", "nb_wrong"), ("R-BERT", "rbert_wrong"), ("C-GCN", "cgcn_wrong")]:
        w = sum(1 for r in subset if r[key])
        print(f"    {name:8s}  wrong: {pct(w, n)}")
    print()

# SECTION 2 — Error overlap
section("2. ERROR OVERLAP")

combos = {
    "NB only":            lambda r: r["nb_wrong"]    and not r["rbert_wrong"] and not r["cgcn_wrong"],
    "R-BERT only":        lambda r: r["rbert_wrong"]  and not r["nb_wrong"]   and not r["cgcn_wrong"],
    "C-GCN only":         lambda r: r["cgcn_wrong"]   and not r["nb_wrong"]   and not r["rbert_wrong"],
    "All 3 wrong":        lambda r: r["all_wrong"],
    "All 3 correct":      lambda r: not r["nb_wrong"] and not r["rbert_wrong"] and not r["cgcn_wrong"],
}

for label, fn in combos.items():
    n = sum(1 for r in records if fn(r))
    by_class = Counter(r["true"] for r in records if fn(r))
    class_str = "  ".join(f"{k}:{v}" for k, v in sorted(by_class.items()))
    print(f"  {label:22s}  {pct(n, total):18s}  [{class_str}]")

# SECTION 3 — Sentences ALL THREE models get wrong
section("3. SENTENCES ALL THREE MODELS GET WRONG")

all_wrong = [r for r in records if r["all_wrong"]]
print(f"  Total: {len(all_wrong)} sentences\n")

for label in LABELS:
    subset = [r for r in all_wrong if r["true"] == label]
    if not subset:
        continue
    subsection(f"  True label: {label}  ({len(subset)} sentences)")

    nb_pred_counts    = Counter(r["nb"]    for r in subset)
    rbert_pred_counts = Counter(r["rbert"] for r in subset)
    cgcn_pred_counts  = Counter(r["cgcn"]  for r in subset)
    print(f"    NB predicted:     {dict(nb_pred_counts)}")
    print(f"    R-BERT predicted: {dict(rbert_pred_counts)}")
    print(f"    C-GCN predicted:  {dict(cgcn_pred_counts)}")

    all_label = [r for r in records if r["true"] == label]
    with_marker = [r for r in subset if r["has_marker"]]
    avg_len_wrong   = sum(r["length"] for r in subset) / len(subset)
    avg_len_overall = sum(r["length"] for r in all_label) / len(all_label)
    print(f"    With explicit marker: {pct(len(with_marker), len(subset))}")
    print(f"    Avg sentence length:  {avg_len_wrong:.1f} tokens (all-wrong)  vs  {avg_len_overall:.1f} tokens (all {label})")
    print()
    show_examples(subset, n=len(subset))

# SECTION 4 — Direction confusion
section("4. DIRECTION CONFUSION (right class, wrong direction)")

causal_n = len(causal_records)
for name, key in [("Naive Bayes", "nb_dir_confused"),
                  ("R-BERT",      "rbert_dir_confused"),
                  ("C-GCN",       "cgcn_dir_confused")]:
    n = sum(1 for r in causal_records if r[key])
    total_errors_causal = sum(1 for r in causal_records if r[f"{key.split('_')[0]}_wrong"])
    print(f"  {name:15s}  direction confused: {pct(n, causal_n)} of causal examples", end="")
    if total_errors_causal:
        print(f"  ({100*n/total_errors_causal:.0f}% of that model's causal errors)")
    else:
        print()

all_dir_confused = [r for r in causal_records
                    if r["nb_dir_confused"] and r["rbert_dir_confused"] and r["cgcn_dir_confused"]]
print(f"\n  All 3 models direction-confused on {len(all_dir_confused)} sentences:")
show_examples(all_dir_confused, n=10)


# SECTION 5 — Explicit causal marker analysis
section("5. EXPLICIT CAUSAL MARKER ANALYSIS")
print("  Does having an explicit causal word ('because', 'caused', 'due to', ...) help?\n")

for label in LABELS:
    subset = [r for r in records if r["true"] == label]
    with_m    = [r for r in subset if r["has_marker"]]
    without_m = [r for r in subset if not r["has_marker"]]
    if not subset:
        continue
    print(f"  True label: {label}  (total {len(subset)})")
    print(f"    With explicit marker ({len(with_m)} examples):")
    for name, key in [("NB", "nb_wrong"), ("R-BERT", "rbert_wrong"), ("C-GCN", "cgcn_wrong")]:
        w = sum(1 for r in with_m if r[key])
        print(f"      {name:8s} error rate: {pct(w, len(with_m)) if with_m else 'n/a'}")
    print(f"    Without explicit marker ({len(without_m)} examples):")
    for name, key in [("NB", "nb_wrong"), ("R-BERT", "rbert_wrong"), ("C-GCN", "cgcn_wrong")]:
        w = sum(1 for r in without_m if r[key])
        print(f"      {name:8s} error rate: {pct(w, len(without_m)) if without_m else 'n/a'}")
    print()

# SECTION 6 — Sentence length analysis
section("6. SENTENCE LENGTH ANALYSIS")

avg_len_all = sum(r["length"] for r in records) / len(records)
print(f"  Overall average sentence length: {avg_len_all:.1f} tokens\n")

def bucket(length):
    if length <= 13:  return "short  (<=13)"
    if length <= 19:  return "medium (14-19)"
    return                    "long   (>19)"

buckets = defaultdict(list)
for r in records:
    buckets[bucket(r["length"])].append(r)

for bname in ["short  (<=13)", "medium (14-19)", "long   (>19)"]:
    subset = buckets[bname]
    if not subset:
        continue
    print(f"  {bname}  ({len(subset)} sentences)")
    for name, key in [("NB", "nb_wrong"), ("R-BERT", "rbert_wrong"), ("C-GCN", "cgcn_wrong")]:
        w = sum(1 for r in subset if r[key])
        print(f"    {name:8s} error rate: {pct(w, len(subset))}")
    print()

# SECTION 7 — Unique failures: only one model is wrong
section("7. UNIQUE FAILURES — sentences only ONE model gets wrong")

for name, wrong_key, other_keys in [
    ("Naive Bayes", "nb_wrong",    ["rbert_wrong", "cgcn_wrong"]),
    ("R-BERT",      "rbert_wrong", ["nb_wrong",    "cgcn_wrong"]),
    ("C-GCN",       "cgcn_wrong",  ["nb_wrong",    "rbert_wrong"]),
]:
    unique = [r for r in records if r[wrong_key] and not any(r[k] for k in other_keys)]
    causal_unique = [r for r in unique if r["true"] in CAUSAL]
    print(f"\n  {name} uniquely wrong: {len(unique)} total, {len(causal_unique)} on causal classes")


print(f"\n{SEP}\nDone.\n")
