"""
data_loader.py  (src/shared/)
==============================
Shared data loading for the 3-class causal relation extraction project.

Supports:
  - SemEval-2010 Task 8 format (train / test files)
  - label_mode='3class'  → maps all 19 SemEval labels to 3 classes:
        Cause-Effect(e1,e2), Cause-Effect(e2,e1), Other
  - label_mode='full'    → keeps the original 19-class labels (for the
                           reproduction baseline in notebooks/reproduction/)

The 3-class mapping is the heart of this project:
  - Cause-Effect(e1,e2) and Cause-Effect(e2,e1) are kept as-is.
  - Every other relation (including the original 'Other') becomes 'Other'.

This module is intentionally model-agnostic: it returns raw example dicts
with the sentence text (and <e1>/<e2> tags intact) so that each model can
apply its own preprocessing / feature extraction on top.

Dataset 2 (to be added later): add a new load_<dataset>_* function here.
"""

import re

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CAUSAL_LABELS = {'Cause-Effect(e1,e2)', 'Cause-Effect(e2,e1)'}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _map_to_3class(label: str) -> str:
    """Maps any SemEval-19-class label to one of the 3 project labels."""
    return label if label in CAUSAL_LABELS else 'Other'


def _parse_semeval_blocks(filepath: str):
    """
    Parses any SemEval-style file where examples are separated by blank lines.

    Each block has the structure:
        <ID>  "<sentence with <e1>/<e2> tags>"
        <label>          (only in training / full-test files)
        Comment: ...     (only in training / full-test files)

    Returns a list of raw dicts:
        {'id': int, 'sentence': str, 'e1': str, 'e2': str,
         'label': str or None}
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    examples = []
    blocks = content.strip().split('\n\n')

    for block in blocks:
        lines = block.strip().split('\n')
        if not lines:
            continue

        # Line 0: ID + quoted sentence
        id_match = re.match(r'^(\d+)\s+', lines[0])
        if not id_match:
            continue
        sent_id = int(id_match.group(1))

        sentence_match = re.search(r'"(.+)"', lines[0])
        if not sentence_match:
            continue
        raw_sentence = sentence_match.group(1)

        e1_match = re.search(r'<e1>(.*?)</e1>', raw_sentence)
        e2_match = re.search(r'<e2>(.*?)</e2>', raw_sentence)
        e1_text = e1_match.group(1) if e1_match else ''
        e2_text = e2_match.group(1) if e2_match else ''

        # Line 1 is the label (present in train + full-test files)
        label = lines[1].strip().replace('\r', '') if len(lines) > 1 else None

        examples.append({
            'id':       sent_id,
            'sentence': raw_sentence,
            'e1':       e1_text,
            'e2':       e2_text,
            'label':    label,
        })

    return examples


# ---------------------------------------------------------------------------

def load_semeval_train(filepath: str, label_mode: str = '3class'):
    """
    Loads SemEval-2010 Task 8 training file.

    Parameters
    ----------
    filepath    : path to TRAIN_FILE.TXT
    label_mode  : '3class' (default) or 'full' (19-class)

    Returns
    -------
    list of dicts: {id, sentence, e1, e2, label}
    Labels are already mapped according to label_mode.
    """
    examples = _parse_semeval_blocks(filepath)
    if label_mode == '3class':
        for ex in examples:
            ex['label'] = _map_to_3class(ex['label'])
    return examples


def load_semeval_test_with_labels(filepath: str, label_mode: str = '3class'):
    """
    Loads the full test file (TEST_FILE_FULL.TXT — includes labels).
    Use this for evaluation.

    Returns
    -------
    list of dicts: {id, sentence, e1, e2, label}
    """
    examples = _parse_semeval_blocks(filepath)
    if label_mode == '3class':
        for ex in examples:
            ex['label'] = _map_to_3class(ex['label'])
    return examples


def get_label_distribution(examples):
    """
    Returns a sorted list of (label, count, pct) tuples for a list of examples.
    Useful for quick data exploration in notebooks.
    """
    from collections import Counter
    total = len(examples)
    counts = Counter(ex['label'] for ex in examples)
    return sorted(
        [(label, count, count / total * 100) for label, count in counts.items()],
        key=lambda x: -x[1]
    )
