"""
Naive-Bayes-specific data loading.

Parsing is delegated entirely to src/shared/data_loader.py.

"""

import re
import os


def save_key_for_scorer(full_filepath, output_filepath):
    """
    Reads TEST_FILE_FULL.TXT and writes a clean  ID<TAB>LABEL  file
    that the official Perl scorer accepts.

    Used only for the 19-class reproduction.
    """
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)

    with open(full_filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    key = {}
    for block in content.strip().split('\n\n'):
        lines = block.strip().split('\n')
        if len(lines) < 2:
            continue
        id_match = re.match(r'^(\d+)\s+', lines[0])
        if not id_match:
            continue
        sent_id = int(id_match.group(1))
        label = lines[1].strip().replace('\r', '')
        key[sent_id] = label

    with open(output_filepath, 'w', encoding='utf-8', newline='') as f:
        for sent_id in sorted(key.keys()):
            f.write(f'{sent_id}\t{key[sent_id]}\n')

    return output_filepath


def load_train_data(filepath, label_mode='full'):
    """
    Loads training data and returns (feature_strings, labels, examples).

    Parameters
    ----------
    filepath   : path to TRAIN_FILE.TXT
    label_mode : 'full' (default, 19-class) or '3class'
                 Passed straight through to the shared loader.
    """
    from src.shared.data_loader import load_semeval_train
    from src.naive_bayes.features import extract_local_context

    examples = load_semeval_train(filepath, label_mode=label_mode)
    feature_strings = [extract_local_context(ex['sentence'], window=2) for ex in examples]
    labels = [ex['label'] for ex in examples]
    return feature_strings, labels, examples


def _parse_semeval_lines(filepath: str):
    """
    Parses the compact test file where each line is one example
    (no blank-line separators, no labels):
        <ID>  "<sentence>"
    """
    examples = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            id_match = re.match(r'^(\d+)\s+', line)
            if not id_match:
                continue
            sent_id = int(id_match.group(1))

            sentence_match = re.search(r'"(.+)"', line)
            if not sentence_match:
                continue
            raw_sentence = sentence_match.group(1)

            e1_match = re.search(r'<e1>(.*?)</e1>', raw_sentence)
            e2_match = re.search(r'<e2>(.*?)</e2>', raw_sentence)

            examples.append({
                'id':       sent_id,
                'sentence': raw_sentence,
                'e1':       e1_match.group(1) if e1_match else '',
                'e2':       e2_match.group(1) if e2_match else '',
                'label':    None,
            })
    return examples


def load_test_data(filepath):
    """
    Loads the compact test file (no labels) and returns
    (feature_strings, examples).
    """
    from src.naive_bayes.features import extract_local_context

    examples = _parse_semeval_lines(filepath)
    feature_strings = [extract_local_context(ex['sentence'], window=2) for ex in examples]
    return feature_strings, examples
