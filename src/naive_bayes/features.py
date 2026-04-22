"""
features.py
===========
Feature extraction: position-labeled 2-word context window around each entity.

For each entity, the words to the left are labeled e1_l:word, the entity
word(s) are labeled e1:word, and the words to the right are labeled e1_r:word.
Same for e2. All left-context words share the same position label regardless
of distance, so position is coarsely encoded in the bag-of-words.
"""

import re
from sklearn.feature_extraction.text import CountVectorizer


def extract_local_context(raw_sentence, window=2):
    """
    Extract position-labeled context window features around each entity.

    Returns a string of space-separated tokens like:
        'e1_l:arrayed e1_l:antenna e1:configuration e1_r:of e1_r:antenna
         e2_l:antenna e2_l:elements e2:elements e2_r:.'
    """
    token_pattern = re.compile(r'(<e1>|</e1>|<e2>|</e2>|[^\s<>]+)')
    raw_tokens = token_pattern.findall(raw_sentence)

    words = []
    e1_positions = []
    e2_positions = []
    inside_e1 = False
    inside_e2 = False

    for tok in raw_tokens:
        if tok == '<e1>':
            inside_e1 = True
        elif tok == '</e1>':
            inside_e1 = False
        elif tok == '<e2>':
            inside_e2 = True
        elif tok == '</e2>':
            inside_e2 = False
        else:
            idx = len(words)
            words.append(tok.lower())
            if inside_e1:
                e1_positions.append(idx)
            elif inside_e2:
                e2_positions.append(idx)

    e1_start, e1_end = e1_positions[0], e1_positions[-1]
    e2_start, e2_end = e2_positions[0], e2_positions[-1]

    features = []
    features += [f'e1_l:{words[i]}' for i in range(max(0, e1_start - window), e1_start)]
    features += [f'e1:{words[i]}' for i in e1_positions]
    features += [f'e1_r:{words[i]}' for i in range(e1_end + 1, min(len(words), e1_end + window + 1))]
    features += [f'e2_l:{words[i]}' for i in range(max(0, e2_start - window), e2_start)]
    features += [f'e2:{words[i]}' for i in e2_positions]
    features += [f'e2_r:{words[i]}' for i in range(e2_end + 1, min(len(words), e2_end + window + 1))]

    return ' '.join(features)


def create_vectorizer(ngram_range=(1, 1)):
    return CountVectorizer(
        ngram_range=ngram_range,
        token_pattern=r'[a-zA-Z0-9_]+:[a-zA-Z0-9]+' 
    )
