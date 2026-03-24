"""
features.py
===========
Feature extraction matching the paper's NB baseline description:
"local context of 2 words only — words in the sentential context only"

We extract only the 2 words immediately before and after each entity,
plus the entity word(s) themselves. Everything else is discarded.
"""

import re
from sklearn.feature_extraction.text import CountVectorizer


def extract_local_context(raw_sentence, window=2):
    """
    Extract a 2-word window around each entity from a tagged sentence.

    Collect the *positions* covered by both windows into an
    ordered set, so overlapping words between close entities are never
    duplicated.

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

    if not e1_positions or not e2_positions:
        return ' '.join(words)  # fallback if parsing failed

    e1_start, e1_end = e1_positions[0], e1_positions[-1]
    e2_start, e2_end = e2_positions[0], e2_positions[-1]

    # Collect all word positions covered by either window into a sorted set.
    # A set automatically removes duplicates. We then sort to preserve order.
    e1_window_positions = set(range(
        max(0, e1_start - window),
        min(len(words), e1_end + window + 1)
    ))
    e2_window_positions = set(range(
        max(0, e2_start - window),
        min(len(words), e2_end + window + 1)
    ))

    # Union of both windows, sorted so we read words left-to-right
    all_positions = sorted(e1_window_positions | e2_window_positions)

    # Read out the words at those positions
    context_words = [words[i] for i in all_positions]

    return ' '.join(context_words)


def create_vectorizer():
    return CountVectorizer(
        ngram_range=(1, 2),
        token_pattern=r'\b[a-zA-Z][a-zA-Z0-9]*\b'
    )