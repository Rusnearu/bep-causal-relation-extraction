"""
Converts SemEval-2010 Task 8 or CausalNewsCorpus to the JSON format expected by C-GCN.

This needs to be run only once locally per dataset.

Usage:
    pip install stanza
    python -c "import stanza; stanza.download('en')"
    python src/cgcn/prepare_data.py --dataset semeval
    python src/cgcn/prepare_data.py --dataset cnc
"""

import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import argparse

import stanza
from src.shared.data_loader import load_semeval_train, load_semeval_test_with_labels, load_cnc_data

DATASETS = {
    'semeval': {
        'train': 'data/semeval2010/raw/TRAIN_FILE.TXT',
        'test':  'data/semeval2010/raw/TEST_FILE_FULL.TXT',
        'out':   'data/semeval2010/cgcn',
    },
    'cnc': {
        'train': 'data/CausalNewsCorpus/train_clean.txt',
        'test':  'data/CausalNewsCorpus/dev_clean.txt',
        'out':   'data/CausalNewsCorpus/cgcn',
    },
}

# NER tag mapping: stanza tags → TACRED-compatible tags used by the GCN repo

STANZA_TO_GCN_NER = {
    'PERSON':   'PERSON',
    'ORG':      'ORGANIZATION',
    'GPE':      'LOCATION',
    'LOC':      'LOCATION',
    'FAC':      'LOCATION',
    'DATE':     'DATE',
    'TIME':     'TIME',
    'MONEY':    'MONEY',
    'PERCENT':  'PERCENT',
    'ORDINAL':  'ORDINAL',
    'CARDINAL': 'NUMBER',
    'DURATION': 'DURATION',
    'SET':      'SET',
}


def map_ner(tag):
    """Convert stanza BIO NER tag to a GCN-compatible type string."""
    if not tag or tag == 'O':
        return 'O'
    bare = tag.split('-')[-1]
    return STANZA_TO_GCN_NER.get(bare, 'MISC')

def get_entity_char_spans(sentence):
    """
    Strips <e1>/<e2> tags and returns (clean_sentence, e1_span, e2_span).
    """
    tag_positions = {
        '<e1>':  sentence.find('<e1>'),
        '</e1>': sentence.find('</e1>'),
        '<e2>':  sentence.find('<e2>'),
        '</e2>': sentence.find('</e2>'),
    }

    e1_cs = tag_positions['<e1>']  + len('<e1>')
    e1_ce = tag_positions['</e1>']
    e2_cs = tag_positions['<e2>']  + len('<e2>')
    e2_ce = tag_positions['</e2>']

    def removed_before(pos):
        return sum(len(t) for t, p in tag_positions.items() if p != -1 and p < pos)

    e1_span = (e1_cs - removed_before(e1_cs), e1_ce - removed_before(e1_ce))
    e2_span = (e2_cs - removed_before(e2_cs), e2_ce - removed_before(e2_ce))

    clean = re.sub(r'</?e[12]>', '', sentence)
    return clean, e1_span, e2_span

def token_span(words, char_start, char_end):
    """Return token indices that cover the given character span."""
    hits = [i for i, w in enumerate(words)
            if w.start_char < char_end and w.end_char > char_start]
    if not hits:
        nearest = min(range(len(words)),
                      key=lambda i: abs(words[i].start_char - char_start))
        return nearest, nearest
    return hits[0], hits[-1]

def entity_ner_type(words, tok_start, tok_end):
    """Most common non-O NER type in the entity span, or 'MISC'."""
    from collections import Counter
    types = [map_ner(getattr(w, 'ner', 'O') or 'O')
             for w in words[tok_start:tok_end + 1]]
    types = [t for t in types if t != 'O']
    return Counter(types).most_common(1)[0][0] if types else 'MISC'

def convert(examples, nlp):
    records, skipped = [], 0
    for i, ex in enumerate(examples):
        if (i + 1) % 500 == 0:
            print(f'  {i + 1}/{len(examples)}...')

        try:
            clean, e1_span, e2_span = get_entity_char_spans(ex['sentence'])
        except Exception as err:
            print(f'  [SKIP] tag parse error ({err}): {ex["sentence"][:60]}')
            skipped += 1
            continue

        doc = nlp(clean)
        if not doc.sentences:
            print(f'  [SKIP] stanza returned no sentences: {clean[:60]}')
            skipped += 1
            continue

        words = doc.sentences[0].words
        subj_s, subj_e = token_span(words, *e1_span)
        obj_s,  obj_e  = token_span(words, *e2_span)

        records.append({
            'token':           [w.text   for w in words],
            'subj_start':      subj_s,
            'subj_end':        subj_e,
            'obj_start':       obj_s,
            'obj_end':         obj_e,
            'subj_type':       entity_ner_type(words, subj_s, subj_e),
            'obj_type':        entity_ner_type(words, obj_s,  obj_e),
            'relation':        ex['label'],
            'stanford_pos':    [w.xpos   for w in words],
            'stanford_ner':    [map_ner(getattr(w, 'ner', 'O') or 'O') for w in words],
            'stanford_head':   [w.head   for w in words],   
            'stanford_deprel': [w.deprel for w in words],
        })

    if skipped:
        print(f'  Skipped {skipped} examples due to parse errors.')
    return records

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='semeval', choices=['semeval', 'cnc'],
                        help='Which dataset to prepare (semeval or cnc)')
    args = parser.parse_args()

    cfg = DATASETS[args.dataset]
    OUT_DIR = cfg['out']
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f'Loading {args.dataset} data (3-class mapping)...')
    if args.dataset == 'semeval':
        train = load_semeval_train(cfg['train'], label_mode='3class')
        test  = load_semeval_test_with_labels(cfg['test'], label_mode='3class')
    else:
        train = load_cnc_data(cfg['train'])
        test  = load_cnc_data(cfg['test'])
    print(f'  Train: {len(train)}  |  Test: {len(test)}')

    print('\nLoading stanza pipeline...')
    nlp = stanza.Pipeline(
        lang='en',
        processors='tokenize,mwt,pos,lemma,depparse,ner',
        tokenize_no_ssplit=True,
        verbose=False,
    )

    print('\nConverting train...')
    train_records = convert(train, nlp)
    print(f'  Done: {len(train_records)} records')

    print('\nConverting test...')
    test_records = convert(test, nlp)
    print(f'  Done: {len(test_records)} records')

    for split, records in [('train', train_records), ('test', test_records)]:
        out_path = os.path.join(OUT_DIR, f'{split}.json')  # OUT_DIR set above from dataset config
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(records, f)
        print(f'Saved {out_path}  ({len(records)} records)')

    print('\nAll done')


if __name__ == '__main__':
    main()
