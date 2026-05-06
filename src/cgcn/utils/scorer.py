"""
Score predictions with gold labels using precision, recall and F1.
"""

from collections import Counter

NO_RELATION = "Other"

def score(key, prediction, verbose=False):
    correct_by_relation = Counter()
    guessed_by_relation = Counter()
    gold_by_relation    = Counter()

    for row in range(len(key)):
        gold = key[row]
        guess = prediction[row]

        if gold == NO_RELATION and guess == NO_RELATION:
            pass
        elif gold == NO_RELATION and guess != NO_RELATION:
            guessed_by_relation[guess] += 1
        elif gold != NO_RELATION and guess == NO_RELATION:
            gold_by_relation[gold] += 1
        elif gold != NO_RELATION and guess != NO_RELATION:
            guessed_by_relation[guess] += 1
            gold_by_relation[gold] += 1
            if gold == guess:
                correct_by_relation[guess] += 1

    if verbose:
        print("Per-relation statistics:")
        relations = gold_by_relation.keys()
        longest_relation = max((len(r) for r in relations), default=0)
        for relation in sorted(relations):
            correct = correct_by_relation[relation]
            guessed = guessed_by_relation[relation]
            gold    = gold_by_relation[relation]
            prec = float(correct) / float(guessed) if guessed > 0 else 1.0
            recall = float(correct) / float(gold) if gold > 0 else 0.0
            f1 = 2.0 * prec * recall / (prec + recall) if prec + recall > 0 else 0.0
            print(("{:<" + str(longest_relation) + "}  P: {:.2%}  R: {:.2%}  F1: {:.2%}  #: {:d}").format(
                relation, prec, recall, f1, gold))
        print("")

    prec_micro = 1.0
    if sum(guessed_by_relation.values()) > 0:
        prec_micro = float(sum(correct_by_relation.values())) / float(sum(guessed_by_relation.values()))
    recall_micro = 0.0
    if sum(gold_by_relation.values()) > 0:
        recall_micro = float(sum(correct_by_relation.values())) / float(sum(gold_by_relation.values()))
    f1_micro = 0.0
    if prec_micro + recall_micro > 0.0:
        f1_micro = 2.0 * prec_micro * recall_micro / (prec_micro + recall_micro)

    if verbose:
        print("Precision (micro): {:.3%}".format(prec_micro))
        print("   Recall (micro): {:.3%}".format(recall_micro))
        print("       F1 (micro): {:.3%}".format(f1_micro))
    return prec_micro, recall_micro, f1_micro
