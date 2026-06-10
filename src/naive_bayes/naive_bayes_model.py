"""
MultinomialNB with alpha=1.0.

"""

from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


def build_naive_bayes_pipeline(vectorizer, alpha=1.0):
    """
    Chains CountVectorizer → MultinomialNB into one sklearn Pipeline.

    Parameters
    vectorizer : CountVectorizer
    alpha : Laplace smoothing parameter = 1.0
    """
    return Pipeline([
        ('bow',        vectorizer),
        ('classifier', MultinomialNB(alpha=alpha))
    ])