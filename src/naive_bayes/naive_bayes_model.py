"""
naive_bayes_model.py
====================
Standard MultinomialNB with alpha=1.0 (classic Laplace smoothing).
This is what a 2010-era 'simple Naive Bayes' implementation would use.
"""

from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


def build_naive_bayes_pipeline(vectorizer, alpha=1.0):
    """
    Chains CountVectorizer → MultinomialNB into one sklearn Pipeline.

    Parameters
    ----------
    vectorizer : CountVectorizer from features.create_vectorizer()
    alpha : float, Laplace smoothing parameter (1.0 = classic default)

    Returns
    -------
    sklearn Pipeline
    """
    return Pipeline([
        ('bow',        vectorizer),
        ('classifier', MultinomialNB(alpha=alpha))
    ])