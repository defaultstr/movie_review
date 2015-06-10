__author__ = 'defaultstr'
from .base import Vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer as tfidf


class TfIdfVectorizer(Vectorizer):
    def __init__(self, n_grams=[1, 2, 3], config_identifier='n_gram_123'):
        super(TfIdfVectorizer, self).__init__(config_identifier=config_identifier)
        self._vec = tfidf(ngram_range=(min(n_grams), max(n_grams)))
        self.n_grams = n_grams

    def fit(self, X, y=None):
        self._vec.fit(X)

    def transform(self, X, y=None):
        return self._vec.transform(X)

    def get_dimension(self):
        return len(self._vec.get_feature_names())




