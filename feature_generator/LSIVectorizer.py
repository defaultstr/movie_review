__author__ = 'defaultstr'
from .base import Vectorizer
from collections import Counter
from gensim.models.lsimodel import LsiModel
import numpy as np


class LSIVectorizer(Vectorizer):
    def __init__(self, n_grams=[1, 2, 3], n_topics=100, config_identifier='n_gram_123'):
        super(LSIVectorizer, self).__init__(config_identifier=config_identifier)
        self._model = None
        self.n_topics = n_topics
        self.n_grams = n_grams

    def _build_dict(self, X):
        cnt = Counter()
        for review in X:
            tokens = self._tokenize(review)
            cnt.update(tokens)
        self.dict = {t: i for i, t in enumerate(cnt)}

    def fit(self, X, y=None):
        self._build_dict(X)
        corpus = []
        for review in X:
            cnt = Counter()
            indices = [self.dict[t] for t in self._tokenize(review)]
            cnt.update(indices)
            corpus.append(cnt.items())
        self._model = LsiModel(corpus, num_topics=self.n_topics)

    def transform(self, X, y=None):
        N = len(X)
        ret = np.empty((N, self.n_topics))
        for i in range(N):
            cnt = Counter()
            indices = []
            for t in self._tokenize(X[i]):
                try:
                    indices.append(self.dict[t])
                except KeyError:
                    pass
            cnt.update(indices)
            ret[i, :] = [x[1] for x in self._model[cnt.items()]]
        return ret

    def get_dimension(self):
        return self.n_topics
