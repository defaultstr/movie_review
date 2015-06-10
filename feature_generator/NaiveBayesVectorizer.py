__author__ = 'defaultstr'

from .base import Vectorizer
from collections import Counter
from scipy.sparse import csr_matrix
import numpy as np
import logging



class NaiveBayesVectorizer(Vectorizer):
    def __init__(self, n_grams=[1, 2, 3], config_identifier='n_gram_123'):
        super(NaiveBayesVectorizer, self).__init__(config_identifier=config_identifier)
        self.n_grams = n_grams


    def _build_dict(self, reviews, y):
        logging.log(logging.DEBUG, 'counting...')
        self.pos_dict = Counter()
        self.neg_dict = Counter()
        for label, review in zip(y, reviews):
            if label == 1:
                self.pos_dict.update(self._tokenize(review))
            elif label == 0:
                self.neg_dict.update(self._tokenize(review))

    def _compute_ratio(self, alpha=1):
        logging.log(logging.DEBUG, 'compute r...')
        all_tokens = set(self.pos_dict.keys()+self.neg_dict.keys())
        dict = {t: i for i, t in enumerate(all_tokens)}
        d = len(dict)
        p = np.ones(d) * alpha
        q = np.ones(d) * alpha
        for t in all_tokens:
            p[dict[t]] += self.pos_dict[t]
            q[dict[t]] += self.neg_dict[t]
        p /= np.sum(np.abs(p))
        q /= np.sum(np.abs(q))
        self.r = np.log(p/q)
        self.dict = dict

    def fit(self, X, y):
        self._build_dict(X, y)
        self._compute_ratio()

    def transform(self, X, y=None):
        logging.log(logging.DEBUG, 'processing transformation...')
        N = len(X)
        d = len(self.r)
        data = []
        row = []
        col = []
        for row_idx, review in enumerate(X):
            tokens = self._tokenize(review)
            indices = []
            for t in tokens:
                try:
                    indices.append(self.dict[t])
                except KeyError:
                    pass
            indices = list(set(indices))
            indices.sort()
            for idx in indices:
                data.append(self.r[idx])
                row.append(row_idx)
                col.append(idx)
        return csr_matrix((data, (row, col)), shape=(N, d))

    def get_dimension(self):
        return len(self.r)





