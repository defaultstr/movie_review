__author__ = 'defaultstr'

from .base import Vectorizer
from gensim.models import word2vec
import numpy as np


class Word2VecVectorizer(Vectorizer):
    def __init__(self,
                 n_grams=[1], #only use word
                 dimension=300, config_identifier='_300',
                 sample=1e-3,
                 window=10,
                 min_count=40,
                 pooling_method='average' #'average', 'max'
                 ):
        super(Word2VecVectorizer, self).__init__(config_identifier=config_identifier)
        self.dimension = dimension
        self.sample = sample
        self.window = window
        self.min_count = min_count
        self.n_grams = n_grams
        self.pooling_method = pooling_method

    def initialize(self, train_X, test_X, unlabeled_X):
        tokens = [self._tokenize(x) for x in train_X] +\
                 [self._tokenize(x) for x in test_X] +\
                 [self._tokenize(x) for x in unlabeled_X]
        self._model = word2vec.Word2Vec(tokens, size=self.dimension,
                                        min_count=self.min_count,
                                        window=self.window,
                                        sample=self.sample)

    def fit(self, X, y=None):
        pass

    def transform(self, X, y=None):
        ret = np.zeros((len(X), self.dimension), dtype='float32')
        for idx, review in enumerate(X):
            tokens = self._tokenize(review)
            vec_buf = np.zeros((len(tokens), self.dimension), dtype='float32')
            index2word = set(self._model.index2word)
            for i, t in enumerate(tokens):
                if t in index2word:
                    vec_buf[i, :] = self._model[t]
            if self.pooling_method == 'average':
                ret[idx] = np.average(vec_buf, axis=0)
        return ret

    def get_dimension(self):
        return self.dimension






