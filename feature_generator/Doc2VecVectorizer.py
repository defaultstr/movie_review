__author__ = 'defaultstr'

from .base import Vectorizer
from gensim.models import doc2vec
import numpy as np


class Doc2VecVectorizer(Vectorizer):
    def __init__(self,
                 n_grams=[1], #only use word
                 dimension=300, config_identifier='_300',
                 sample=1e-3,
                 window=10,
                 min_count=40,
                 negative=5,
                 hs=1,
                 ):
        super(Doc2VecVectorizer, self).__init__(config_identifier=config_identifier)
        self.dimension = dimension
        self.sample = sample
        self.window = window
        self.min_count = min_count
        self.n_grams = n_grams
        self.negative = negative
        self.hs = hs

    def initialize(self, train_X, test_X, unlabeled_X):
        tokens = [doc2vec.LabeledSentence(self._tokenize(x), x)  for x in train_X] +\
                 [doc2vec.LabeledSentence(self._tokenize(x), x)  for x in test_X] +\
                 [doc2vec.LabeledSentence(self._tokenize(x), x)  for x in unlabeled_X]
        
        self._model = doc2vec.Doc2Vec(tokens, size=self.dimension,
                                        min_count=self.min_count,
                                        window=self.window,
                                        negative=self.negative,
                                        hs=self.hs,
                                        sample=self.sample)

    def fit(self, X, y=None):
        pass

    def transform(self, X, y=None):
        ret = np.zeros((len(X), self.dimension), dtype='float32')
        for idx, review in enumerate(X):
            ret[idx] = self._model[review]
        return ret

    def get_dimension(self):
        return self.dimension






