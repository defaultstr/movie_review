__author__ = 'defaultstr'

import xgboost as xgb
import numpy as np


class XGBoost(object):

    def __init__(self):
        self.bst = None

    def fit(self, X, y):
        dtrain = xgb.DMatrix(X)
        dtrain.set_label(y)
        plist = {'objective': 'binary:logistic', 'slient': 1}.items()
        self.bst = xgb.train(plist, dtrain, 200, [])

    def predict(self, X):
        dtest = xgb.DMatrix(X)
        pred_y = self.bst.predict(dtest)
        pred_y = np.array([1 if y > 0.5 else 0 for y in pred_y])
        return pred_y
