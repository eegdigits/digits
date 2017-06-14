# -*- coding: utf-8 -*-
"""
Stubs to reshape data from CSP output shape to expected LDA input shape and vice versa.
Useful to combine mne.decoding.CSP and sklearn.LDA in a pipeline.
Needs to implement fit_transform/transform and get_params.
"""
from ..data import select

class CSPWrap():
    def __init__(self):
        pass

    def fit(self, X, y):
        pass

    def fit_transform(self, X, y):
        return self.transform(X)

    def transform(self, X):
        samplen = len(X)
        chlen = len(select.getchannelnames(X))
        return X.values.reshape(samplen, chlen, -1).astype('float32')

    def get_params(self, **kwargs):
        return {}


class CSPFlatten():

    def __init__(self):
        pass

    def fit(self, X, y):
        pass

    def fit_transform(self, X, y):
        return self.transform(X)

    def transform(self, X):
        # very ugyl, must check ordering here!
        return X.reshape(X.shape[0], -1)

    def get_params(self, **kwargs):
        return {}
