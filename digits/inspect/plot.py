from matplotlib import rcParams
import matplotlib.pyplot as plt
import numpy as np

def normhist(X, width=4, bins=100, left=None, right=None):
    """
    Trivial wrapper for plt.hist with normalized center bin and left/right
    limits according to width * standard deviation.
    """
    rcParams['figure.figsize'] = (10, 2)
    if hasattr(X,'flatten'):
        X = X.flatten()
    std = np.std(X)
    mean = np.mean(X)
    if right is None:
        right = mean + width * std
    if left is None:
        left = mean - width * std
    bins = np.linspace(left, right, bins)
    _ = plt.hist(X, bins=bins)
