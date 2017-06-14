# -*- coding: utf-8 -*-
"""
This micro package contains a class Confusion and ConfusionGrid. They are
useful helpers to plot a confusion matrix based on a fitted sklearn estimator,
some testing inputs and outputs.
"""
from sklearn import metrics
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from joblib import Parallel, delayed
from sklearn.utils.extmath import cartesian


class Confusion:

    def __init__(self, estimator, X_test, y_test, normalize=True, title="Confusion Matrix"):
        self.y_test = y_test
        self.y_pred = estimator.predict(X_test)
        self.acc = sum(self.y_pred == self.y_test) / len(self.y_test)
        self.title = title + " [Acc={0:.1f}%]".format(100*self.acc)
        self.colors = plt.cm.Blues
        self.matrix = metrics.confusion_matrix(y_test, self.y_pred)
        self.vmax = None
        self.normalize = normalize
        if normalize:
            mat = self.matrix
            self.matrix = mat.astype('float') / mat.sum(axis=1)[:, np.newaxis]
            self.matrix *= 100
            self.vmax = 100.0

    # http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    def plot(self, compact=False, save_file=None, font_size=20):
        """
        Plot the single confusion matrix and optionally save to a file.
        """
        rcParams['figure.figsize'] = (6,4)
        plt.imshow(self.matrix, interpolation='nearest',
                   cmap=self.colors, vmin=0, vmax=self.vmax)
        #plt.tight_layout()
        plt.title(self.title)
        tick_marks = np.arange(self.matrix.shape[0])
        if not compact:
            if self.normalize:
                plt.colorbar(label='Classification ratio for predicted labels in %')
            else:
                plt.colorbar(label='Classification count for predicted labels')
            plt.xticks(tick_marks, np.unique(self.y_test).astype('str'), rotation=45)
            plt.xlabel('Predicted label')
        else:
            plt.xticks([])
            plt.xlabel('')
        plt.yticks(tick_marks, np.unique(self.y_test).astype('str'))
        plt.ylabel('True label')
        ax = plt.gca()
        for x, y in cartesian(( np.arange(len(self.matrix)), np.arange(len(self.matrix)) )):
            val = '{:2.0f}'.format(self.matrix[y,x])
            ax.text(x, y, val, va='center', ha='center', size=str(font_size))
        if save_file:
            plt.savefig(save_file)
        plt.show()


class ConfusionGrid:

    def __init__(self, grid, estimator, X_train, X_test, y_train, y_test, width=2, bestof=12):
        self.best_scores = scores.best_params(grid, bestof)
        self.estimator = estimator
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.subplot_cols = width
        self.subplot_rows =  int(bestof/width) + 1

    def plot(self, n_jobs=-1, verbose=1):
        """
        Plot a grid of confusion matrices based on all tested meta parameters of a classifiers.
        Re-fits the classifier in parallel.
        """
        rcParams['figure.figsize'] = (10,25)
        clfs = Parallel(n_jobs=n_jobs, backend="threading")(
            delayed(get_clf)(self.estimator, params, self.X_train, self.y_train,
                             ix, len(self.best_scores), verbose)
                for ix, (params,_,_) in enumerate(self.best_scores)
        )
        for ix, clf in enumerate(clfs):
            params, _, _ = self.best_scores[ix]
            score = clf.score(self.X_test, self.y_test)
            plt.subplot(self.subplot_rows, self.subplot_cols, ix+1)
            title='Score={0:4.2f}\n{1}'.format(score, format_params(params))
            conf = Confusion(clf, self.X_test, self.y_test, title=title)
            conf.plot(compact=True)

def get_clf(estimator, params, X, y, ix, len, verbose=0):
    if verbose > 0:
        print("Re-fitting estimator ({0}/{1})".format(ix+1, len))
    clf = estimator(**params)
    return clf.fit(X,y)

def format_params(params):
    res_string = ""
    for name, val in params.items():
        if np.isreal(val):
            val = "{0:4.2f}".format(val)
        res_string += "{0}={1} ".format(name, val)
    return res_string
