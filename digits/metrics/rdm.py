# -*- coding: utf-8 -*-
"""
Trivial wrapper class to plot a RDM/decoding matrix with matplotlib.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from itertools import combinations

class RDM():

    def __init__(self, data):
        # make symmetric
        if not np.allclose(data.transpose(), data):
            data = data + data.transpose()
        # scale to percentages for easy plotting
        if np.max(data) <= 1.0 :
            data = data * 100

        self.mean = data.sum() / np.sum(data != 0)
        self.data = data

        # set diagonal to 100%
        #if data[0,0] == 0:
        #    data += 100 * np.eye(10)

    # representational dissimilarity matrix
    def plot(self, title='Representational Dissimilarity Matrix', save_file=None):
        data = self.data
        rcParams['figure.figsize'] = (6,4)
        plt.imshow(data, interpolation='nearest', cmap='RdBu', vmin=0, vmax=100)
        plt.title(title)
        plt.colorbar(label='Validation Accuracy in %\n mean:{:.2f} %'.format(self.mean))
        tick_marks = np.arange(data.shape[0])
        plt.xticks(tick_marks, tick_marks.astype('str'), rotation=45)
        plt.yticks(tick_marks, tick_marks.astype('str'))
        plt.ylabel('Digit')
        plt.xlabel('Digit')
        ax = plt.gca()
        for x, y in combinations(np.arange(10), 2):
            val = '{:2.0f}'.format(data[x,y])
            ax.text(x, y, val, va='center', ha='center')
            ax.text(y, x, val, va='center', ha='center')
        if save_file:
            plt.savefig(save_file)
        plt.show()
