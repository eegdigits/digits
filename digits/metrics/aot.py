import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# accuracy over time
class AOT():
    """
    Used for plotting accuarcy over time. Expects a numpy array of multiple
    accuracy scores (probably from multiple binary classifications) on one
    dimension and timepoints on the other axis.
    """

    def __init__(self, data, time_width=1, time_offset=0, std=2):
        self.data = data
        self.time_width = time_width
        self.time_offset = time_offset
        self.x = np.arange(data.shape[1]) * time_width - time_offset
        self.std = std

    def plot(self, color='#08429C', alpha=0.3, title='Classification Accuracy over time', subtitle='', save_file=None):
        data = self.data
        rcParams['figure.figsize'] = (12,5)
        plt.plot(self.x, data.mean(axis=0), color=color)
        plt.fill_between(self.x,
                         data.mean(axis=0) - self.std*data.std(axis=0),
                         data.mean(axis=0) + self.std*data.std(axis=0),
                         alpha=alpha, edgecolor=color, facecolor=color)
        plt.axvline(0, color='#404040')
        plt.axhline(0.5, color='#404040')
        plt.title('\n'.join([title, subtitle]))
        xlabels = self.x[::2]
        plt.xticks(xlabels, xlabels.astype('str'), rotation=45)
        plt.ylabel('Mean Accuracy \nwith standard deviation ({}x)'.format(self.std))
        plt.xlabel('Time Points with Sliding window of width {0}'.format(self.time_width))
        axes = plt.gca()
        axes.set_ylim([0.3, 1.0])
        if save_file:
            plt.savefig(save_file)
        plt.show()
