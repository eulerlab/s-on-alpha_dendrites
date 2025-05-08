import numpy as np
from matplotlib import pyplot as plt


def plot_heatmap(ax, stim, vabsmax=None):
    if vabsmax is None:
        vabsmax = np.max(np.abs(stim))
    im = ax.imshow(stim, vmin=-vabsmax, vmax=vabsmax, cmap='bwr')
    plt.colorbar(im, ax=ax)
