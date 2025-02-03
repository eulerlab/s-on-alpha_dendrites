import os

import numpy as np
from matplotlib import pyplot as plt

from djimaging.utils.data_utils import load_h5_data
from djimaging.user.alpha.utils.populate_alpha import PROJECT_ROOT


def plot_chirp(ax, filepath=os.path.join(PROJECT_ROOT, 'data/Stimulus/chirp.h5'), plot_time=True, to_xlims=False,
               tmax=None, y0=None, yscale=None, **kwargs):


    chirp_stim = load_h5_data(filepath)
    y = chirp_stim['chirp']
    if plot_time:
        x = np.arange(0, chirp_stim['chirp'].size) * 1e-3
    else:
        x = np.arange(len(y))

    if tmax is not None:
        assert plot_time
        assert tmax > x[-1]
        x = np.append(x, tmax)
        y = np.append(y, 0)

    if yscale is not None:
        y = y * yscale / y.max()

    if y0 is not None:
        y = y + y0

    ax.plot(x, y, **kwargs)
    if to_xlims:
        ax.set_xlim(x.min(), x.max())


def plot_noise(ax, frame_idx=0, filepath=os.path.join(PROJECT_ROOT, 'data/Stimulus/noise.h5')):
    from djimaging.utils.data_utils import load_h5_data
    noise_stim = load_h5_data(filepath)
    ax.imshow(noise_stim['k'][:, :, frame_idx], cmap='Greys', clip_on=False, interpolation='none')


def plot_sinespot(ax, delay=0, **kwargs):
    t_stim_sinespot = np.linspace(delay, 2, 101)
    stim_sinespot = np.clip(np.sin(t_stim_sinespot * np.pi * 2), 0, None)
    stim_sinespot[t_stim_sinespot < 0] = 0
    ax.plot(t_stim_sinespot, stim_sinespot, **kwargs)


def plot_spot_spatial(ax, s_list, w=800, h=600, space=100):
    ax.set_aspect('equal', 'box')

    ax.set_xlim(0, w * len(s_list) + space * (len(s_list) - 1))
    ax.set_ylim(0, h)

    for i, s in enumerate(s_list):
        off = w * i + space * i

        ax.add_patch(plt.Rectangle((0 + off, 0), w, h, color='k', clip_on=False))
        ax.add_patch(plt.Circle((w / 2 + off, h / 2), s / 2, color='w', clip_on=True))
        ax.add_patch(plt.Rectangle((0 + off, 0), w, h, edgecolor='k', facecolor='none', clip_on=False))
