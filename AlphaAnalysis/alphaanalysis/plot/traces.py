import numpy as np

from alphaanalysis.plot import lines, plot_cluster_mean


def plot_traces(ax, tab, v_lines=None, xlim=None, color='k', time_name='average_times', trace_name='average',
                h_lines=None, lw=1.2, dt=0):
    clip_t = xlim[1] if xlim is not None else np.inf

    if len(tab) > 0:
        plot_cluster_mean(
            ax, times=tab.fetch(time_name), traces=tab.fetch(trace_name), color=color, clip_t=clip_t, lw=lw, dt=dt)

    if v_lines is not None:
        lines(ax, ts=v_lines, orientation='v')

    if h_lines is not None:
        lines(ax, ts=h_lines, orientation='h')

    ax.set_xlabel('Time [s]')

    if xlim is not None:
        ax.set_xlim(xlim)
