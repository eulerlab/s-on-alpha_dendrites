import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from . import style


def plot_location_grid(ax, text=True, left='n', right='t', top='d', bottom='v', plot_circles=True, change_axis=True):
    if change_axis:
        ax.axis('equal')
        ax.axis('off')

    radians = np.linspace(0, 2 * np.pi, 100)
    grid_kw = dict(c='gray', alpha=1, zorder=-10, clip_on=False)

    if plot_circles:
        ax.plot(np.sin(radians), np.cos(radians), **grid_kw)
        ax.plot(0.5 * np.sin(radians), 0.5 * np.cos(radians), **grid_kw)

    ax.plot([-1, 1], [0, 0], **grid_kw)
    ax.plot([0, 0], [-1, 1], **grid_kw, )

    # Labels
    if text:
        text_kw = dict(c='dimgray', fontsize=plt.rcParams['axes.titlesize'])
        ax.text(x=-1.1, y=0, s=left, ha='right', va='center', **text_kw)
        ax.text(x=1.1, y=0, s=right, ha='left', va='center', **text_kw)
        ax.text(x=0, y=1.1, s=top, ha='center', va='bottom', **text_kw)
        ax.text(x=0, y=-1.1, s=bottom, ha='center', va='top', **text_kw)


def plot_cell_locations(df_cell_location, indicator, markerby=None, text=True, alpha=0.7, ax=None, legend=True,
                        **kwargs):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    else:
        fig = None

    plot_location_grid(ax, text=text)

    df_cell_location = df_cell_location.sort_values('group').copy()
    df_cell_location['x'] = -df_cell_location['temporal_nasal_pos']
    df_cell_location['y'] = df_cell_location['ventral_dorsal_pos']

    sns.scatterplot(
        ax=ax, data=df_cell_location, x="x", y="y", hue='group', palette=style.get_palette(indicator),
        style=markerby, alpha=alpha, legend='brief', edgecolor='k', **kwargs)
    ax.legend(loc='upper left', bbox_to_anchor=(0.95, 1.1))

    if not legend:
        # remove legend
        ax.get_legend().remove()

    return fig, ax
