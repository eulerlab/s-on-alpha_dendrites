import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def plot_paths(ax, paths, soma_xyz=None, soma_radius=10, i1=0, i2=1, offset=(0, 0, 0), **plot_kw):
    """Plot dendrites and soma if given"""
    if soma_xyz is not None:
        ax.add_patch(plt.Circle((soma_xyz[i1] + offset[i1], soma_xyz[i2] + offset[i2]), radius=soma_radius,
                                edgecolor='dimgray', facecolor='gray', clip_on=False, lw=0.5, zorder=100, alpha=0.8))

    color = plot_kw.pop('color', 'black')

    for path in paths:
        ax.plot(path[:, i1] + offset[i1], path[:, i2] + offset[i2], color=color,
                solid_capstyle='round', solid_joinstyle='round', **plot_kw)


def plot_convex_hull(ax, hull_points, offset_um=(0, 0), hull_kw=None):
    """Plot convex hull"""

    hull_kw_ = dict(color='k', lw=0.5, ls='--', alpha=0.5)
    if hull_kw is not None:
        hull_kw_.update(hull_kw)

    ax.plot(hull_points[:, 0] + offset_um[0], hull_points[:, 1] + offset_um[1], **hull_kw_)


def plot_all_morphologies(df_morph, order_groups_by='cell_tag', figsize=(8, 5), DEBUG=False,
                          plot_names=False, plot_side_views=True, min_cols=1, verbose=True, order=None):
    """Plot all morphologies, top and sideviews for given grouping"""
    groups = df_morph.groupby('group')
    n_cols = np.maximum(np.max(groups['group'].count()), min_cols)
    n_rows = 2 * len(groups)

    fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize, sharex='all',
                            gridspec_kw=dict(height_ratios=(3, 1) * len(groups)), squeeze=False)
    sns.despine(left=True, right=True, bottom=True, top=True)

    for ax in axs.flat:
        ax.set_aspect('equal', 'datalim')
        ax.set(xticks=[], yticks=[])

    # Plot test boxes
    for ax in axs.flat:
        ax.vlines([-100, 100], -10, 10, color='r', alpha=float(DEBUG))
        ax.plot([-15, 15, 15, -15, -15], [-15, -15, 15, 15, -15], c='r', alpha=float(DEBUG))

    for i, (group_name, group) in enumerate(groups):

        if order is not None:
            order = np.asarray(order)
            i = np.argmax(order == group_name)

        if not plot_names:
            axs[2 * i, 0].set_ylabel(group_name, rotation=0, labelpad=10)

        for j, (row_name, row) in enumerate(group.sort_values(order_groups_by, ascending=True).iterrows()):
            if verbose:
                print(i, j, row_name)
            plot_paths(ax=axs[2 * i, j], paths=pd.DataFrame(row.df_paths).path, soma_xyz=row.soma_xyz,
                       soma_radius=10, i1=0, i2=1, offset=-row.soma_xyz, lw=0.5, clip_on=False)

            if plot_side_views:
                plot_paths(ax=axs[2 * i + 1, j], paths=pd.DataFrame(row.df_paths).path, soma_xyz=row.soma_xyz,
                           soma_radius=10, i1=0, i2=2, offset=-row.soma_xyz, lw=0.5, clip_on=False)

            else:
                axs[2 * i + 1, j].axis('off')

    plt.tight_layout(pad=0.5, h_pad=1, w_pad=1)

    for i, (group_name, group) in enumerate(groups):

        if order is not None:
            order = np.asarray(order)
            i = np.argmax(order == group_name)

        for j, (row_name, row) in enumerate(group.sort_values(order_groups_by, ascending=True).iterrows()):
            axs[2 * i, j].set_title(row[order_groups_by], loc='left', fontsize=plt.rcParams['font.size'],
                                    y=1.1, ha='left', c='dimgray')

    fig.align_ylabels()
    return fig, axs
