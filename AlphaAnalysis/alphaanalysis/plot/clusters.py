import numpy as np
import seaborn as sns

from .stimulus import plot_chirp
from .plot import align_x_box, change_box


def plot_cluster_mean(ax, times, traces, color='k', clip_t=np.inf, lw=1.2, dt=0):
    from djimaging.utils.math_utils import truncated_vstack
    from djimaging.utils.plot_utils import plot_mean_trace_and_std

    times = truncated_vstack(times, rtol=10)
    traces = truncated_vstack(traces, rtol=10)

    assert times.shape == traces.shape

    time = np.mean(times, axis=0) + dt

    clip_idx_l = np.sum([time < 0])
    clip_idx_r = np.sum([time < clip_t])

    plot_mean_trace_and_std(
        ax, time=time[clip_idx_l:clip_idx_r], traces=traces[:, clip_idx_l:clip_idx_r],
        label=None, color=color, color2='gray', lw=lw, downsample_factor=1)


def plot_dendrogram(model, cluster_idxs, features, colors, file_path=None, suffixes=None, invert_y_axis=False):
    from scipy.cluster import hierarchy
    import matplotlib

    # Compute linkage from model
    linkage_matrix = hierarchy.linkage(
        np.hstack(features), method=model.linkage, metric=model.metric, optimal_ordering=True)

    # Plot the corresponding dendrogram
    # dendr = hierarchy.dendrogram(
    # linkage_matrix, color_threshold=model.distance_threshold, above_threshold_color='k', orientation='right', ax=ax)

    cl_maps = []

    vabsmax = np.max([np.max(np.abs(features_i)) for features_i in features])
    for i, features_i in enumerate(features):
        cl_map = sns.clustermap(
            features_i,
            row_cluster=True,
            row_linkage=linkage_matrix,
            row_colors=[matplotlib.colors.to_rgb(colors[i]) for i in cluster_idxs - np.min(cluster_idxs)],
            col_cluster=False,
            figsize=(2, 4),
            cbar_pos=(0.25, .07, .25, .02),
            cbar_kws=dict(orientation='horizontal', ticks=(-4, 0, 4)),
            vmin=-vabsmax, vmax=vabsmax, cmap='coolwarm',
            linewidths=0,
            tree_kws=dict(lw=0.5),
        )

        cl_map.ax_heatmap.tick_params(axis='both', which='both', length=0)
        cl_map.ax_heatmap.set(xticks=[], yticks=[])

        stim_ax = cl_map.ax_col_dendrogram
        plot_chirp(stim_ax, c='k', lw=0.8, plot_time=False, to_xlims=True)  # TODO not aligned well
        align_x_box(ax=stim_ax, ref_ax=cl_map.ax_heatmap)
        change_box(stim_ax, dy=-0.15)
        stim_ax.set_title(suffixes[i], fontsize=8)

        cl_maps.append(cl_map)
        if invert_y_axis:
            cl_map.ax_heatmap.invert_yaxis()
            cl_map.ax_row_dendrogram.invert_yaxis()
            cl_map.ax_row_colors.invert_yaxis()
        cl_map.ax_row_colors.set_rasterization_zorder(1000000)
        cl_map.ax_heatmap.set_rasterization_zorder(1000000)

        cis, ci_counts = np.unique(cluster_idxs, return_counts=True)
        print(cis, ci_counts)
        if invert_y_axis:
            cis = cis[::-1]
            ci_counts = ci_counts[::-1]

        for ci, ci_count in zip(cis[:-1], np.cumsum(ci_counts[:-1])):
            cl_map.ax_heatmap.axhline(ci_count, c='k', ls='--', alpha=0.7, lw=0.8)

        if file_path is not None:
            cl_map.savefig(file_path + f'_{suffixes[i]}.pdf', dpi=600)

        cl_maps.append(cl_map)
    return cl_maps
