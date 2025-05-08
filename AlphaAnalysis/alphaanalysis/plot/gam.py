import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np

from alphaanalysis.plot import move_xaxis_outward, make_share_xlims


def plot_gam_fits(df_pred, x, y, side_groups, pairs_sig_regions, group='group', df=None, f_se=1,
                  colors=None, figsize=(3, 2.3), height_ratios=(5, 1), ms=0.5, ma=0.5, legend=True, axs=None):
    if axs is None:
        fig, axs = plt.subplots(2, 1, figsize=figsize, height_ratios=height_ratios)
        sns.despine()

    df_pred = df_pred.copy()
    if 'ub.fit' not in df_pred.columns:
        df_pred['ub.fit'] = df_pred['fit'] + f_se * df_pred['se.fit']
    if 'lb.fit' not in df_pred.columns:
        df_pred['lb.fit'] = df_pred['fit'] - f_se * df_pred['se.fit']

    if df is not None:
        x_min, x_max = df[x].min(), df[x].max()
        y_min, y_max = df[y].min(), df[y].max()
    else:
        x_min, x_max = df_pred[x].min(), df_pred[x].max()
        y_min, y_max = df_pred['fit'].min(), df_pred['fit'].max()

    # Plot predictions and points
    ax = axs[0]

    for i, side in enumerate(side_groups):
        group_pred = df_pred[df_pred[group] == side]
        group_pred = group_pred.sort_values(x)
        ax.plot(group_pred[x], group_pred['fit'], label=side, color=colors[i])
        ax.fill_between(group_pred[x], group_pred['lb.fit'], group_pred['ub.fit'], color=colors[i], alpha=0.3, lw=0)

    if df is not None:  # Plotting points is optional
        sns.scatterplot(ax=ax, data=df, x=x, y=y, hue=group, hue_order=side_groups, palette=colors, s=ms, alpha=ma,
                        legend=False, clip_on=False, edgecolor='none', zorder=100)
        ax.set_xlabel(None)

    ax.set(ylim=(y_min, y_max), xticks=[])
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), handlelength=1)
    ax.spines['bottom'].set_visible(False)

    # Plot regions of significance
    ax = axs[1]
    yticks = []
    yticklabels = []

    labeled = False
    for i, (pair, sig_regions) in enumerate(pairs_sig_regions):
        yticks.append(-i)
        yticklabels.append(pair)
        ax.hlines(y=-i, xmin=x_min, xmax=x_max, color='gray', lw=0.8, ls='--')
        for sig_region in sig_regions:
            ax.hlines(y=-i, xmin=sig_region[0], xmax=sig_region[1], color='r', lw=2,
                      label='*' if not labeled else '_', clip_on=False)
            labeled = True

    ax.set_yticks(yticks, yticklabels)
    if legend:
        ax.legend(loc='center left', bbox_to_anchor=(1, 1), handlelength=1)
    move_xaxis_outward(ax)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y', length=0)

    make_share_xlims(axs)

    return axs


def get_sig_intervals(df_diff, x='soma_dist'):
    sig_idxs_lb = np.where(np.diff(np.sign(df_diff.est - df_diff.CI) > 0))[0]
    sig_idxs_ub = np.where(np.diff(np.sign(df_diff.est + df_diff.CI) < 0))[0]

    # Edge cases
    if (df_diff.est - df_diff.CI).iloc[0] > 0:
        sig_idxs_lb = np.append(0, sig_idxs_lb)

    if (df_diff.est - df_diff.CI).iloc[-1] > 0:
        sig_idxs_lb = np.append(sig_idxs_lb, df_diff.shape[0] - 1)

    if (df_diff.est + df_diff.CI).iloc[0] < 0:
        sig_idxs_ub = np.append(0, sig_idxs_ub)

    if (df_diff.est + df_diff.CI).iloc[-1] < 0:
        sig_idxs_ub = np.append(sig_idxs_ub, df_diff.shape[0] - 1)

    sig_intervals = []
    for sig_idxs in [sig_idxs_lb, sig_idxs_ub]:
        print(sig_idxs)
        for idxa, idxb in zip(sig_idxs[0::2], sig_idxs[1::2]):
            sig_intervals.append((df_diff[x].values[idxa], df_diff[x].values[idxb]))
    return sig_intervals
