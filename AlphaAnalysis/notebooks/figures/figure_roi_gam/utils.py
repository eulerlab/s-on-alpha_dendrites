import numpy as np
import seaborn as sns
from alphaanalysis import plot as plota
from matplotlib import pyplot as plt
from scipy.stats import skew


def prep_df(df):
    id_cols = [col for col in df.columns if
               (col.endswith('_id') and not col in ['roi_id']) or col.endswith('_hash') or col in ['condition',
                                                                                                   'stim_name']]
    for id_col in id_cols:
        if df[id_col].nunique() == 1:
            df.drop(id_col, axis=1, inplace=True)
    df['cell_id'] = (df['date'].astype(str) + '_' + df['exp_num'].astype(str)).astype('category')
    df['field_id'] = (df['date'].astype(str) + '_' + df['exp_num'].astype(str) + '_' + df['field'].astype(str)).astype(
        'category')
    df['group'] = df['group'].astype('category')
    # df['soma_dist'] = df['soma_dist']**0.5
    return df


def plot_skew(*arrs, f_trans):
    fig, ax = plt.subplots(1, 1)
    for i, arr in enumerate(arrs):
        t_arr = f_trans(arr)
        s = skew(t_arr)
        sns.kdeplot(t_arr, ax=ax, label=f"{s:.2f}", color=f'C{i}')
        sns.histplot(t_arr, ax=ax, label="_", stat='density', color=f'C{i}')
    ax.legend()
    plt.show()


def plot_scatter_metrics(df, metrics, groupby='group'):
    import warnings
    
    warnings.simplefilter(action='ignore', category=FutureWarning)
    
    groups = df.groupby(groupby)

    fig, axs = plt.subplots(len(metrics), 1, figsize=(3, 4), sharex='all', squeeze=False)
    axs = axs.flatten()
    for i, (group, df_group) in enumerate(groups):
        for j, ax in enumerate(axs):
            sns.regplot(ax=ax, data=df_group, x="soma_dist", y=metrics[j], scatter_kws=dict(s=2), label=group)
            ax.legend()

    plt.tight_layout()


def plot_fits(df, df_preds, pairs_sig_regions_list, titles, ys, ylabels, x='soma_dist', xlabel='Dist. to soma [µm]',
              order=None, colors=None, is_row=True, figsize=None, **gam_kws):
    import warnings
    
    warnings.simplefilter(action='ignore', category=FutureWarning)
    
    if is_row:
        nrows = 2
        ncols = len(df_preds)
        height_ratios = (4, 1)
    else:
        nrows = 2 * len(df_preds)
        ncols = 1
        height_ratios = [4, 1] * len(df_preds)

    if figsize is None:
        figsize = (1 + 1.8 * ncols, 0.55 + 0.6 * nrows)

    print(figsize)

    fig, axs_all = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, height_ratios=height_ratios, squeeze=False,
                                sharey='row' if np.unique(ys).size == 1 else False)
    sns.despine()

    gam_kws_ = dict(ms=2, ma=0.5, f_se=1)
    if len(gam_kws) > 0:
        gam_kws_.update(gam_kws)

    for i, (df_pred, pairs_sig_regions) in enumerate(zip(df_preds, pairs_sig_regions_list)):
        title = titles[i]
        y = ys[i]
        ylabel = ylabels[i]

        if is_row:
            axs = axs_all[:, i]
        else:
            axs = axs_all[i * 2:(i + 1) * 2, 0]

        plota.plot_gam_fits(
            df=df, df_pred=df_pred, x=x, y=y,
            pairs_sig_regions=pairs_sig_regions,
            side_groups=order, colors=colors, axs=axs, **gam_kws_)

        axs[0].set_title(title)

        if i == len(df_preds) - 1:
            axs[1].set(xlabel=xlabel)
        else:
            axs[1].set_xticklabels([])

        axs[0].set(ylabel=ylabel)
        axs[0].set_xlim(0, df[x].max())
        axs[1].set_xlim(0, df[x].max())

    for ax in axs_all[:, :-1].flat:
        ax.legend().set_visible(False)

    axs[0].legend(loc='center left', bbox_to_anchor=(1.02, 0.8), handlelength=1)
    axs[1].legend(loc='center left', bbox_to_anchor=(1.02, 0.5), handlelength=1)

    plt.tight_layout(w_pad=3)
    return fig, axs
