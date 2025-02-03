import numpy as np
import seaborn as sns


def plot_hists(ax, cls_data, all_data, binrange, c):
    sns.histplot(ax=ax, data=all_data, color='k', bins=15, binrange=binrange, edgecolor='None')
    downscale_patchs(ax=ax, f=all_data.size / cls_data.size)
    if cls_data.size > 0:
        sns.histplot(ax=ax, data=cls_data, color=c, bins=15, binrange=binrange, edgecolor='None')

    max_height = np.max([patch.get_height() for patch in ax.patches])
    ax.set_ylim(0, max_height * 1.05)


def downscale_patchs(ax, f):
    for patch in ax.patches:
        current_height = patch.get_height()
        new_height = current_height / f
        patch.set_height(new_height)


def annotated_countplot(ax, counts, order, palette, cmin=20, cbase=50):
    try:
        sns.countplot(ax=ax, x=counts, order=order, palette=palette, legend=False)
    except:
        sns.countplot(ax=ax, x=counts, order=order, palette=palette)

    for p in ax.patches:
        c = p.get_height()
        if np.isfinite(c):
            if c > cmin:
                text_pos_y, color = 0.5 * c, 'w'
            else:
                text_pos_y, color = cbase, 'k'

            ax.annotate(int(c), (p.get_x() + 0.6 * p.get_width(), text_pos_y),
                        ha='center', va='center', color=color, rotation=90)
