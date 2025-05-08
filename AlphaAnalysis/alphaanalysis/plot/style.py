import seaborn as sns

PALETTE = 'tab10'

_calcium_color_dict = {
    'n': sns.color_palette(PALETTE)[1],
    'd': sns.color_palette(PALETTE)[0],
    't': sns.color_palette(PALETTE)[2],
}

_glutamate_color_dict = {
    'n': sns.color_palette(PALETTE)[1],
    't': sns.color_palette(PALETTE)[2],
}


def get_order(indicator):
    if indicator == 'calcium':
        return ['n', 'd', 't']
    elif indicator == 'glutamate':
        return ['n', 't']
    else:
        raise NotImplementedError(indicator)


def get_palette(indicator):
    if indicator == 'calcium':
        return _calcium_color_dict
    elif indicator == 'glutamate':
        return _glutamate_color_dict
    else:
        raise NotImplementedError(indicator)


def get_group_color(group, indicator):
    return get_palette(indicator)[group]


_label_dict = {
    'nasal': 'N',
    'temporal': 'T',
    'temporal-dorsal': 'TD',
    'temporal-ventral': 'TV',
    'indicator': '',
}


def set_legend_side_labels(ax):
    for t in ax.get_legend().texts:
        t.set_text(_label_dict.get(t.get_text(), t.get_text()))


def set_side_xlabels(ax):
    ax.set_xticklabels([_label_dict.get(t.get_text(), t.get_text()) for t in ax.get_xticklabels()])
