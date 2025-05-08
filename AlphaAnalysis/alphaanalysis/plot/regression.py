import pingouin as pg
import numpy as np
import seaborn as sns


def plot_regression(ax, x, y, c=None, label="", n_tests=1, scatter_kwargs=None, robust=False):
    if scatter_kwargs is None:
        scatter_kwargs = dict(alpha=0.5)

    x = x[np.isfinite(x) & np.isfinite(y)]
    y = y[np.isfinite(x) & np.isfinite(y)]

    if robust:
        method = 'percbend'
    else:
        method = 'pearson'

    test = pg.corr(x=x, y=y, method=method)
    r_stat = test.loc[method, 'r']
    pval = test.loc[method, 'p-val']

    if pval < 0.001 / n_tests:
        sig = '***'
    elif pval < 0.01 / n_tests:
        sig = '**'
    elif pval < 0.05 / n_tests:
        sig = '*'
    else:
        sig = 'n.s.'

    ax.scatter(x=x, y=y, color=c, **scatter_kwargs)
    sns.regplot(ax=ax, x=x, y=y, scatter=False, color=c, robust=robust, ci=None,
                line_kws=dict(lw=2, alpha=0.8), label=label + f"r={r_stat:.2f}, {sig}")

    return test
