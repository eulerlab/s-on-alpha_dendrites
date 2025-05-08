import numpy as np

from alphacnn.response.distributions import normal


def w_syn_truncated_normal(dist, std, cut, w_tot=1.) -> np.ndarray:
    if std > 0:
        w_syn = normal(dist, 1, std)
    else:
        w_syn = np.zeros_like(dist)

    w_syn[dist > cut] = 0.0

    if np.sum(w_syn) > 0:
        w_syn = w_tot * w_syn / np.sum(w_syn)
    return w_syn


def fun_ws_nsl(dist, **kw) -> np.ndarray:
    return w_syn_truncated_normal(dist, **kw)


def fun_ss_nsl(dist, **kw) -> np.ndarray:
    return w_syn_truncated_normal(dist, **kw)


def fun_ws_tmp(dist, **kw) -> np.ndarray:
    return w_syn_truncated_normal(dist, **kw)


def fun_ss_tmp(dist, **kw) -> np.ndarray:
    return w_syn_truncated_normal(dist, **kw)


def compute_rgc_weights(rf_dia_n, bc_cdist, weight_fun, weight_kws):
    xs = np.arange(rf_dia_n) * bc_cdist
    xs = xs - np.mean(xs)
    dist_to_center = np.sum((np.stack(np.meshgrid(xs, xs, indexing='ij'), axis=2)) ** 2, axis=2) ** 0.5
    weights = weight_fun(dist_to_center, **weight_kws)
    return weights
