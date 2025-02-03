import warnings

import numpy as np
import scipy
from matplotlib import pyplot as plt
from sklearn.utils.extmath import randomized_svd


def center_surround_cosine_1d(x, mu, width_c, power_c, width_s, power_s, return_all=False):
    x = np.atleast_1d(x) - mu

    z_c = (0.5 * np.cos(x * np.pi / width_c) + 0.5) ** power_c
    z_c[x > width_c] = 0
    z_c[x < -width_c] = 0

    if width_s > 0:
        z_s = (0.5 * np.cos((x - width_c) * np.pi / width_s + np.pi) + 0.5) ** power_s
        z_s[(x >= -width_c) & (x <= width_c)] = 0
        z_s[x > width_c + 2 * width_s] = 0
        z_s[x < - width_c - 2 * width_s] = 0
        z_s = -z_s
    else:
        z_s = 0

    z = z_c + z_s

    if return_all:
        return z, z_c, z_s
    else:
        return z


def center_surround_cosine_2d(x, y, width_c, width_s, power_c, power_s, weight_c=1., weight_s=1.):
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    xx, yy = np.meshgrid(x, y, indexing='ij')
    dd = (xx ** 2 + yy ** 2) ** 0.5
    zz = center_surround_cosine_1d(dd, 0, width_c, power_c, width_s, power_s)

    pos_weight = np.sum(np.abs(zz[zz > 0]))
    neg_weight = np.sum(np.abs(zz[zz < 0]))

    if pos_weight > 0:
        zz[zz > 0] *= weight_c / pos_weight

    if neg_weight > 0:
        zz[zz < 0] *= weight_s / neg_weight

    return zz, xx, yy, dd


def create_srf(pxl_cxs, pxl_cys, width_c, width_s, weight_c, weight_s, power_c=1., power_s=1., dtype=np.float32):
    rf, xx, yy, dd = center_surround_cosine_2d(pxl_cxs, pxl_cys, width_c, width_s, power_c, power_s, weight_c, weight_s)
    rf = rf.astype(dtype)
    return rf


def create_trf(size_t, fps, amp_main, amp_pre, onset_main, offset_main, onset_pre, offset_pre, norm=False):
    dt = 1 / fps
    rf_time = (np.arange(-size_t, 0) + 1) * dt

    assert 0 < onset_main < offset_main < onset_pre < offset_pre < np.abs(rf_time[0])

    fit_time = np.concatenate([
        np.linspace(rf_time[0], -offset_pre, 10, endpoint=True),
        np.array([np.mean([-offset_pre, -onset_pre]),
                  np.mean([-onset_pre, -offset_main]),
                  np.mean([-offset_main, -onset_main])]),
        np.linspace(-onset_main, 0, 10, endpoint=True)])

    fit_amp = np.concatenate([
        np.zeros(10), np.array([amp_pre, 0, amp_main]), np.zeros(10)])

    assert fit_time.size == fit_amp.size

    trf = scipy.interpolate.interp1d(x=fit_time, y=fit_amp, kind='quadratic')(rf_time)

    if norm:
        w_pos = np.sum(trf[trf > 0])
        w_neg = np.sum(-trf[trf < 0])
        trf = trf / np.maximum(w_pos, w_neg)

    return trf


def merge_strf(srf, trf):
    """Reconstruct STRF from sRF and tRF"""
    assert trf.ndim == 1, trf.ndim
    assert srf.ndim == 2, srf.ndim
    rf = np.kron(trf, srf.flat).reshape(trf.shape + srf.shape)
    return rf


def split_strf(strf):
    """
    Assuming an RF is time-space separable, get spatial and temporal filters using SVD.
    From RFEst.
    """
    dims = strf.shape

    dims_tRF = dims[0]
    dims_sRF = dims[1:]
    U, S, Vt = randomized_svd(strf.reshape(dims_tRF, -1), 3, random_state=0)
    srf = Vt[0].reshape(*dims_sRF)
    trf = U[:, 0]

    trf /= np.max(np.abs(trf))
    srf = (srf / np.max(np.abs(srf))) * np.max(np.abs(strf))

    merged_strf = merge_strf(srf, trf)
    if not np.isclose(np.mean(merged_strf), np.mean(strf)):
        warnings.warn(f"{np.mean(merged_strf)}, {np.mean(strf)}")

    return srf, trf


def plot_srf(rf, ax=None, vabsmax=None, cb=False, stim_extent=None):
    dims = rf.shape
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(4, 4 * dims[1] / dims[0]))

    if vabsmax is None:
        vabsmax = np.nanmax(np.abs(rf))

    _rf = rf.copy()
    _rf[_rf == 0] = np.nan
    im = ax.imshow(_rf.T, vmin=-vabsmax, vmax=vabsmax, cmap='coolwarm', origin='lower', extent=stim_extent)

    if cb:
        plt.colorbar(im, ax=ax)


def plot_trf(trf, ax=None, vabsmax=None, t_trf=None, fps=None):
    if vabsmax is None:
        vabsmax = np.nanmax(np.abs(trf))

    if t_trf is None:
        t_trf = np.arange(-trf.size, 0) + 1
        if fps is not None:
            t_trf = t_trf / float(fps)
            xlab = 'Time [s]'
        else:
            xlab = 'Time'
    else:
        xlab = 'Time'
        assert fps is None

    assert trf.shape == t_trf.shape

    ax.plot(t_trf, trf)
    ax.set_ylim(-1.05 * vabsmax, 1.05 * vabsmax)
    ax.set_xlabel(xlab)
