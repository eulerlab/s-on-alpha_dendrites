import numpy as np
from matplotlib import pyplot as plt

from alphacnn.geometry import hexagon


def get_square_grid(xlim, ylim, area=None, cdist=None, borderdist=None, verbose=0):
    assert (area is None) != (cdist is None)

    if area is not None:
        cdist = np.sqrt(area)
    else:
        area = cdist ** 2

    if borderdist is None:
        borderdist = 0

    dx = cdist
    nx = int(np.floor((abs(xlim[1] - xlim[0]) - 2 * borderdist) / dx))
    cxs = np.arange(nx) * dx
    cxs = cxs - np.mean(cxs)

    dy = cdist
    ny = int(np.floor((abs(ylim[1] - ylim[0]) - 2 * borderdist) / dy))
    cys = np.arange(nx) * dy
    cys = cys - np.mean(cys)

    cxys = np.stack(np.meshgrid(cxs, cys), axis=2).reshape(-1, 2)

    hdists = hexagon.get_square_coords(cdist / 2.).T
    hxys = np.moveaxis(np.tile(cxys, (4, 1, 1)), 0, 2) + hdists

    return cxys, hxys, cdist, area


def get_hex_grid(xlim, ylim, area=None, cdist=None, borderdist=None, verbose=0):
    assert (area is None) != (cdist is None)

    if area is not None:
        ri = hexagon.area2ri(area)
        ru = hexagon.ri2ru(ri)
        cdist = ri * 2.
    else:
        ri = cdist / 2.
        ru = hexagon.ri2ru(ri)
        area = hexagon.ri2area(ri)

    if verbose:
        print(f"cdist={cdist:.2g}, area={area:.2g}, ri={ri:.2g}, ru={ru:.2g}")

    if borderdist is None:
        borderdist = ru

    ru = abs(ru)
    ri = abs(ri)

    dx = 3. * ru
    nx = int(np.ceil((abs(xlim[1] - xlim[0]) - 2 * borderdist) / dx)) + 2
    cxs_a = np.arange(nx) * dx
    cxs_a -= cxs_a[nx // 2]

    while cxs_a[0] < xlim[0] + borderdist:
        cxs_a = cxs_a[1:]

    while cxs_a[-1] > xlim[1] - borderdist:
        cxs_a = cxs_a[:-1]

    cxs_b = np.append(cxs_a, cxs_a[-1] + dx) - dx / 2.

    while cxs_b[0] < xlim[0] + borderdist:
        cxs_b = cxs_b[1:]

    while cxs_b[-1] > xlim[1] - borderdist:
        cxs_b = cxs_b[:-1]

    dy = ri

    ny = int(np.ceil((abs(ylim[1] - ylim[0]) - 2 * borderdist) / dy)) + 2
    cys = np.arange(ny) * dy
    cys -= cys[ny // 2]

    while cys[0] < ylim[0] + borderdist:
        cys = cys[1:]

    while cys[-1] > ylim[1] - borderdist:
        cys = cys[:-1]

    if verbose:
        print(f"x in {cxs_a[0]:.2g} to {np.maximum(cxs_a[-1], cxs_b[-1]):.2g}")
        print(f"y in {cys[0]:.2g} to {cys[-1]:.2g}")

    cxs_all = []
    cys_all = []

    for i, cy in enumerate(cys):
        if i % 2 == 0:
            cxs_all.append(cxs_a)
            cys_all.append(np.full(cxs_a.size, cy))
        else:
            cxs_all.append(cxs_b)
            cys_all.append(np.full(cxs_b.size, cy))

    cxs_all = np.concatenate(cxs_all)
    cys_all = np.concatenate(cys_all)
    cxys = np.vstack([cxs_all, cys_all]).T

    hdists = hexagon.get_hex_coords(ri).T
    hxys = np.moveaxis(np.tile(cxys, (6, 1, 1)), 0, 2) + hdists

    return cxys, hxys, cdist, area


def plot_grid(cxys, hxys=None, circledist=None, ax=None, scatter_kws=None, hex_kws=None, circledist_kws=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    ax.set_aspect('equal', 'box')

    if scatter_kws is None:
        scatter_kws = dict()

    if hex_kws is None:
        hex_kws = dict(alpha=0.7)

    if circledist_kws is None:
        circledist_kws = dict(fc='none')

    if scatter_kws.get('alpha', 0) > 0:
        ax.scatter(*cxys.T, **scatter_kws)

    if hxys is not None:
        # for hxy in hxys:
        #     ax.plot(np.append(hxy[0], hxy[0, 0]), np.append(hxy[1], hxy[1, 0]), **hex_kws)
        hxys = np.array(hxys)  # Ensure hxys is a NumPy array
        x = np.column_stack((hxys[:, 0], hxys[:, 0, 0]))
        y = np.column_stack((hxys[:, 1], hxys[:, 1, 0]))
        ax.plot(x.T, y.T, **hex_kws)

    if circledist is not None:
        for cxy in cxys:
            ax.add_patch(plt.Circle(cxy, circledist, **circledist_kws))

    return ax
