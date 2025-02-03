import numpy as np
from matplotlib import pyplot as plt


def plot_frame(ax, frame, pixel_size=1., zero_center=False, extent=None, add_colorbar=False, cmap=None, **kwargs):
    if not ((frame.ndim == 2) or ((frame.ndim == 3) and (frame.shape[2] == 3))):
        raise ValueError(f"frame must be either 2D or 3D with last dim size 3, got {frame.shape}")
    cmap = cmap or ('gray' if frame.ndim == 2 else None)

    h, w = frame.shape
    if zero_center:
        assert extent is None
        extent = (-w * pixel_size / 2, w * pixel_size / 2, -h * pixel_size / 2, h * pixel_size / 2)
    elif extent is None:
        extent = (0, w * pixel_size, 0, h * pixel_size)

    im = ax.imshow(frame, aspect='equal', cmap=cmap, extent=extent, **kwargs)
    if add_colorbar:
        plt.colorbar(im, ax=ax, use_gridspec=True)
    return im


def get_video_cmap_params(video, sym=False, cmap=None):
    if sym:
        vabsmax = np.max(np.abs(video))
        vmin, vmax = -vabsmax, +vabsmax
        cmap = cmap or 'bwr'
    else:
        vmin, vmax = np.min(video), np.max(video)
        cmap = cmap or 'gray'

    if video.ndim == 4 and video.shape[3] == 3:
        cmap = None

    return vmin, vmax, cmap


def plot_video_frames(video, fis=None, pixel_size=1., zero_center=False, extent=None,
                      n_rows=2, n_cols=4, axs=None, add_colorbar=True, sym=False, cmap=None, **kwargs):
    if axs is None:
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 2.5), sharex='all', sharey='all')

    axs = axs.flatten()
    if fis is None:
        fis = np.linspace(0, video.shape[0] - 1, len(axs), endpoint=True).astype(int)

    vmin, vmax, cmap = get_video_cmap_params(video, sym=sym, cmap=cmap)

    for ax, fi in zip(axs, fis):
        ax.set_title(f"frame={fi}")
        plot_frame(ax, frame=video[fi], pixel_size=pixel_size, zero_center=zero_center, extent=extent,
                   vmin=vmin, vmax=vmax, add_colorbar=add_colorbar and (fi == fis[-1]), cmap=cmap,
                   **kwargs)

    return axs, fis


def plot_target_positions(target_pos, fis=None, n_past=10, n_rows=2, n_cols=4, axs=None, **kwargs):
    if axs is None:
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 2.5), sharex='all', sharey='all')

    axs = axs.flatten()
    if fis is None:
        fis = np.linspace(0, target_pos.shape[0] - 1, len(axs), endpoint=True).astype(int)

    for ax, fi in zip(axs, fis):
        ax.set_title(f"frame={fi}")
        if n_past > 0:
            fi_past = np.maximum(fi - n_past, 0)
            ax.plot(target_pos[fi_past:fi + 1, 0], target_pos[fi_past:fi + 1, 1], '-', alpha=0.6, zorder=10,
                    **kwargs)
        ax.plot(target_pos[fi, 0], target_pos[fi, 1], 'o', mfc='none', mew=3, alpha=0.9, zorder=10, **kwargs)
    return axs, fis
