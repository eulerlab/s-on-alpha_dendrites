import numpy as np
from matplotlib import animation, pyplot as plt

from alphacnn.visualize.plot_stimulus import plot_frame, get_video_cmap_params


def array_to_anim(video: np.ndarray, fps=60, cbar=True, axis_off=True, xy_upsample=0, pos=None,
                  pixel_size=1., zero_center=False, extent=None, cmap=None, cmap_sym=False) \
        -> animation.FuncAnimation:
    assert video.ndim >= 3
    vmin, vmax, cmap = get_video_cmap_params(video, sym=cmap_sym, cmap=cmap)

    if xy_upsample > 1:
        video = np.repeat(np.repeat(video, xy_upsample, axis=1), xy_upsample, axis=2)

    video_ratio = video.shape[1] / video.shape[2]

    fig, ax = plt.subplots(1, 1, figsize=(5 + int(cmap is not None), np.clip(5 * video_ratio, 2, 10)))

    im = plot_frame(ax, frame=video[0], pixel_size=pixel_size, zero_center=zero_center, extent=extent,
                    vmin=vmin, vmax=vmax, add_colorbar=cbar, cmap=cmap)

    if pos is not None:
        ps = ax.plot(pos[0, 0], pos[0, 1], 'go', mfc='none', ms=20, mew=3)

    if axis_off:
        ax.axis('off')

    plt.close()

    def init():
        im.set_data(video[0])
        if pos is not None:
            ps[0].set_data(pos[0, 0], pos[0, 1])

    def animate(i):
        im.set_data(video[i])
        if pos is not None:
            ps[0].set_data(pos[i, 0], pos[i, 1])

    anim = animation.FuncAnimation(
        fig, animate, init_func=init, frames=video.shape[0], interval=1000. / fps)
    return anim


def plot_video_frames(video, positions=None):
    fig, axs = plt.subplots(3, 6, figsize=(12, 6), sharex='all', sharey='all')
    axs = axs.flat

    fis = np.arange(0, video.shape[0], np.maximum(1, video.shape[0] // len(axs)))
    for ax, frame, fi in zip(axs, video[fis], fis):
        ax.set_title(fi)
        ax.imshow(frame, aspect='equal')

        if positions is not None:
            # TODO: implement
            raise NotImplementedError

    plt.tight_layout()
