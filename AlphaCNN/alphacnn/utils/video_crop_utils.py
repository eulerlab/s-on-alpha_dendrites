import numpy as np
from scipy.interpolate import RectBivariateSpline

from alphacnn.utils import video_io_utils


def get_crop(video, xc: float, yc: float, width: float, height: float, src_pixel_size: float, trg_pixel_size: float):
    """Get crop of video. If pixel_size is equal in src and target will not interpolate"""

    src_npix_x, src_npix_y = video.shape[2], video.shape[1]
    src_xlim, src_ylim = video_io_utils.get_video_extent(video, src_pixel_size)

    src_xs = np.linspace(src_xlim[0], src_xlim[1], src_npix_x, endpoint=True)
    src_ys = np.linspace(src_ylim[0], src_ylim[1], src_npix_y, endpoint=True)

    if np.isclose(src_pixel_size, trg_pixel_size):
        return get_crop_same_scale(video, src_xs=src_xs, src_ys=src_ys, xc=xc, yc=yc, width=width, height=height,
                                   pixel_size=src_pixel_size)
    else:
        return get_crop_interpolation(video, src_xs=src_xs, src_ys=src_ys, xc=xc, yc=yc,
                                      width=width, height=height, trg_pixel_size=trg_pixel_size)


def get_crop_same_scale(video, src_xs, src_ys, xc: float, yc: float, width: float, height: float, pixel_size: float):
    x0 = np.argmin(np.abs(src_xs - (xc - width / 2)))
    x1 = int(np.round(x0 + width / pixel_size))
    y0 = np.argmin(np.abs(src_ys - (yc - height / 2)))
    y1 = int(np.round(y0 + height / pixel_size))
    return video[:, -y1:-y0, x0:x1].copy()


def get_crop_interpolation(video, src_xs, src_ys, xc: float, yc: float, width: float, height: float,
                           trg_pixel_size: float):
    if video.ndim == 3:
        video = np.expand_dims(video.copy(), axis=3)
        squeeze = True
    else:
        squeeze = False

    trg_xlim = (xc - width / 2, xc + width / 2)
    trg_ylim = (yc - height / 2, yc + height / 2)
    trg_xs = np.arange(trg_xlim[0], np.nextafter(trg_xlim[1], trg_xlim[1] + trg_pixel_size), trg_pixel_size)
    trg_ys = np.arange(trg_ylim[0], np.nextafter(trg_ylim[1], trg_ylim[1] + trg_pixel_size), trg_pixel_size)

    crop = np.zeros((video.shape[0], trg_ys.size, trg_xs.size, video.shape[3]), dtype=video.dtype)

    ix0, ix1 = np.argmax(src_xs > trg_xs[0]) - 1, np.argmin(src_xs < trg_xs[-1]) + 1
    iy0, iy1 = np.argmax(src_ys > trg_ys[0]) - 1, np.argmin(src_ys < trg_ys[-1]) + 1

    relevant_video = np.flip(video.astype(np.float32), axis=1)[:, iy0:iy1 + 1, ix0:ix1 + 1, :]
    relevant_xs = src_xs[ix0:ix1 + 1]
    relevant_ys = src_ys[iy0:iy1 + 1]

    assert relevant_xs[0] < trg_xs[0]
    assert relevant_xs[-1] > trg_xs[-1]

    assert relevant_ys[0] < trg_ys[0]
    assert relevant_ys[-1] > trg_ys[-1]

    for fi in range(relevant_video.shape[0]):
        for ci in range(relevant_video.shape[3]):
            crop[fi, :, :, ci] = (RectBivariateSpline(
                x=relevant_ys, y=relevant_xs, z=relevant_video[fi, :, :, ci])(trg_ys, trg_xs)).astype(video.dtype)

    if squeeze:
        crop = np.squeeze(crop, axis=3)

    return np.flip(crop, axis=1)


def video_to_mouse_gray_scale(video, znorm=True):
    assert video.ndim == 4, video.ndim
    video = (np.mean(video[:, :, :, :2], axis=-1) / 255.).copy().astype(np.float32)

    if znorm:
        video = (video - np.mean(video)) / np.std(video)

    return video
