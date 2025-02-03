import os

import numpy as np
import skvideo.io
from matplotlib import pyplot as plt


def load_video(video_path, flip_left=False):
    """Load video"""
    src_video = skvideo.io.vread(video_path)
    if ('left' in os.path.split(video_path)[-1].lower()) and flip_left:
        src_video = np.flip(src_video, axis=2)

    return src_video


def load_fps(video_path):
    """Load frames per second from video meta data"""
    metadata = skvideo.io.ffprobe(video_path)['video']

    meta_rate = metadata['@r_frame_rate']

    split_rate = [float(n) for n in meta_rate.split('/')]
    fps = split_rate[0] / split_rate[1]

    if np.isclose(fps, int(fps)):
        fps = int(fps)

    return fps


def get_video_extent(video, pixel_size):
    """Get extent of video, define center as (0, 0)"""
    src_npix_x, src_npix_y = video.shape[2], video.shape[1]
    xlim = (-src_npix_x / 2 * pixel_size, src_npix_x / 2 * pixel_size)
    ylim = (-src_npix_y / 2 * pixel_size, src_npix_y / 2 * pixel_size)
    return xlim, ylim


def save_to_mp4(arr: np.ndarray, fps: int, outputfile: str, input_kw=None, output_kw=None, norm=False,
                xy_upsample: int = 0, cmap=None):
    """Save numpy array to mp4 file"""
    if norm:
        arr = arr.astype(np.float32)
        arr = (arr - np.percentile(arr, q=1)) / (np.percentile(arr, q=99) - np.percentile(arr, q=1))
        arr = np.clip(arr, 0, 1)

    if cmap is not None:
        colormap = plt.get_cmap(cmap)
        arr = colormap(np.stack([frame for frame in arr], axis=0))

    if np.max(arr) <= 1.:
        arr = arr * 255.

    arr = arr.astype(np.uint8)

    if xy_upsample > 1:
        arr = np.repeat(np.repeat(arr, xy_upsample, axis=1), xy_upsample, axis=2)

    inputdict = {'-r': f"{fps / 1}"}
    if input_kw is not None:
        inputdict.update(input_kw)

    outputdict = {'-vcodec': 'ffv1'}
    if output_kw is not None:
        outputdict.update(output_kw)

    if not outputfile.endswith('.mp4'):
        outputfile = outputfile + '.mp4'

    writer = skvideo.io.FFmpegWriter(outputfile)
    for frame in arr:
        writer.writeFrame(frame)

    writer.close()
