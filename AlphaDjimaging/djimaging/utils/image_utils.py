import numpy as np
from matplotlib import pyplot as plt

from djimaging.utils.math_utils import normalize_zero_one


def resize_image(image, output_shape: tuple, order: int = 0):
    from skimage.transform import resize
    image = image.copy().astype(float)
    finite_mask = np.isfinite(image)
    image[~finite_mask] = np.nanmin(image)
    image_resized = resize(
        image, output_shape=output_shape, order=order, mode='constant', anti_aliasing=False)
    mask_resized = resize(
        finite_mask.astype(int), output_shape=output_shape, order=0, mode='constant', anti_aliasing=False)
    assert image_resized.shape == mask_resized.shape
    image_resized[mask_resized == 0] = np.nan
    return image_resized


def rotate_image(image, angle, order=1, cval=None):
    from skimage.transform import rotate
    finite_mask = np.isfinite(image)
    image = image.copy()

    minv = np.nanmin(image)
    image[~finite_mask] = minv
    image_rotated = rotate(image, angle=angle, resize=True, order=order, cval=minv if cval is None else cval)
    mask_rotated = rotate(finite_mask.astype(int), angle=angle, resize=True, order=0, cval=0)
    assert image_rotated.shape == mask_rotated.shape
    image_rotated[mask_rotated == 0] = np.nan if cval is None else cval
    return image_rotated


def rescale_image(image, scale, order=0):
    if not hasattr(scale, '__iter__'):
        scale = (scale, scale)
    elif len(scale) == 2:
        scale = (scale[0], scale[1])

    if image.ndim == 3 and len(scale) == 2:
        scale = (scale[0], scale[1], 1)

    output_shape = np.ceil(np.asarray(image.shape) * np.asarray(scale)).astype('int')
    resized_image = resize_image(image, output_shape=output_shape, order=order)
    return resized_image


def color_image(data_img, cmap='viridis', gamma=1.0, alpha=255):
    assert data_img.ndim == 2
    assert isinstance(alpha, int)

    # convert to float if data_img is not a float
    if not np.issubdtype(data_img.dtype, np.floating):
        data_img = data_img.astype(np.float32)

    color_img = (plt.get_cmap(cmap)(normalize_zero_one(data_img) ** gamma) * 255).astype(int)
    color_img[:, :, -1] = alpha
    return color_img


def int_rescale_image(data_img, scale: int):
    if scale > 1:
        return int_upscale_image(data_img, scale)
    elif scale < -1:
        return int_downscale_image(data_img, abs(scale))
    else:
        return data_img


def int_upscale_image(data_img, upscale: int):
    return np.repeat(np.repeat(data_img, upscale, axis=0), upscale, axis=1)


def int_downscale_image(data_img, downscale: int):
    """Take mean of blocks of size upscale"""
    data_img = data_img.copy()
    # data_img = data_img[::downscale][:, ::downscale]
    # Take mean of blocks of size downscale
    data_img = np.mean(
        np.mean(
            data_img.reshape(
                data_img.shape[0] // downscale, downscale, data_img.shape[1] // downscale, downscale, -1,
            ),
            axis=1
        ),
        axis=2
    )

    return data_img
