import matplotlib.pyplot as plt
import numpy as np


class UnwarpGAM:
    def __init__(self, gam):
        self.gam = gam

    def unwarp_coords(self, xyz, scale_to_linestack=1.0):
        assert isinstance(scale_to_linestack, (float, int)) or np.array(scale_to_linestack).size == 3
        if xyz.ndim == 1:
            assert xyz.size == 3
            was_1d = True
        else:
            was_1d = False

        xyz = np.atleast_2d(xyz)
        assert xyz.shape[1] == 3, xyz.shape

        xyz_new = xyz.copy()
        xyz_new = xyz_new * scale_to_linestack
        xyz_new[:, 2] = xyz_new[:, 2] + self.gam.predict(X=xyz_new[:, :2])
        xyz_new = xyz_new / scale_to_linestack

        if was_1d:
            xyz_new = xyz_new.flatten()

        return xyz_new

    def unwarp_linestack(self, linestack):
        return apply_gam(self.gam, linestack)


def unwarp_linestack(linestack_warped, linestack_unwarped, plot=True):
    linestack_warped = linestack_to_binary(linestack_warped)
    linestack_unwarped = linestack_to_binary(linestack_unwarped)

    if plot:
        fig = plot_linestacks(linestack_warped, linestack_unwarped)
        fig.suptitle('Warped vs. unwarped')
        plt.show()

        fig = plot_z_projection_difference(linestack_warped, linestack_unwarped)
        fig.suptitle('Warped vs. unwarped - z-projection')
        plt.show()

    gam, x, y, z = fit_gam(linestack_warped, linestack_unwarped)

    if plot:
        fig = plot_gam_fit(gam, x, y, z, xmax=linestack_warped.shape[0], ymax=linestack_warped.shape[1])
        fig.suptitle('Warped vs. unwarped - GAM fit to differences')
        plt.show()

    linestack_unwarped_new = apply_gam(gam, linestack_warped)

    if plot:
        fig = plot_linestacks(linestack_unwarped, linestack_unwarped_new)
        fig.suptitle('Unwarped vs. reconstructed unwarped')
        plt.show()

        fig = plot_z_projection_difference(linestack_unwarped, linestack_unwarped_new)
        fig.suptitle('Unwarped vs. reconstructed unwarped - z-projection')
        plt.show()

    return linestack_unwarped, UnwarpGAM(gam)


def linestack_to_binary(linestack):
    linestack = np.nan_to_num(linestack.copy())
    assert np.unique(linestack).size == 2, np.unique(linestack)
    linestack = linestack / np.max(linestack)
    return linestack.astype(int)


def fit_gam(linestack_warped, linestack_unwarped, smooth_kw=None):
    """Fit GAM, either use all pixels, or if that is not possible, only the top pixels for each xy"""
    from pygam import LinearGAM, te

    x1, y1, z1 = np.where(linestack_warped > 0)
    x2, y2, z2 = np.where(linestack_unwarped > 0)

    if np.sum(linestack_warped) == np.sum(linestack_unwarped) and np.all(x1 == x2) and np.all(y1 == y2):
        zdiff = z2 - z1
        x, y = x1, y1
    else:
        x, y = np.where(np.max(linestack_warped, axis=2) > 0)

        z_depth_pixel_warped = np.argmax(linestack_warped, axis=2).astype(float)
        z_depth_pixel_unwarped = np.argmax(linestack_unwarped, axis=2).astype(float)
        z_depth_pixel_diff = z_depth_pixel_unwarped - z_depth_pixel_warped

        zdiff = z_depth_pixel_diff[x, y]

    if smooth_kw is None:
        smooth_kw = dict()
    gam = LinearGAM(te(0, 1, **smooth_kw))
    gam.gridsearch(np.stack([x, y]).T, zdiff)

    return gam, x, y, zdiff


def apply_gam(gam, linestack_warped):
    xls, yls, zls = np.where(linestack_warped)
    zls_unwarped_new = zls + gam.predict(X=np.stack([xls, yls]).T)

    linestack_unwarped = np.zeros_like(linestack_warped)
    linestack_unwarped[xls, yls, np.round(zls_unwarped_new).astype(int)] = 1
    return linestack_unwarped


def plot_linestacks(linestack1, linestack2):
    fig, axs = plt.subplots(2, 3, figsize=(15, 10), sharex='col', sharey='col')

    for i, ax_col in enumerate(axs.T):
        ax_col[0].set(title='Linestack 1')
        ax_col[0].imshow(np.max(linestack1, axis=i).T, aspect='auto', origin='lower')

        ax_col[1].set(title='Linestack 2')
        ax_col[1].imshow(np.max(linestack2, axis=i).T, aspect='auto', origin='lower')
    return fig


def plot_z_projection_difference(linestack1, linestack2):
    assert np.sum(linestack1) == np.sum(linestack2), 'Number of pixels do not match!'

    z_depth_pixel_1 = np.argmax(linestack1, axis=2).astype(float)
    x1, y1 = np.where(z_depth_pixel_1 > 0)
    z1 = z_depth_pixel_1[x1, y1]

    z_depth_pixel_2 = np.argmax(linestack2, axis=2).astype(float)
    x2, y2 = np.where(z_depth_pixel_2 > 0)
    z2 = z_depth_pixel_2[x2, y2]

    zdiff = z2 - z1

    zmin = np.min([np.min(z1), np.min(z2)])
    zmax = np.min([np.max(z1), np.max(z2)])

    fig, axs = plt.subplots(1, 3, figsize=(20, 5), sharex='all')

    for ax in axs:
        ax.axis('equal')

    axs[0].set(title='Linestack 1')
    im1 = axs[0].scatter(x=x1, y=y1, c=z1, cmap='jet', vmin=zmin, vmax=zmax)
    plt.colorbar(im1, ax=axs[0])

    axs[1].set(title='Linestack 2')
    im2 = axs[1].scatter(x=x2, y=y2, c=z2, cmap='jet', vmin=zmin, vmax=zmax)
    plt.colorbar(im2, ax=axs[1])

    axs[2].set(title='Linestack 2 - Linestack 1')
    sidx = np.argsort(np.abs(zdiff))
    im3 = axs[2].scatter(x=x1[sidx], y=y1[sidx], c=zdiff[sidx],
                         cmap='coolwarm', vmin=-np.max(np.abs(zdiff)), vmax=np.max(np.abs(zdiff)))
    plt.colorbar(im3, ax=axs[2])

    return fig


def plot_gam_fit(gam, x, y, z, xmax=None, ymax=None):
    if xmax is None:
        xmax = np.max(x)
    if ymax is None:
        ymax = np.max(y)

    xgrid = np.arange(0, xmax, 101)
    ygrid = np.arange(0, ymax, 101)
    xx, yy = np.meshgrid(xgrid, ygrid)

    zpred = gam.predict(X=np.stack([xx.flat, yy.flat]).T)
    zz = np.reshape(zpred, xx.shape)

    fig = plt.figure(figsize=(20, 10))

    angles = np.linspace(0, 360, 10, endpoint=False)
    for i, angle in enumerate(angles):
        ax = fig.add_subplot(2, int(angles.size / 2), i + 1, projection='3d')
        ax.scatter(x, y, z, c='r', marker='.')
        ax.plot_surface(xx, yy, zz, alpha=0.5)
        ax.view_init(30, angle + 10)

    return fig
