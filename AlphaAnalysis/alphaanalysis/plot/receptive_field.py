import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse
from matplotlib import pyplot as plt

from alphaanalysis.plot import plot_paths, offsets, plot_convex_hull

from djimaging.utils.math_utils import normalize


def fetch_and_plot_srf_and_morph(
        key, roi_kind, fit_kind,
        rf_tab, morph_tab, roi_pos_tab, exp_tab, field_tab,
        pixel_size_um=30, srf=None, srf_outline=None,
        blur_std=1, blur_npix=2, upsample_srf_scale=5, srf_norm_kind='amp_one',
        ax=None, plot_im=True, plot_center=True, plot_outline=True, plot_morph=True, plot_roi=True,
        outline_kw=None, add_offset_rf=(0, 0), fit_srf_xlim=False,
        hull_points=None, hull_kws=None
):
    """Fetch and plot sRF and morphology on the same axis"""

    if srf is None:
        srf = (rf_tab & key).fetch1('srf')

    if (plot_center or plot_outline) and srf_outline is None:
        srf_outline = fetch_srf_outline(key, rf_tab, fit_kind)

    if plot_morph:
        df_paths, soma_xyz = (morph_tab & key).fetch1('df_paths', 'soma_xyz')
        df_paths = pd.DataFrame(df_paths)
    else:
        df_paths = None
        soma_xyz = None

    if roi_kind != 'roi':
        key = key.copy()
        key['field'] = key['field'][:2]

    # Compute offsets
    stack_pixel_size_um, stack_nx, stack_ny = (
            (field_tab & "z_stack_flag=1") & (exp_tab & key).proj()).fetch1('pixel_size_um', 'nxpix', 'nypix')
    field_cpos_stack_xy = (roi_pos_tab & key).fetch1('rec_cpos_stack_xyz')[:2]

    if roi_kind == 'roi':
        roi_cpos_stack_xy = (roi_pos_tab & key).fetch1('roi_cal_pos_stack_xyz')[:2]
    else:
        roi_cpos_stack_xy = None

    stack_offset_um, morph_field_offset_um, roi_field_offset_um = offsets.get_offsets(
        stack_nx, stack_ny, stack_pixel_size_um, field_cpos_stack_xy, roi_cpos_stack_xy)

    if roi_kind != 'roi':
        roi_field_offset_um = np.zeros(2)

    ax = plot_soma_srf_and_morph(
        srf, srf_outline, df_paths, soma_xyz,
        stack_offset_um, morph_field_offset_um, roi_field_offset_um,
        ax=ax, stim_pixel_size_um=pixel_size_um,
        upsample_srf_scale=upsample_srf_scale, blur_std=blur_std, blur_npix=blur_npix,
        plot_im=plot_im, plot_center=plot_center, plot_outline=plot_outline, plot_morph=plot_morph, plot_roi=plot_roi,
        outline_kw=outline_kw, cmap='bwr', add_offset_rf=add_offset_rf, fit_srf_xlim=fit_srf_xlim,
        srf_norm_kind=srf_norm_kind,
        hull_points=hull_points, hull_kws=hull_kws
    )

    return ax


def fetch_srf_outline(key, rf_tab, fit_kind):
    if 'dog' in fit_kind:
        srf_outline = (rf_tab & key).fetch1('srf_eff_center_params')
    elif fit_kind == 'gauss':
        srf_outline = (rf_tab & key).fetch1('srf_params')
    elif fit_kind == 'contour':
        srf_contours = (rf_tab & key).fetch1('srf_contours')
        srf_outline = srf_contours[list(srf_contours.keys())[0]][0]
    else:
        raise ValueError(f'Unknown fit_kind: {fit_kind}')
    return srf_outline


def plot_soma_srf_and_morph(
        srf, srf_outline, df_paths, soma_xyz, stack_offset_um, morph_field_offset_um, roi_field_offset_um,
        ax=None, stim_pixel_size_um=30, cmap='bwr',
        upsample_srf_scale=0, blur_std=0, blur_npix=0, srf_norm_kind='none',
        outline_kw=None, plot_im=True, plot_center=True, plot_outline=True, plot_morph=True, plot_roi=True,
        add_offset_rf=(0, 0), fit_srf_xlim=False,
        hull_points=None, hull_kws=None,
):
    """
    To align sRF and morphology the following must be matched:
    - The center of sRF is the center of the recording field
    - The stack and the recording field have an offset
    """
    stack_offset_um = np.asarray(stack_offset_um)
    morph_field_offset_um = np.asarray(morph_field_offset_um)
    roi_field_offset_um = np.asarray(roi_field_offset_um)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.set_aspect('equal', 'box')

    plot_srf(
        ax=ax, srf=srf, srf_outline=srf_outline, vabsmax='auto',
        pixel_size_um=stim_pixel_size_um, cmap=cmap,
        upsample_srf_scale=upsample_srf_scale, blur_std=blur_std, blur_npix=blur_npix,
        norm_kind=srf_norm_kind,
        outline_kw=outline_kw, plot_outline=plot_outline, plot_center=plot_center, plot_im=plot_im,
        center_offset_um=(add_offset_rf[0] - 15, add_offset_rf[1] - 15),
        fix_xlim=fit_srf_xlim,
    )

    if plot_morph:
        plot_paths(
            ax=ax, paths=df_paths.path, soma_xyz=soma_xyz, lw=0.3, clip_on=False,
            offset=morph_field_offset_um - stack_offset_um)

    if hull_points is not None:
        plot_convex_hull(ax, hull_points, offset_um=(morph_field_offset_um - stack_offset_um), hull_kw=hull_kws)

    if plot_roi:
        ax.plot(*roi_field_offset_um, marker='.', ms=6, mfc='c', mec='b', lw=0.5, alpha=0.8)

    return ax


def plot_srf(
        ax, srf=None, srf_outline=None, vabsmax=None, n_std=2,
        pixel_size_um=30, marker_kw=None, outline_kw=None, cmap='coolwarm',
        upsample_srf_scale=0, blur_std=0, blur_npix=0,
        center_offset_um=(-15, -15),
        norm_kind='none',
        plot_outline=False, plot_center=False, plot_im=True, fix_xlim=False):
    """Plot sRF and outline"""

    if plot_center or plot_outline:
        if srf_outline is None:
            raise ValueError('srf_outline must be provided to plot sRF center or outline')

        plot_srf_outline(ax, srf, pixel_size_um, srf_outline, center_offset_um,
                         plot_outline=plot_outline, plot_center=plot_center, n_std=n_std, outline_kw=outline_kw,
                         marker_kw=marker_kw)

    if plot_im:
        if srf is None:
            raise ValueError('srf must be provided to plot sRF image')

        plot_srf_image(
            ax, srf, center_offset_um=center_offset_um, pixel_size_um=pixel_size_um,
            cmap=cmap, vabsmax=vabsmax, fix_xlim=fix_xlim,
            upsample_srf_scale=upsample_srf_scale, blur_std=blur_std, blur_npix=blur_npix,
            norm_kind=norm_kind)


def plot_srf_outline(ax, srf, pixel_size_um, srf_outline, center_offset_um,
                     flip_y=False, plot_outline=True, plot_center=False,
                     n_std=2, outline_kw=None, marker_kw=None):
    from djimaging.user.alpha.tables.rf_contours.srf_contour_utils import compute_cntr_center

    w_um = srf.shape[1] * pixel_size_um
    h_um = srf.shape[0] * pixel_size_um

    if isinstance(srf_outline, dict):

        dx = (srf_outline['x_mean'] + 0.5) * pixel_size_um
        dy = (srf_outline['y_mean'] + 0.5) * pixel_size_um
        theta = srf_outline['theta']

        if flip_y:
            theta = -theta
            dy = -dy

        srf_fit_center = (
            dx - w_um / 2 + center_offset_um[0],
            dy - h_um / 2 + center_offset_um[1],
        )

        if plot_outline:
            plot_srf_ellipse(
                ax, srf_fit_center, x_stddev=srf_outline['x_stddev'], y_stddev=srf_outline['y_stddev'], n_std=n_std,
                theta=theta, pixel_size_um=pixel_size_um, outline_kw=outline_kw)

        if plot_center:
            plot_srf_center(ax, srf_fit_center, marker_kw=marker_kw)
    else:
        assert srf_outline.shape[1] == 2
        if flip_y:
            srf_outline = srf_outline * np.array([+1, -1])

        srf_fit_center = compute_cntr_center(srf_outline)

        if plot_outline:
            plot_srf_contour(ax, srf_outline, center_offset_um, outline_kw=outline_kw)
        if plot_center:
            plot_srf_center(ax, srf_fit_center, marker_kw=marker_kw)

    return srf_fit_center


def plot_srf_ellipse(ax, xy_center, x_stddev, y_stddev, theta, pixel_size_um, n_std=2, outline_kw=None):
    """"Plot sRF ellipse outline"""
    plot_outline_kw = dict(lw=1, color='purple', ls='-')
    if outline_kw is not None:
        plot_outline_kw.update(outline_kw)

    ax.add_patch(Ellipse(
        xy=xy_center,
        width=n_std * 2 * x_stddev * pixel_size_um,
        height=n_std * 2 * y_stddev * pixel_size_um,
        angle=np.rad2deg(theta), fill=False, **plot_outline_kw))


def plot_srf_contour(ax, srf_outline, offset=(0, 0), outline_kw=None):
    """Plot contour line"""
    plot_outline_kw = dict(lw=1, color='purple', ls='-')
    if outline_kw is not None:
        plot_outline_kw.update(outline_kw)
    xs = srf_outline[:, 0] + offset[0]
    ys = srf_outline[:, 1] + offset[1]
    ax.plot(xs, ys, **plot_outline_kw)


def plot_srf_center(ax, xy_center, marker_kw=None):
    plot_marker_kw = dict(color='green', ms=3, zorder=100, marker='o')
    if marker_kw is not None:
        plot_marker_kw.update(marker_kw)
    ax.plot(xy_center[0], xy_center[1], **plot_marker_kw)


def plot_srf_image(ax, srf, center_offset_um=(0, 0), pixel_size_um=1., cmap='coolwarm', vabsmax=None,
                   upsample_srf_scale=0, blur_std=0, blur_npix=0, norm_kind='none', fix_xlim=False):
    from djimaging.tables.receptivefield.rf_utils import resize_srf, smooth_rf

    w_um = srf.shape[1] * pixel_size_um
    h_um = srf.shape[0] * pixel_size_um

    extent = (
        -w_um / 2 + center_offset_um[0], w_um / 2 + center_offset_um[0],
        -h_um / 2 + center_offset_um[1], h_um / 2 + center_offset_um[1]
    )

    if blur_npix > 0:
        srf = smooth_rf(rf=srf, blur_std=blur_std, blur_npix=blur_npix)

    if upsample_srf_scale > 1:
        srf = resize_srf(srf, scale=upsample_srf_scale)

    if norm_kind != 'none':
        srf = normalize(srf, norm_kind=norm_kind)

    if vabsmax == 'auto':
        if norm_kind == 'amp_one':
            vabsmax = 1
        else:
            vabsmax = np.max(np.abs(srf))

    im = ax.imshow(
        srf, vmin=-vabsmax, vmax=vabsmax, cmap=cmap, zorder=0, extent=extent, origin='lower', interpolation='none')

    if fix_xlim:
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])

    return im
