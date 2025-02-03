import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from alphaanalysis.plot import plot_paths, plot_srf_outline


def plot_rois_on_morph(
        ax, cell, highlight_list, fit_kind,
        morph_tab, rf_tab, roi_pos_metrics_tab, roi_cal_stack_pos_tab,
        line_stack_tab, field_stack_pos_tab, field_tab, experiment_tab,
        restr=None, annotate=True, plot_srf_contours=False, plot_srf_offset=False, plot_rf_rois_only=False,
        path_kws=None, roi_kws=None, offset_kws=None, rf_contour_kws=None, cmap='plasma', add_colorbar=False,
        palettes=None, pixel_size_um=30, max_d_dist=None,
        stim_offset_xy_um=(-15, -15), add_offset_rf_xy_um='auto'):
    date, exp_num = cell
    key = dict(date=date, exp_num=exp_num)

    if restr is None:
        restr = dict()

    # fetch
    paths = pd.DataFrame((morph_tab & key).fetch1('df_paths')).path
    soma_xyz = (morph_tab & key).fetch1('soma_xyz')

    rois_tab = (roi_pos_metrics_tab & key & restr)
    if plot_rf_rois_only:
        rois_tab &= rf_tab

    field_ids, roi_ids, rois_pos_xyz = (roi_pos_metrics_tab & rois_tab.proj() & key & restr).fetch(
        'field', 'roi_id', 'roi_pos_xyz')
    rois_pos_xyz = np.stack(rois_pos_xyz.T)

    # plot
    _offset_kws = dict(c='c', lw=0.8, ls='-')
    if offset_kws is not None:
        _offset_kws.update(offset_kws)

    _path_kws = dict(lw=0.3, clip_on=False)
    if path_kws is not None:
        _path_kws.update(path_kws)

    plot_paths(ax=ax, paths=paths, soma_xyz=soma_xyz, **_path_kws)

    _, field_nums = np.unique(field_ids, return_inverse=True)
    field_nums += 1

    _roi_kws = dict(marker='.', ms=4, c='c', mec='b', alpha=0.9, mew=0.5, zorder=200)
    if roi_kws is not None:
        _roi_kws.update(roi_kws)

    _rf_contour_kws = dict(ls='-', lw=1, alpha=0.5)
    if rf_contour_kws is not None:
        _rf_contour_kws.update(rf_contour_kws)

    for field_num, roi_id, (x, y, z) in zip(field_nums, roi_ids, rois_pos_xyz):
        ax.plot(x, y, label=f'field_num={field_num}' if roi_id == roi_ids[field_nums == field_num][0] else None,
                **_roi_kws)

    for j, (field_id, roi_id) in enumerate(highlight_list):
        key = [dict(date=date, exp_num=exp_num, roi_id=roi_id, field=field_id)
               for field_id in [f'd{field_id}', f'D{field_id}']]
        x, y, z = (rois_tab & key).fetch1('roi_pos_xyz')

        ax.plot(x, y, marker='o', ms=4, c=palettes[cell][j + 1], zorder=10000, mec='dimgray', alpha=0.9)

        if annotate:
            ax.annotate(
                xy=(x, y), text=j + 1, c='k', xytext=np.array([x + 25, y + 25]), va='center',
                bbox=dict(boxstyle="round", fc=(1, 1, 1, 0.75), ec=(0, 0, 0, 0.5)),
                arrowprops=dict(arrowstyle="->", color='none'), zorder=10000)

    sm = None

    stack_pixel_size_um = ((field_tab & "z_stack_flag=1") * line_stack_tab & (experiment_tab & key)).fetch1(
        'pixel_size_um')

    restricted_rf_tab = rf_tab * field_stack_pos_tab * roi_cal_stack_pos_tab * roi_pos_metrics_tab & key & restr
    srf_list, field_cpos_stack_xyz_list, roi_cpos_stack_xyz_list, d_dist_list = restricted_rf_tab.fetch(
        'srf', 'rec_cpos_stack_xyz', 'roi_cal_pos_stack_xyz', 'd_dist_to_soma')
    srf_outlines = fetch_srf_outlines(key, restricted_rf_tab, fit_kind=fit_kind)

    if plot_srf_contours or plot_srf_offset:
        if isinstance(cmap, str):
            if max_d_dist is None:
                max_d_dist = roi_pos_metrics_tab.fetch('d_dist_to_soma').max()

            cmapper = sns.color_palette(cmap, as_cmap=True)
            sm = plt.cm.ScalarMappable(cmap=cmapper, norm=plt.Normalize(vmin=0, vmax=max_d_dist))
            sm.set_array([])

            if add_colorbar:
                from mpl_toolkits.axes_grid1.inset_locator import inset_axes

                cbar = plt.colorbar(sm, cax=inset_axes(ax,
                                                       width="5%", height="50%", loc='center left',
                                                       bbox_to_anchor=(0.9, 0., 1, 1),
                                                       bbox_transform=ax.transAxes,
                                                       borderpad=0,
                                                       ))
                cbar.ax.set_ylabel('Dist. to soma [µm]', rotation=90)
        else:
            cmapper = cmap

        for j, (srf, srf_outline, field_cpos_stack_xyz, roi_cpos_stack_xyz, d_dist) in enumerate(zip(
                srf_list, srf_outlines, field_cpos_stack_xyz_list, roi_cpos_stack_xyz_list, d_dist_list)):

            field_center = field_cpos_stack_xyz[:2] * stack_pixel_size_um
            roi_center = roi_cpos_stack_xyz[:2] * stack_pixel_size_um
            center_offset_um = field_center + np.asarray(stim_offset_xy_um) + np.asarray(add_offset_rf_xy_um)

            if cmapper is not None:
                _rf_contour_kws['color'] = cmapper(d_dist / max_d_dist)

            srf_fit_center = plot_srf_outline(
                ax, srf=srf, srf_outline=srf_outline,
                pixel_size_um=pixel_size_um, center_offset_um=center_offset_um,
                plot_outline=plot_srf_contours, plot_center=False, outline_kw=_rf_contour_kws)

            if plot_srf_offset:
                ax.plot([roi_center[0], srf_fit_center[0]], [roi_center[1], srf_fit_center[1]], **_offset_kws)

    ax.set_aspect('equal', 'box')
    ax.set(xticks=[], yticks=[])
    ax.axis('off')

    return sm


def fetch_srf_outlines(key, rf_tab, fit_kind):
    if 'dog' in fit_kind:
        srf_outlines = (rf_tab & key).fetch('srf_eff_center_params')
    elif fit_kind == 'gauss':
        srf_outlines = (rf_tab & key).fetch('srf_params')
    elif fit_kind == 'contour':
        srf_contours = (rf_tab & key).fetch('srf_contours')
        srf_outlines = np.array([srf_contour[list(srf_contour.keys())[0]][0] for srf_contour in srf_contours],
                                dtype=object)
    else:
        raise ValueError(f'Unknown fit_kind: {fit_kind}')
    return srf_outlines
