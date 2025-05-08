import os
import warnings
from abc import abstractmethod

import datajoint as dj
import numpy as np
import pandas as pd

from djimaging.user.alpha.tables.morphology.match_utils import (
    get_rec_center_in_stack_coordinates, match_rec_to_stack,
    create_template_from_recording, add_soma_to_linestack, calibrate_one_roi, find_roi_pos_stack,
    compute_roi_pos_metrics, plot_match_template_to_image)
from djimaging.user.alpha.tables.morphology.morphology_roi_utils import compute_dendritic_distance_to_soma, \
    plot_roi_positions_xyz
from djimaging.utils.data_utils import load_h5_data, load_h5_table
from djimaging.utils.dj_utils import get_primary_key, make_hash
from djimaging.utils.scanm_utils import get_setup_xscale, extract_roi_idxs


class RoiStackPosParamsTemplate(dj.Lookup):
    database = ""

    @property
    def definition(self):
        definition = """
        roi_pos_params_hash: varchar(32) # unique param set hash
        ---
        fig_folder='' : varchar(191)
        pad_scale=1.0 : float
        pad_more=50 : int unsigned
        dist_score_factor=1e-3 :float
        soma_radius=10 :float
        match_max_dist_um = 50.0 : float
        match_z_scale = 0.5 : float
        """
        return definition

    def add_default(self, skip_duplicates=False, **update_kw):
        """Add default preprocess parameter to table"""
        key = dict()
        key.update(**update_kw)
        key["roi_pos_params_hash"] = make_hash(key)
        self.insert1(key, skip_duplicates=skip_duplicates)


class FieldStackPosTemplate(dj.Computed):
    database = ''

    @property
    def definition(self):
        definition = """
        -> self.field_table
        -> self.params_table
        ---
        rec_c_warning_flag: tinyint unsigned
        rec_cpos_stack_xy_raw: longblob
        rec_cpos_stack_xyz: longblob
        rec_cpos_stack_fit_dist: float
        """
        return definition

    @property
    def key_source(self):
        try:
            return (self.field_table().RoiMask() * self.params_table()).proj()
        except (AttributeError, TypeError):
            pass

    @property
    @abstractmethod
    def userinfo_table(self):
        pass

    @property
    @abstractmethod
    def experiment_table(self):
        pass

    @property
    @abstractmethod
    def field_table(self):
        pass

    @property
    @abstractmethod
    def linestack_table(self):
        pass

    @property
    @abstractmethod
    def params_table(self):
        pass

    @property
    @abstractmethod
    def morph_table(self):
        pass

    @property
    @abstractmethod
    def template_table(self):
        pass

    class RoiStackPos(dj.Part):
        @property
        def definition(self):
            definition = """
            -> master
            -> self.roi_table
            ---
            roi_pos_stack_xyz : blob
            """
            return definition

        @property
        @abstractmethod
        def roi_table(self): pass

    class FitInfo(dj.Part):
        @property
        def definition(self):
            definition = """
            -> master
            ---
            score: float
            score_map : longblob
            linestack_xy : longblob
            template_raw : longblob
            template_fit : longblob
            roi_coords_rot : longblob
            exp_center : blob
            xyz_fit : blob
            rescale=1.0 : float
            """
            return definition

    def _fetch_and_compute(self, key, pad_more=200, shift=(0, 0), dist_score_factor=None, pad_linestack=0):

        exp_key = key.copy()
        exp_key.pop('field')

        data_stack_name = (self.userinfo_table() & key).fetch1('data_stack_name')
        rec_filepath, npixartifact = (self.field_table() & key).fetch1('fromfile', 'npixartifact')
        wparams_rec = load_h5_table('wParamsNum', filename=rec_filepath)

        if len((self.template_table & key).proj()) > 0:
            template = (self.template_table & key).fetch1('roi_mask')
            template = (template < 0).astype(float)
            template[:npixartifact, :] = np.nan

        else:
            template = create_template_from_recording(load_h5_data(rec_filepath),
                                                      npixartifact=npixartifact, data_stack_name=data_stack_name)

        linestack, stack_filepath = (self.field_table() * self.linestack_table() & exp_key).fetch1(
            'linestack', 'fromfile')
        wparams_stack = load_h5_table('wParamsNum', filename=stack_filepath)

        pixel_size_um_xy, pixel_size_um_z = \
            ((self.field_table & "z_stack_flag=1") * self.linestack_table() & exp_key).fetch1(
                'pixel_size_um', 'z_step_um')
        pixel_sizes_stack = np.array([pixel_size_um_xy, pixel_size_um_xy, pixel_size_um_z])

        roi_mask = (self.field_table().RoiMask() & key).fetch1('roi_mask')

        setup_xscale = get_setup_xscale((self.experiment_table().ExpInfo() & key).fetch1('setupid'))

        soma_xyz = (self.morph_table() & exp_key).fetch1('soma_xyz')

        params = (self.params_table() & key).fetch1()
        fig_folder = params.pop('fig_folder')
        soma_radius = params.pop('soma_radius')

        if fig_folder is not None:
            os.makedirs(fig_folder, exist_ok=True)

            savefilename = fig_folder + '/'
            for k, v in key.items():
                if 'hash' not in k:
                    savefilename += f'{k}_{v}__'

            savefilename = savefilename[:-2] + '.png'

        else:
            savefilename = None

        from IPython.display import clear_output
        clear_output(wait=True)

        rec_cx_stack, rec_cy_stack, rec_c_warning_flag = get_rec_center_in_stack_coordinates(
            wparams_rec=wparams_rec, wparams_stack=wparams_stack,
            linestack=linestack, pixel_sizes_stack=pixel_sizes_stack)

        _dist_score_factor = 0. if rec_c_warning_flag == 1 else params['dist_score_factor']
        if dist_score_factor is not None:
            _dist_score_factor = dist_score_factor

        roi_idxs, rois_pos_stack_xyz, rec_cpos_stack_xyz, fit_dict = match_rec_to_stack(
            ch_average=template, roi_mask=roi_mask, setup_xscale=setup_xscale,
            wparams_rec=wparams_rec, wparams_stack=wparams_stack, pixel_sizes_stack=pixel_sizes_stack,
            linestack=linestack, rec_cxy_stack=np.array([rec_cx_stack, rec_cy_stack]),
            soma_xyz=soma_xyz, soma_radius=soma_radius, pad_linestack=pad_linestack,
            angle_adjust=0, pad_scale=params['pad_scale'], shift=shift,
            pad_more=params['pad_more'] + pad_more if rec_c_warning_flag == 1 else params['pad_more'],
            dist_score_factor=_dist_score_factor,
            rescales=None, savefilename=savefilename, seed=42)

        assert set(roi_idxs) == set(np.abs(extract_roi_idxs(roi_mask)))
        assert len(rois_pos_stack_xyz) == len(roi_idxs)

        pos_key = key.copy()
        pos_key['rec_c_warning_flag'] = int(rec_c_warning_flag)
        pos_key['rec_cpos_stack_xy_raw'] = np.array([rec_cx_stack, rec_cy_stack]).astype(np.float32)
        pos_key['rec_cpos_stack_xyz'] = np.asarray(rec_cpos_stack_xyz).astype(np.float32)
        pos_key['rec_cpos_stack_fit_dist'] = \
            np.sum((pos_key['rec_cpos_stack_xy_raw'] - pos_key['rec_cpos_stack_xyz'][:2]) ** 2) ** 0.5

        info_key = key.copy()
        info_key['score'] = fit_dict['score']
        info_key['score_map'] = fit_dict['score_map']
        info_key['linestack_xy'] = fit_dict['linestack_xy']
        info_key['template_raw'] = fit_dict['template_raw']
        info_key['template_fit'] = fit_dict['template_fit']
        info_key['roi_coords_rot'] = fit_dict['roi_coords_rot']
        info_key['exp_center'] = fit_dict['exp_center']
        info_key['xyz_fit'] = fit_dict['xyz_fit']
        info_key['rescale'] = fit_dict['rescale']

        roi_keys = []
        for roi_idx, roi_pos_stack_xyz in zip(roi_idxs, rois_pos_stack_xyz):
            roi_key = key.copy()
            roi_key['roi_id'] = int(abs(roi_idx))

            artifact_flag = (self.RoiStackPos().roi_table() & roi_key).fetch1('artifact_flag')

            if artifact_flag == 0:
                roi_key['roi_pos_stack_xyz'] = np.asarray(roi_pos_stack_xyz).astype(np.float32)
                roi_keys.append(roi_key)

        return pos_key, info_key, roi_keys

    def make(self, key, pad_more=200, shift=(0, 0), dist_score_factor=None, pad_linestack=0):
        pos_key, info_key, roi_keys = self._fetch_and_compute(
            key, pad_more=pad_more, shift=shift, dist_score_factor=dist_score_factor, pad_linestack=pad_linestack)

        self.insert1(pos_key)
        self.FitInfo().insert1(info_key)
        for roi_key in roi_keys:
            self.RoiStackPos().insert1(roi_key)

    def plot1(self, key=None, savefilename=None):
        key = get_primary_key(table=self, key=key)

        fit_dict = (self.FitInfo() & key).fetch1()

        score = fit_dict['score']
        score_map = fit_dict['score_map']
        linestack_xy = fit_dict['linestack_xy']
        xyz_fit = fit_dict['xyz_fit']
        exp_center = fit_dict['exp_center']
        template_fit = fit_dict['template_fit']
        template_raw = fit_dict['template_raw']
        roi_coords_rot = fit_dict['roi_coords_rot']

        plot_match_template_to_image(
            image=linestack_xy, template_fit=template_fit,
            score_map=score_map, best_xy=xyz_fit[:2], best_score=score,
            template_raw=template_raw, roi_coords_rot=roi_coords_rot,
            exp_center=exp_center, savefilename=savefilename)


class FieldPathPosTemplate(dj.Computed):
    database = ''

    @property
    def definition(self):
        definition = """
        -> self.fieldstackpos_table
        ---
        field_path_pos_x : float
        field_path_pos_y : float
        field_path_pos_z : float
        """
        return definition

    @property
    def key_source(self):
        try:
            return self.fieldstackpos_table.proj()
        except (AttributeError, TypeError):
            pass

    @property
    @abstractmethod
    def fieldstackpos_table(self):
        pass

    @property
    @abstractmethod
    def linestack_table(self):
        pass

    @property
    @abstractmethod
    def field_table(self):
        pass

    def make(self, key):
        exp_key = key.copy()
        exp_key.pop('field')

        pixel_size_um_xy, pixel_size_um_z = \
            ((self.field_table & "z_stack_flag=1") * self.linestack_table() & exp_key).fetch1(
                'pixel_size_um', 'z_step_um')
        pixel_sizes_stack = np.array([pixel_size_um_xy, pixel_size_um_xy, pixel_size_um_z])

        field_xyz_stack = (self.fieldstackpos_table & key).fetch1('rec_cpos_stack_xyz')
        field_path_pos_xyz = field_xyz_stack * pixel_sizes_stack

        self.insert1(dict(**key,
                          field_path_pos_x=field_path_pos_xyz[0],
                          field_path_pos_y=field_path_pos_xyz[1],
                          field_path_pos_z=field_path_pos_xyz[2]))


class FieldCalibratedStackPosTemplate(dj.Computed):
    database = ''

    @property
    def definition(self):
        definition = """
        -> self.fieldstackpos_table
        ---
        success_rate: float
        """
        return definition

    @property
    def key_source(self):
        try:
            return self.fieldstackpos_table.proj()
        except (AttributeError, TypeError):
            pass

    @property
    @abstractmethod
    def fieldstackpos_table(self):
        pass

    @property
    @abstractmethod
    def linestack_table(self):
        pass

    @property
    @abstractmethod
    def field_table(self):
        pass

    @property
    @abstractmethod
    def morph_table(self):
        pass

    @property
    @abstractmethod
    def params_table(self):
        pass

    class RoiCalibratedStackPos(dj.Part):
        @property
        def definition(self):
            definition = """
            -> master
            -> self.roi_table
            ---
            roi_cal_pos_stack_xyz : blob
            success_cal_flag : tinyint unsigned
            soma_by_dist_flag : tinyint unsigned
            """
            return definition

        @property
        @abstractmethod
        def roi_table(self):
            pass

    def make(self, key):
        exp_key = key.copy()
        exp_key.pop('field')

        linestack, stack_filepath = (self.field_table() * self.linestack_table() & exp_key).fetch1(
            'linestack', 'fromfile')

        pixel_size_um_xy, pixel_size_um_z = \
            ((self.field_table & "z_stack_flag=1") * self.linestack_table() & exp_key).fetch1(
                'pixel_size_um', 'z_step_um')
        pixel_sizes_stack = np.array([pixel_size_um_xy, pixel_size_um_xy, pixel_size_um_z])

        soma_xyz = (self.morph_table() & exp_key).fetch1('soma_xyz')
        soma_radius, z_scale, max_dist = (self.params_table() & key).fetch1(
            'soma_radius', 'match_z_scale', 'match_max_dist_um')

        soma_linestack = add_soma_to_linestack(
            np.zeros_like(linestack), pixel_sizes_stack,
            soma_xyz, radius_xyz=soma_radius, fill_value=1)

        z_factor = z_scale * pixel_sizes_stack[2] / pixel_sizes_stack[0]

        linestack_coords_xyz = np.vstack(np.where(linestack)).T
        soma_linestack_coords_xyz = np.vstack(np.where(soma_linestack)).T

        soma_stack_xyz = calibrate_one_roi(
            soma_xyz / pixel_sizes_stack, linestack_coords_xyz, z_factor=z_factor, return_dist=False)

        rois_stack_pos = (self.fieldstackpos_table().RoiStackPos() & key).fetch(as_dict=True)

        # Calibrate
        roi_keys = []
        for roi_dict in rois_stack_pos:
            quality, roi_cal_pos_stack_xyz, soma_by_dist = find_roi_pos_stack(
                roi_raw_xyz=roi_dict['roi_pos_stack_xyz'],
                linestack_coords_xyz=linestack_coords_xyz,
                soma_linestack_coords_xyz=soma_linestack_coords_xyz,
                max_dist=max_dist, z_factor=z_factor)

            if soma_by_dist:
                roi_cal_pos_stack_xyz = soma_stack_xyz

            roi_key = key.copy()
            roi_key['roi_id'] = roi_dict['roi_id']
            roi_key['roi_cal_pos_stack_xyz'] = np.asarray(roi_cal_pos_stack_xyz).astype(np.float32)
            roi_key['soma_by_dist_flag'] = abs(int(soma_by_dist))
            roi_key['success_cal_flag'] = int(quality)

            roi_keys.append(roi_key)

        success_rate = np.mean([roi_key['success_cal_flag'] for roi_key in roi_keys])

        # Insert
        field_key = key.copy()
        field_key['success_rate'] = success_rate

        self.insert1(field_key)
        for roi_key in roi_keys:
            self.RoiCalibratedStackPos().insert1(roi_key)


class FieldPosMetricsTemplate(dj.Computed):
    database = ''

    @property
    def definition(self):
        definition = """
        -> self.fieldcalibratedstackpos_table
        ---
        ref_roi_id : int  # Reference median ROI
        d_dist_to_soma : float  # Median (of ROIs) distance to soma 
        norm_d_dist_to_soma : float  # Median (of ROIs) normalized distance to soma
        ec_dist_to_soma : float  # Median (of ROIs) distance to soma
        max_d_dist: float
        """
        return definition

    @property
    def key_source(self):
        try:
            return self.fieldcalibratedstackpos_table.proj()
        except (AttributeError, TypeError):
            pass

    @property
    @abstractmethod
    def fieldcalibratedstackpos_table(self):
        pass

    @property
    @abstractmethod
    def morph_table(self):
        pass

    class RoiPosMetrics(dj.Part):
        @property
        def definition(self):
            definition = """
           -> master
           -> self.roi_table
           ---
           path_id : int
           loc_on_path : blob
           roi_pos_xyz : blob
           roi_pos_xyz_rel_to_soma : blob
           d_dist_to_soma : float
           norm_d_dist_to_soma : float
           ec_dist_to_soma : float
           ec_dist_to_density_center : float
           branches_to_soma : blob
           num_branches_to_soma : int
           """
            return definition

        @property
        @abstractmethod
        def roi_table(self):
            pass

    def make(self, key):
        exp_key = key.copy()
        exp_key.pop('field')

        roi_ids, roi_pos_stack_xyzs = (self.fieldcalibratedstackpos_table().RoiCalibratedStackPos() & key).fetch(
            'roi_id', 'roi_cal_pos_stack_xyz')
        soma_xyz, density_center, df_paths = (self.morph_table() & exp_key).fetch1(
            'soma_xyz', 'density_center', 'df_paths')

        df_paths = pd.DataFrame(df_paths)

        max_d_dist = np.max([compute_dendritic_distance_to_soma(df_paths, i) for i in df_paths.index])

        roi_keys = []
        for roi_id, roi_pos_stack_xyz in zip(roi_ids, roi_pos_stack_xyzs):
            try:
                roi_metrics = compute_roi_pos_metrics(roi_pos_stack_xyz, df_paths, soma_xyz, density_center, max_d_dist)
            except IndexError:
                warnings.warn(f'Failed to compute_roi_pos_metrics for roi_id={roi_id} for key={key} '
                              f'because linestack was padded too much')
                continue
            except:
                raise Exception(f'Failed to compute_roi_pos_metrics for roi_id={roi_id} for key={key}')

            roi_key = key.copy()
            roi_key['roi_id'] = roi_id
            roi_key.update(**roi_metrics)
            roi_keys.append(roi_key)

        # Get reference ROI and params
        d_dists = np.array([roi_key['d_dist_to_soma'] for roi_key in roi_keys])
        ref_idx = np.argmin(np.abs(d_dists - np.median(d_dists)))

        # Insert
        field_key = key.copy()
        field_key['ref_roi_id'] = roi_keys[ref_idx]['roi_id']
        field_key['d_dist_to_soma'] = roi_keys[ref_idx]['d_dist_to_soma']
        field_key['norm_d_dist_to_soma'] = roi_keys[ref_idx]['norm_d_dist_to_soma']
        field_key['ec_dist_to_soma'] = roi_keys[ref_idx]['ec_dist_to_soma']
        field_key['max_d_dist'] = max_d_dist

        self.insert1(field_key)
        for roi_key in roi_keys:
            self.RoiPosMetrics().insert1(roi_key)

    def plot1(self, key=None):
        key = get_primary_key(self, key)

        exp_key = key.copy()
        exp_key.pop('field')

        rois_pos_xyz = np.stack((self.RoiPosMetrics() & exp_key).fetch('roi_pos_xyz')).T
        d_dist_to_soma = (self.RoiPosMetrics() & exp_key).fetch('d_dist_to_soma')
        df_paths = pd.DataFrame((self.morph_table() & exp_key).fetch1('df_paths'))
        soma_xyz = (self.morph_table() & exp_key).fetch1('soma_xyz')
        max_d_dist = np.max((self & exp_key).fetch('max_d_dist'))

        fig, axs = plot_roi_positions_xyz(
            rois_pos_xyz, df_paths.path, soma_xyz, c=d_dist_to_soma, layer_on_z=None, layer_off_z=None,
            vmax=max_d_dist, plot_rois=True, plot_morph=True, plot_soma=True,
            xlim=None, ylim=None, zlim=None)
        return fig, axs


class FieldRoiPosMetricsTemplate(dj.Computed):
    database = ''

    @property
    def definition(self):
        definition = """
        # Copies from FieldPosMetricsTemplate for easy access
        -> self.field_table
        -> self.fieldposmetrics_table.proj(roi_field='field')
        ---
        ref_roi_id : int  # Reference median ROI
        d_dist_to_soma : float  # Median (of ROIs) distance to soma 
        norm_d_dist_to_soma : float  # Median (of ROIs) normalized distance to soma
        ec_dist_to_soma : float  # Median (of ROIs) distance to soma
        max_d_dist: float
        """
        return definition

    @property
    def key_source(self):
        try:
            return self.field_table.proj() & (self.roi_kind_table & "roi_kind='field'")
        except (AttributeError, TypeError):
            pass

    @property
    @abstractmethod
    def fieldposmetrics_table(self):
        pass

    @property
    @abstractmethod
    def roi_kind_table(self):
        pass

    @property
    @abstractmethod
    def field_table(self):
        pass

    def make(self, key):
        key_roi = key.copy()
        key_roi['field'] = key_roi['field'].replace('FieldROI', '')

        if len(self.fieldposmetrics_table & key_roi) != 1:
            warnings.warn(
                f'{len(self.fieldposmetrics_table & key_roi)} entries in fieldposmetrics_table for key={key_roi}')
            return

        entry = (self.fieldposmetrics_table & key_roi).fetch1()
        entry['field'] = key['field']
        entry['roi_field'] = key_roi['field']
        self.insert1(entry)
