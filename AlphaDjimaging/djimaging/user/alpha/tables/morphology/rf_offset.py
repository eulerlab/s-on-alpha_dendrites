from abc import abstractmethod

import datajoint as dj
import numpy as np
from djimaging.tables.receptivefield import FitDoG2DRFTemplate
from djimaging.user.alpha.tables.rf_contours.srf_contour import RFContoursTemplate
from djimaging.user.alpha.tables.rf_contours.srf_contour_utils import compute_cntr_center
from matplotlib import pyplot as plt


class RfOffsetTemplate(dj.Computed):
    database = ''

    _stimulus_offset_dx_um = 0.
    _stimulus_offset_dy_um = 0.

    @property
    def definition(self):
        definition = """
        # Computes distance to center
        -> self.rf_fit_tab
        ---
        rf_dx_um: float
        rf_dy_um: float
        rf_d_um: float
        """
        return definition

    @property
    def key_source(self):
        try:
            return self.rf_fit_tab.proj()
        except (AttributeError, TypeError):
            pass

    @property
    @abstractmethod
    def stimulus_tab(self):
        pass

    @property
    @abstractmethod
    def rf_split_tab(self):
        pass

    @property
    @abstractmethod
    def rf_fit_tab(self):
        pass

    def fetch1_pixel_size(self, key):
        pixel_size_x_um, pixel_size_y_um = self.rf_fit_tab().fetch1_pixel_size(key)
        if not np.isclose(pixel_size_x_um, pixel_size_y_um):
            raise ValueError("Pixel size is not isotropic")
        return pixel_size_x_um

    def fetch_and_compute(self, key):
        """Compute offset w.r.t. stimulus center in microns"""
        pixel_size_um = self.fetch1_pixel_size(key)

        if isinstance(self.rf_fit_tab(), RFContoursTemplate):
            srf_contours = (self.rf_fit_tab & key).fetch1('srf_contours')
            levels = (self.rf_fit_tab.rf_contours_params_table & key).fetch1('levels')
            if len(srf_contours) == 0:
                return np.nan, np.nan
            elif len(srf_contours[levels[0]]) == 0:
                return np.nan, np.nan
            else:
                rf_dx_um, rf_dy_um = compute_cntr_center(srf_contours[levels[0]][0])
                return rf_dx_um, rf_dy_um

        else:
            srf = (self.rf_split_tab & key).fetch1('srf')
            if isinstance(self.rf_fit_tab(), FitDoG2DRFTemplate):
                srf_params = (self.rf_fit_tab & key).fetch1('srf_eff_center_params')
            else:
                srf_params = (self.rf_fit_tab & key).fetch1('srf_params')

            # Compute offset w.r.t. stimulus center.
            # Plus one half to get pixel center, e.g. if the RF is centered on bottom left pixel,
            # the fit will be 0, 0. For a 2x2 stimulus the offset is half a pixel: 0.5 - 2 / 2 = -0.5
            rf_dx_um = ((srf_params['x_mean'] + 0.5) - (srf.shape[1] / 2)) * pixel_size_um
            rf_dy_um = ((srf_params['y_mean'] + 0.5) - (srf.shape[0] / 2)) * pixel_size_um

            return rf_dx_um, rf_dy_um

    def make(self, key):
        # Position of sRF center in stimulus coordinates
        rf_dx_um, rf_dy_um = self.fetch_and_compute(key)

        # Corrected for stimulus offset if any
        rf_dx_um += self._stimulus_offset_dx_um
        rf_dy_um += self._stimulus_offset_dy_um

        if not np.isfinite(rf_dx_um) or not np.isfinite(rf_dy_um):
            return

        rf_d_um = (rf_dx_um ** 2 + rf_dy_um ** 2) ** 0.5
        self.insert1(dict(**key, rf_dx_um=rf_dx_um, rf_dy_um=rf_dy_um, rf_d_um=rf_d_um))

    def plot(self, exp_key):
        rf_dx_um, rf_dy_um = (self & exp_key).fetch('rf_dx_um', 'rf_dy_um')

        fig, ax = plt.subplots(1, 1, figsize=(5, 5))

        for i, (rf_dx_um_i, rf_dy_um_i) in enumerate(zip(rf_dx_um, rf_dy_um)):
            ax.plot([0, rf_dx_um_i], [0, rf_dy_um_i], '-', c=f'C{i % 10}', alpha=0.4)

        return fig, ax


'''
class RfRoiOffsetTemplate(dj.Computed):
    database = ''

    @property
    def definition(self):
        definition = """
        -> self.rf_fit_tab
        -> self.roi_cal_stack_pos_tab
        ---
        rf_dx_um: float
        rf_dy_um: float
        rf_x_um : float
        rf_y_um : float
        """
        return definition

    @property
    def key_source(self):
        try:
            return self.rf_fit_tab.proj() * self.roi_cal_stack_pos_tab.proj()
        except (AttributeError, TypeError):
            pass

    @property
    @abstractmethod
    def stimulus_tab(self):
        pass

    @property
    @abstractmethod
    def field_tab(self):
        pass

    @property
    @abstractmethod
    def rf_split_tab(self):
        pass

    @property
    @abstractmethod
    def rf_fit_tab(self):
        pass

    @property
    @abstractmethod
    def field_stack_pos_tab(self):
        pass

    @property
    @abstractmethod
    def roi_cal_stack_pos_tab(self):
        pass

    def make(self, key):
        exp_key = key.copy()
        exp_key.pop('field')

        stim_pixel_size_um = (self.stimulus_tab & key).fetch1("stim_dict")['pix_scale_x_um']

        srf = (self.rf_split_tab & key).fetch1('srf')

        try:
            srf_params = (self.rf_fit_tab & key).fetch1('srf_eff_center_params')
        except:
            srf_params = (self.rf_fit_tab & key).fetch1('srf_params')

        srfc_x = srf_params.get('x_mean_0', srf_params['x_mean'])
        srfc_y = srf_params.get('y_mean_0', srf_params['y_mean'])

        stack_pixel_size_um = (self.field_tab() & "z_stack_flag=1" & exp_key).fetch1('pixel_size_um')

        field_xyz_stack = (self.field_stack_pos_tab & key).fetch1('rec_cpos_stack_xyz')
        field_xyz_um = field_xyz_stack * stack_pixel_size_um

        roi_xyz_stack = (self.roi_cal_stack_pos_tab & key).fetch1('roi_cal_pos_stack_xyz')
        roi_xyz_um = roi_xyz_stack * stack_pixel_size_um

        # Corrects for shift of dense noise, and for shift during fitting of sRF
        # These two effect actually cancel, but for clarity we keep them separate
        srfc_x_um = (-(srf.shape[1] + 1) / 2 + (srfc_x + 0.5)) * stim_pixel_size_um + field_xyz_um[0]
        srfc_y_um = (-(srf.shape[0] + 1) / 2 + (srfc_y + 0.5)) * stim_pixel_size_um + field_xyz_um[1]

        rf_dx_um = srfc_x_um - roi_xyz_um[0]
        rf_dy_um = srfc_y_um - roi_xyz_um[1]

        self.insert1(dict(**key, rf_dx_um=rf_dx_um, rf_dy_um=rf_dy_um, rf_x_um=srfc_x_um, rf_y_um=srfc_y_um))

    def plot(self, exp_key):
        rf_x_um, rf_y_um, rf_dx_um, rf_dy_um = (self & exp_key).fetch('rf_x_um', 'rf_y_um', 'rf_dx_um', 'rf_dy_um')

        fig, ax = plt.subplots(1, 1, figsize=(5, 5))

        for i, (rf_x_um_i, rf_y_um_i, rf_dx_um_i, rf_dy_um_i) in enumerate(zip(rf_x_um, rf_y_um, rf_dx_um, rf_dy_um)):
            ax.plot(rf_x_um_i, rf_y_um_i, 'o', label='rf', c=f'C{i % 10}')
            ax.plot([rf_x_um_i, rf_x_um_i - rf_dx_um_i], [rf_y_um_i, rf_y_um_i - rf_dy_um_i],
                    '-', c=f'C{i % 10}', alpha=0.4)

        return fig, ax
'''
