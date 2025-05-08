import warnings
from abc import abstractmethod
from copy import deepcopy

import datajoint as dj
import numpy as np
from djimaging.utils import math_utils

from djimaging.utils.trace_utils import sort_traces
from matplotlib import pyplot as plt

from djimaging.tables.receptivefield.rf_properties_utils import compute_gauss_srf_area, compute_surround_index, \
    fit_rf_model, compute_trf_transience_index, compute_half_amp_width, compute_main_peak_lag, compute_center_index
from djimaging.tables.receptivefield.rf_utils import compute_explained_rf, split_strf, \
    compute_polarity_and_peak_idxs
from djimaging.utils.dj_utils import get_primary_key
from djimaging.utils.plot_utils import plot_srf, plot_trf, plot_signals_heatmap, set_long_title, srf_extent


class SplitRFParamsTemplate(dj.Lookup):
    database = ""

    @property
    def definition(self):
        definition = """
        split_rf_params_id: int # unique param set id
        ---
        blur_std : float
        blur_npix : int unsigned
        upsample_srf_scale : int unsigned
        peak_nstd : float  # How many standard deviations does a peak need to be considered peak?
        npeaks_max : int unsigned # Maximum number of peaks, ignored if zero
        """
        return definition

    def add_default(self, skip_duplicates=False, **params):
        """Add default preprocess parameter to table"""
        key = dict(
            split_rf_params_id=1,
            blur_std=1.,
            blur_npix=1,
            upsample_srf_scale=0,
            peak_nstd=1,
            npeaks_max=0,
        )
        key.update(**params)
        self.insert1(key, skip_duplicates=skip_duplicates)


class SplitRFTemplate(dj.Computed):
    database = ""
    _max_dt_future = np.inf

    @property
    def definition(self):
        definition = '''
        # Compute basic receptive fields
        -> self.rf_table
        -> self.split_rf_params_table
        ---
        srf: longblob  # spatio receptive field
        trf: longblob  # temporal receptive field
        split_qidx : float  # Quality index as explained variance of the sRF tRF split between 0 and 1
        trf_peak_idxs : blob  # Indexes of peaks in tRF
        '''
        return definition

    @property
    def key_source(self):
        try:
            return self.rf_table.proj() * self.split_rf_params_table.proj()
        except (AttributeError, TypeError):
            pass

    @property
    @abstractmethod
    def rf_table(self):
        pass

    @property
    @abstractmethod
    def split_rf_params_table(self):
        pass

    def make(self, key):
        # Get data
        strf = (self.rf_table() & key).fetch1("rf")
        rf_time = self.fetch1_rf_time(key=key)

        # Get preprocess params
        blur_std, blur_npix, upsample_srf_scale, peak_nstd, npeaks_max = self.split_rf_params_table().fetch1(
            'blur_std', 'blur_npix', 'upsample_srf_scale', 'peak_nstd', 'npeaks_max')

        # Get tRF and sRF
        srf, trf, strf_fit = split_strf(
            strf, blur_std=blur_std, blur_npix=blur_npix, upsample_srf_scale=upsample_srf_scale)

        # Make tRF always positive, so that sRF reflects the polarity of the RF
        polarity, peak_idxs = compute_polarity_and_peak_idxs(
            rf_time=rf_time, trf=trf, nstd=peak_nstd, npeaks_max=npeaks_max if npeaks_max > 0 else None,
            max_dt_future=self._max_dt_future)

        if polarity == -1:
            srf *= -1
            trf *= -1

        split_qidx = compute_explained_rf(strf, strf_fit)

        # Save
        rf_key = deepcopy(key)
        rf_key['srf'] = srf
        rf_key['trf'] = trf
        rf_key['trf_peak_idxs'] = peak_idxs
        rf_key['split_qidx'] = split_qidx
        self.insert1(rf_key)

    def fetch1_rf_time(self, key):
        try:
            rf_time = (self.rf_table & key).fetch1('rf_time')
        except dj.DataJointError:
            model_dict = (self.rf_table & key).fetch1('model_dict')

            if "rf_time" in model_dict:
                rf_time = model_dict['rf_time']
            else:
                dt, rf = (self.rf_table & key).fetch1('dt', 'rf')
                n_t = rf.shape[0]
                shift = model_dict['shift']
                rf_time = (-np.arange(n_t) * dt - shift['stimulus'] * dt)[::-1]

        return rf_time

    def plot1(self, key=None):
        key = get_primary_key(table=self, key=key)

        rf_time = self.fetch1_rf_time(key=key)
        srf, trf, peak_idxs = (self & key).fetch1("srf", "trf", "trf_peak_idxs")

        fig, axs = plt.subplots(1, 2, figsize=(8, 3))
        ax = axs[0]
        plot_srf(srf, ax=ax)

        ax = axs[1]
        plot_trf(trf, t_trf=rf_time, peak_idxs=peak_idxs, ax=ax)

        plt.tight_layout()
        plt.show()

    def plot(self, restriction=None, sort=False):
        if restriction is None:
            restriction = dict()

        trf = math_utils.padded_vstack((self & restriction).fetch('trf'))

        if sort:
            trf = sort_traces(trf)

        ax = plot_signals_heatmap(signals=trf, figsize=(8, 20))
        ax.set(title='tRF')
        plt.show()


class FitGauss2DRFTemplate(dj.Computed):
    database = ""

    @property
    def definition(self):
        definition = """
        -> self.split_rf_table
        ---
        srf_fit: mediumblob
        srf_params: blob
        rf_area_um2: float # Area covered by 2 standard deviations
        rf_cdia_um: float # Circle equivalent diameter
        center_index: float # Weight and sign of center in sRF
        surround_index: float # Weight and sign of surround in sRF
        rf_qidx: float # Quality index as explained variance of the sRF estimation between 0 and 1
        """
        return definition

    _polarity = None

    @property
    def key_source(self):
        try:
            return self.split_rf_table.proj()
        except (AttributeError, TypeError):
            pass

    @property
    @abstractmethod
    def split_rf_table(self):
        pass

    @property
    @abstractmethod
    def stimulus_table(self):
        pass

    def fetch1_pixel_size(self, key):
        stim_dict = (self.stimulus_table() & key).fetch1("stim_dict")
        stim_pixel_size_x_um = stim_dict["pix_scale_x_um"]
        stim_pixel_size_y_um = stim_dict["pix_scale_y_um"]
        upsample_srf_scale = (self.split_rf_table.split_rf_params_table() & key).fetch1('upsample_srf_scale')
        pixel_size_x_um = stim_pixel_size_x_um / upsample_srf_scale if upsample_srf_scale > 1 else stim_pixel_size_x_um
        pixel_size_y_um = stim_pixel_size_y_um / upsample_srf_scale if upsample_srf_scale > 1 else stim_pixel_size_y_um

        return pixel_size_x_um, pixel_size_y_um

    def make(self, key):
        srf = (self.split_rf_table() & key).fetch1("srf")

        # Fit RF model
        srf_fit, srf_params, qi = fit_rf_model(srf, kind='gauss', polarity=self._polarity)

        # Pixel size
        pixel_size_x_um, pixel_size_y_um = self.fetch1_pixel_size(key=key)

        # Compute properties
        area, diameter = compute_gauss_srf_area(srf_params, pixel_size_x_um, pixel_size_y_um)
        center_index = compute_center_index(srf=srf, srf_center=srf_fit)
        surround_index = compute_surround_index(srf=srf, srf_center=srf_fit)

        # Save
        fit_key = deepcopy(key)
        fit_key['srf_fit'] = srf_fit
        fit_key['srf_params'] = srf_params
        fit_key['rf_cdia_um'] = diameter
        fit_key['rf_area_um2'] = area
        fit_key['center_index'] = center_index
        fit_key['surround_index'] = surround_index
        fit_key['rf_qidx'] = qi

        self.insert1(fit_key)

    def plot1(self, key=None):
        key = get_primary_key(table=self, key=key)
        srf = (self.split_rf_table() & key).fetch1("srf")
        srf_fit, rf_qidx = (self & key).fetch1("srf_fit", 'rf_qidx')
        pixel_size_x_um, pixel_size_y_um = self.fetch1_pixel_size(key=key)

        extent = srf_extent(srf.shape, pixel_size_x_um)
        vabsmax = np.maximum(np.max(np.abs(srf)), np.max(np.abs(srf_fit)))

        fig, axs = plt.subplots(1, 3, figsize=(10, 3))

        ax = axs[0]
        plot_srf(srf, ax=ax, vabsmax=vabsmax, extent=extent)
        ax.set_title('sRF')

        ax = axs[1]
        plot_srf(srf_fit, ax=ax, vabsmax=vabsmax, extent=extent)
        ax.set_title('sRF fit')

        ax = axs[2]
        plot_srf(srf - srf_fit, ax=ax, vabsmax=vabsmax, extent=extent)
        ax.set_title(f'Difference: QI={rf_qidx:.2f}')

        plt.tight_layout()
        plt.show()

    def plot(self, by=("preprocess_id",)):
        columns = ["rf_qidx", "rf_cdia_um", "center_index", "surround_index"]
        df = self.proj(*columns).fetch(format='frame').reset_index()
        for params, group in df.groupby(list(by)):
            print(by)
            print(params)
            group.hist(column=columns)
            plt.show()


class FitDoG2DRFTemplate(dj.Computed):
    database = ""

    @property
    def definition(self):
        definition = """
        -> self.split_rf_table
        ---
        srf_fit: mediumblob
        srf_center_fit: mediumblob
        srf_surround_fit: mediumblob
        srf_params: blob
        srf_eff_center: mediumblob
        srf_eff_center_params: blob
        rf_qidx: float
        rf_area_um2: float # Area covered by 2 standard deviations
        rf_cdia_um: float # Circle equivalent diameter
        center_index: float # Weight and sign of center in sRF
        surround_index: float # Weight and sign of surround in sRF
        """
        return definition

    _polarity = None

    @property
    def key_source(self):
        try:
            return self.split_rf_table.proj()
        except (AttributeError, TypeError):
            pass

    @property
    @abstractmethod
    def split_rf_table(self):
        pass

    @property
    @abstractmethod
    def stimulus_table(self):
        pass

    def fetch1_pixel_size(self, key):
        stim_dict = (self.stimulus_table() & key).fetch1("stim_dict")
        stim_pixel_size_x_um = stim_dict["pix_scale_x_um"]
        stim_pixel_size_y_um = stim_dict["pix_scale_y_um"]
        upsample_srf_scale = (self.split_rf_table.split_rf_params_table() & key).fetch1('upsample_srf_scale')
        pixel_size_x_um = stim_pixel_size_x_um / upsample_srf_scale if upsample_srf_scale > 1 else stim_pixel_size_x_um
        pixel_size_y_um = stim_pixel_size_y_um / upsample_srf_scale if upsample_srf_scale > 1 else stim_pixel_size_y_um

        return pixel_size_x_um, pixel_size_y_um

    def make(self, key, verbose=False):
        srf = (self.split_rf_table() & key).fetch1("srf")

        srf_fit, srf_center_fit, srf_surround_fit, srf_params, eff_polarity, qi = fit_rf_model(
            srf, kind='dog', polarity=self._polarity)

        if (self._polarity is None) or (eff_polarity == self._polarity):
            srf_eff_center = eff_polarity * np.clip(eff_polarity * srf_fit, 0, None)
        else:
            srf_eff_center = np.zeros_like(srf_fit)

        if np.allclose(srf_eff_center, 0.):
            if verbose:
                warnings.warn(f'Failed to fit DoG for key={key}')
            return

        center_index = compute_center_index(srf=srf, srf_center=srf_eff_center)
        surround_index = compute_surround_index(srf=srf, srf_center=srf_eff_center)

        # Compute area from gaussian fit to effective center
        srf_gauss_params = fit_rf_model(srf_eff_center, kind='gauss', polarity=eff_polarity,
                                        center=(srf_params['x_mean_0'], srf_params['y_mean_0']))[1]

        if srf_gauss_params is None:
            if verbose:
                warnings.warn(f'Failed to fit DoG center for key={key}')
            return

        pixel_size_x_um, pixel_size_y_um = self.fetch1_pixel_size(key=key)
        area, diameter = compute_gauss_srf_area(srf_gauss_params, pixel_size_x_um, pixel_size_y_um)

        # Save
        fit_key = deepcopy(key)
        fit_key['srf_fit'] = srf_fit
        fit_key['srf_center_fit'] = srf_center_fit
        fit_key['srf_surround_fit'] = srf_surround_fit
        fit_key['srf_params'] = srf_params
        fit_key['srf_eff_center'] = srf_eff_center
        fit_key['srf_eff_center_params'] = srf_gauss_params
        fit_key['rf_cdia_um'] = diameter
        fit_key['rf_area_um2'] = area
        fit_key['center_index'] = center_index
        fit_key['surround_index'] = surround_index
        fit_key['rf_qidx'] = qi

        self.insert1(fit_key)

    def plot1(self, key=None):
        key = get_primary_key(table=self, key=key)
        srf = (self.split_rf_table() & key).fetch1("srf")
        srf_fit, srf_center_fit, srf_surround_fit, srf_eff_center, rf_qidx = (self & key).fetch1(
            "srf_fit", 'srf_center_fit', 'srf_surround_fit', 'srf_eff_center', 'rf_qidx')

        vabsmax = np.maximum(np.max(np.abs(srf)), np.max(np.abs(srf_fit)))
        pixel_size_x_um, pixel_size_y_um = self.fetch1_pixel_size(key=key)
        extent = srf_extent(srf.shape, pixel_size_x_um)

        fig, axs = plt.subplots(2, 3, figsize=(10, 6))

        ax = axs[0, 0]
        plot_srf(srf, ax=ax, vabsmax=vabsmax, extent=extent)
        ax.set_title('sRF')

        ax = axs[0, 1]
        plot_srf(srf_fit, ax=ax, vabsmax=vabsmax, extent=extent)
        ax.set_title('sRF fit')

        ax = axs[0, 2]
        plot_srf(srf - srf_fit, ax=ax, vabsmax=vabsmax, extent=extent)
        ax.set_title(f'Difference: QI={rf_qidx:.2f}')

        ax = axs[1, 0]
        plot_srf(srf_center_fit, ax=ax, vabsmax=vabsmax, extent=extent)
        ax.set_title('sRF center fit')

        ax = axs[1, 1]
        plot_srf(srf_surround_fit, ax=ax, vabsmax=vabsmax, extent=extent)
        ax.set_title('sRF surround fit')

        ax = axs[1, 2]
        plot_srf(srf_eff_center, ax=ax, vabsmax=vabsmax, extent=extent)
        ax.set_title('sRF effective center')

        plt.tight_layout()
        plt.show()

    def plot(self, by=("preprocess_id",)):
        columns = ["rf_qidx", "rf_cdia_um", "center_index", "surround_index"]
        df = self.proj(*columns).fetch(format='frame').reset_index()
        for params, group in df.groupby(list(by)):
            print(by)
            print(params)
            group.hist(column=columns)
            plt.show()


class TempRFPropertiesTemplate(dj.Computed):
    database = ""
    _max_dt_future = np.inf

    @property
    def definition(self):
        definition = """
        -> self.split_rf_table
        ---
        transience_idx : float
        half_amp_width : float
        main_peak_lag : float
        """
        return definition

    @property
    def key_source(self):
        try:
            return self.split_rf_table.proj()
        except (AttributeError, TypeError):
            pass

    @property
    @abstractmethod
    def rf_table(self):
        pass

    @property
    @abstractmethod
    def split_rf_table(self):
        pass

    def fetch1_rf_time(self, key):
        try:
            rf_time = (self.rf_table & key).fetch1('rf_time')
        except dj.DataJointError:
            model_dict = (self.rf_table & key).fetch1('model_dict')

            if "rf_time" in model_dict:
                rf_time = model_dict['rf_time']
            else:
                dt, rf = (self.rf_table & key).fetch1('dt', 'rf')
                n_t = rf.shape[0]
                shift = model_dict['shift']
                rf_time = (-np.arange(n_t) * dt - shift['stimulus'] * dt)[::-1]

        return rf_time

    def make(self, key, plot=False):
        rf_time = self.fetch1_rf_time(key=key)
        trf, trf_peak_idxs = (self.split_rf_table & key).fetch1('trf', 'trf_peak_idxs')

        transience_idx = compute_trf_transience_index(
            rf_time, trf, trf_peak_idxs, max_dt_future=self._max_dt_future)
        half_amp_width = compute_half_amp_width(
            rf_time, trf, trf_peak_idxs, plot=plot, max_dt_future=self._max_dt_future)
        main_peak_lag = compute_main_peak_lag(
            rf_time, trf, trf_peak_idxs, plot=plot, max_dt_future=self._max_dt_future)

        key = key.copy()
        key['transience_idx'] = transience_idx if transience_idx is not None else -1.
        key['half_amp_width'] = half_amp_width if half_amp_width is not None else -1.
        key['main_peak_lag'] = main_peak_lag if main_peak_lag is not None else -1.

        self.insert1(key)

    def plot(self):
        fig, axs = plt.subplots(1, 3, figsize=(12, 3))
        for ax, name in zip(axs, ['transience_idx', 'half_amp_width', 'main_peak_lag']):
            ax.hist(self.fetch(name))
            ax.set_title(name)
        plt.tight_layout()
        return fig, axs

    def plot1(self, key=None):
        key = get_primary_key(table=self, key=key)

        rf_time = self.fetch1_rf_time(key=key)
        trf, peak_idxs = (self.split_rf_table & key).fetch1('trf', 'trf_peak_idxs')
        transience_idx, half_amp_width, main_peak_lag = (self & key).fetch1(
            'transience_idx', 'half_amp_width', 'main_peak_lag')

        fig, ax = plt.subplots(1, 1, figsize=(6, 2.5))
        plot_trf(trf, t_trf=rf_time, peak_idxs=peak_idxs, ax=ax)
        ax.axvline(-main_peak_lag, color='k', ls='--', label='main_peak')
        ax.axvline(-main_peak_lag - half_amp_width / 2, color='gray', ls=':', label='width')
        ax.axvline(-main_peak_lag + half_amp_width / 2, color='gray', ls=':', label='_')
        ax.axvline(0, color='gray', ls='-', label='_', zorder=-2)
        set_long_title(ax=ax, title=key, fontsize=8)
        plt.legend()
        plt.tight_layout()
        return fig, ax
