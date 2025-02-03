from abc import abstractmethod
from copy import deepcopy

import datajoint as dj
import numpy as np
from djimaging.utils.dj_utils import get_primary_key
from djimaging.utils.math_utils import truncated_vstack
from matplotlib import pyplot as plt


class WbgSpotsTemplate(dj.Computed):
    database = ""
    _stim_name = "vspots"
    _t_base = 1
    _t_peak_a = 1.0
    _t_peak_b = 2.0
    _t_max = 4.0

    @property
    def definition(self):
        definition = '''
        -> self.group_snippets_table
        ---
        n_reps : int
        w_qidx : float
        b_qidx : float
        g_qidx : float
        w_pref_size : float
        b_pref_size : float
        g_pref_size : float
        col_pref : enum('W', 'B', 'G')
        w_spot_surround_index : float
        b_spot_surround_index : float
        g_spot_surround_index : float
        data_dict : longblob
        '''
        return definition

    @property
    @abstractmethod
    def group_snippets_table(self):
        pass

    @property
    @abstractmethod
    def presentation_table(self):
        pass

    @property
    def key_source(self):
        try:
            return self.group_snippets_table().proj() & dict(stim_name=self._stim_name)
        except (AttributeError, TypeError):
            pass

    def compute_entry(self, key, plot=False):
        snippets = (self.group_snippets_table() & key).fetch1('snippets')
        fs = (self.presentation_table().ScanInfo & key).fetch1('scan_frequency')

        idx_base = int(self._t_base * fs)
        idx_peak_a = int(self._t_peak_a * fs)
        idx_peak_b = int(self._t_peak_b * fs)

        snippets = deepcopy(snippets)

        # Remove incomplete traces and subtract baselines
        for name, snippet in snippets.items():
            for i, trace in enumerate(snippet.T):
                if not np.all(np.isfinite(trace[:idx_peak_b])):
                    trace[:] = np.nan
                else:
                    baseline = np.median(trace[0:idx_base])
                    trace -= baseline

        # Compute baselines and mean spot responses
        snippets_peaks = {}
        for name, snippet in snippets.items():
            snippets_peaks[name] = []
            for i, trace in enumerate(snippet.T):
                peak = np.mean(trace[idx_peak_a:idx_peak_b])
                snippets_peaks[name].append(peak)

        # Get max response for each rep
        rep_peaks = np.ones(4)
        for rep in range(4):
            for name, snippet in snippets.items():
                if len(snippets_peaks[name]) > rep and np.isfinite(snippets_peaks[name][rep]):
                    rep_peaks[rep] = np.maximum(rep_peaks[rep], snippets_peaks[name][rep])

        # Normalize snippets and peaks
        snippets_norm = deepcopy(snippets)
        snippets_peaks_norm = deepcopy(snippets_peaks)
        for name, snippet in snippets.items():
            for i, trace in enumerate(snippet.T):
                snippets_norm[name][:, i] = trace / rep_peaks[i]
                snippets_peaks_norm[name][i] = snippets_peaks[name][i] / rep_peaks[i]

        tuning_curves = {}

        sizes = [100, 200, 300, 400, 600, 1000]

        n_reps = 0

        for col in ['W', 'B', 'G']:
            col_means = np.full(len(sizes), np.nan)
            for i, size in enumerate(sizes):
                col_means[i] = np.nanmean(snippets_peaks_norm[col + str(size).zfill(4)])
                n_reps = np.maximum(n_reps, np.sum(np.isfinite(snippets_peaks_norm[col + str(size).zfill(4)])))

            tuning_curves[col] = col_means

        w_pref_idx = np.nanargmax(tuning_curves['W'])
        w_pref_size = sizes[w_pref_idx]
        w_spots_surround_index = np.clip(tuning_curves['W'][-1] / tuning_curves['W'][w_pref_idx], 0, None) - 1

        b_pref_idx = np.nanargmax(tuning_curves['B'])
        b_pref_size = sizes[b_pref_idx]
        b_spots_surround_index = np.clip(tuning_curves['B'][-1] / tuning_curves['B'][b_pref_idx], 0, None) - 1

        g_pref_idx = np.nanargmax(tuning_curves['G'])
        g_pref_size = sizes[g_pref_idx]
        g_spots_surround_index = np.clip(tuning_curves['G'][-1] / tuning_curves['G'][g_pref_idx], 0, None) - 1

        col_pref = 'B' if tuning_curves['B'][b_pref_idx] > tuning_curves['G'][g_pref_idx] else 'G'

        w_q_snips = truncated_vstack([tr for tr in snippets['W0300'].T if np.any(np.isfinite(tr))] +
                                     [tr for tr in snippets['W0400'].T if np.any(np.isfinite(tr))]).T

        b_q_snips = truncated_vstack([tr for tr in snippets['B0300'].T if np.any(np.isfinite(tr))] +
                                     [tr for tr in snippets['B0400'].T if np.any(np.isfinite(tr))]).T

        g_q_snips = truncated_vstack([tr for tr in snippets['G0300'].T if np.any(np.isfinite(tr))] +
                                     [tr for tr in snippets['G0400'].T if np.any(np.isfinite(tr))]).T

        w_qidx = np.var(np.mean(w_q_snips, axis=1)) / np.mean(np.var(w_q_snips, axis=0))
        b_qidx = np.var(np.mean(b_q_snips, axis=1)) / np.mean(np.var(b_q_snips, axis=0))
        g_qidx = np.var(np.mean(g_q_snips, axis=1)) / np.mean(np.var(g_q_snips, axis=0))

        if plot:
            self.make_plot(snippets_norm, fs, sizes, snippets_peaks_norm, tuning_curves,
                           title=f"w_qidx={w_qidx:.2f}, b_qidx={b_qidx:.2f}, g_qidx={g_qidx:.2f}\n" +
                                 f"w_pref_size={w_pref_size}, b_pref_size={b_pref_size}, g_pref_size={g_pref_size}\n" +
                                 f"w_si={w_spots_surround_index:.2f}, "
                                 f"b_si={b_spots_surround_index:.2f}, "
                                 f"g_si={g_spots_surround_index:.2f}")

        return (snippets_norm, fs, sizes, snippets_peaks_norm, tuning_curves, n_reps,
                w_qidx, b_qidx, g_qidx, col_pref,
                w_pref_size, w_spots_surround_index,
                b_pref_size, b_spots_surround_index,
                g_pref_size, g_spots_surround_index)

    def make_plot(self, snippets_norm, fs, sizes, snippets_peaks, snippets_col_peak, title=None):
        idx_max = int(self._t_max * fs)

        fig, axs = plt.subplots(1, 3, figsize=(12, 3), width_ratios=[2, 1, 1])

        fig.suptitle(title)

        ax = axs[0]
        for name, snippet in snippets_norm.items():
            for i, trace in enumerate(snippet.T):
                if 'W' in name:
                    col = 'k'
                elif 'B' in name:
                    col = 'violet'
                else:
                    col = 'green'
                ax.plot(np.arange(idx_max) / fs, trace[:idx_max], c=col, alpha=0.5)
        ax.axvline(self._t_base, c='dimgray', ls='--')
        ax.axvline(self._t_peak_a, c='r', ls='--')
        ax.axvline(self._t_peak_b, c='r', ls='--')

        ax = axs[1]
        ax.plot(sizes, snippets_col_peak['W'], c='k')
        ax.plot(sizes, snippets_col_peak['B'], c='violet')
        ax.plot(sizes, snippets_col_peak['G'], c='green')
        for col in ['W', 'B', 'G']:
            if 'W' in col:
                c = 'k'
            elif 'B' in col:
                c = 'violet'
            else:
                c = 'green'

            for size in sizes:
                snips = snippets_peaks[col + str(size).zfill(4)]
                ax.scatter(np.ones(len(snips)) * size, snips, c=c, s=1)

        ax.set_xlabel('Spot size [µm]')
        ax.set_ylabel('Peak response')

        ax = axs[2]
        ax.plot(sizes, snippets_col_peak['W'] - np.max(snippets_col_peak['W']), c='k')
        ax.plot(sizes, snippets_col_peak['B'] - np.max(snippets_col_peak['B']), c='violet')
        ax.plot(sizes, snippets_col_peak['G'] - np.max(snippets_col_peak['G']), c='green')

        ax.set_xlabel('Spot size [µm]')
        ax.set_ylabel('Peak response')

        plt.tight_layout()

        plt.show()

    def make(self, key, plot=False):
        if plot:
            print(key)

        (snippets_norm, fs, sizes, snippets_peaks, tuning_curves, n_reps,
         w_qidx, b_qidx, g_qidx, col_pref,
         w_pref_size, w_spots_surround_index,
         b_pref_size, b_spots_surround_index,
         g_pref_size, g_spots_surround_index) = self.compute_entry(key, plot=plot)

        data_dict = dict(
            snippets=snippets_norm,
            fs=fs,
            sizes=sizes,
            snippets_peaks=snippets_peaks,
            tuning_curves=tuning_curves,
        )

        self.insert1(dict(
            key,
            n_reps=n_reps,
            w_qidx=w_qidx,
            b_qidx=b_qidx,
            g_qidx=g_qidx,
            col_pref=col_pref,
            w_pref_size=w_pref_size,
            b_pref_size=b_pref_size,
            g_pref_size=g_pref_size,
            w_spot_surround_index=w_spots_surround_index,
            b_spot_surround_index=b_spots_surround_index,
            g_spot_surround_index=g_spots_surround_index,
            data_dict=data_dict,
        ))

    def plot1(self, key=None):
        key = get_primary_key(table=self, key=key)
        self.compute_entry(key, plot=True)
