"""
Example usage:
@schema
class SineSpotSurroundIndex(response.SineSpotSurroundIndexTemplate):
    _stim_name = 'sinespot'
    _dt_spot = 1.  # Duration of the spot in seconds including the pause between spots
    _dt_baseline = 0.25  # Duration of the baseline in seconds
    snippets_table = Snippets
"""

from abc import abstractmethod

import datajoint as dj
import numpy as np

from matplotlib import pyplot as plt

from djimaging.utils.dj_utils import get_primary_key
from djimaging.utils.trace_utils import get_mean_dt


class SineSpotSurroundIndexTemplate(dj.Computed):
    database = ""

    _stim_name = 'sinespot'

    _dt_spot = 1.  # Duration of the spot in seconds including the pause between spots
    _dt_baseline = 0.25  # Duration of the baseline in seconds

    @property
    def definition(self):
        definition = f"""
        -> self.snippets_table
        ---
        sinespot_surround_index = NULL : float
        response_mus : blob  # (repetitions x spots) matrix of responses 
        """
        return definition

    @property
    @abstractmethod
    def snippets_table(self):
        pass

    @property
    def key_source(self):
        try:
            return self.snippets_table.proj() & f"stim_name = '{self._stim_name}'"
        except (AttributeError, TypeError):
            pass

    def compute_entry(self, key, plot=False):
        snippets, snippets_times, triggertimes_snippets = (self.snippets_table & key).fetch1(
            'snippets', 'snippets_times', 'triggertimes_snippets')

        dt = get_mean_dt(snippets_times.T)[0]
        n_reps = snippets.shape[1]
        fs = 1. / dt

        r_mus = np.full((n_reps, 6), np.nan)

        n_idxs_base = int(np.round(self._dt_baseline * fs))
        n_idxs_spot = int(np.round(self._dt_spot * fs))

        if plot:
            fig, axs = plt.subplots(n_reps, 1, figsize=(8, 5))

        for rep_i in range(n_reps):
            if plot:
                ax = axs[rep_i]
                ax.plot(snippets_times[:, rep_i], snippets[:, rep_i], c='k')

            for spot_j in range(6):
                idxs_base_j = np.arange(n_idxs_base) + n_idxs_spot * spot_j
                idxs_r_j = np.arange(n_idxs_base, n_idxs_spot) + n_idxs_spot * spot_j

                base = np.median(snippets[idxs_base_j, rep_i])
                r_j = np.mean(snippets[idxs_r_j, rep_i] - base)

                r_mus[rep_i, spot_j] = r_j

                if plot:
                    c = f"C{spot_j}"
                    ax = axs[rep_i]
                    ax.plot(snippets_times[idxs_base_j, rep_i], snippets[idxs_base_j, rep_i],
                            color=c, alpha=0.5, ls='--')
                    ax.fill_between(snippets_times[idxs_r_j, rep_i],
                                    np.ones(idxs_r_j.size) * base,
                                    snippets[idxs_r_j, rep_i], color=c)

                    for tt in triggertimes_snippets[:, rep_i]:
                        ax.axvline(tt, c='r')

        r_ls = r_mus[:, 0]
        r_gs = r_mus[:, 1]

        l_mu = np.median(r_ls)
        g_mu = np.median(r_gs)
        surround_index = g_mu - np.clip(l_mu, 0, None)

        if plot:
            plt.tight_layout(rect=(0, 0, 0.7, 1))
            ax = fig.add_axes(rect=(0.8, 0.8, 0.15, 0.15))

            for spot_j in range(6):
                ax.scatter(np.ones(n_reps) * spot_j, r_mus[:, spot_j], color=f'C{spot_j}', alpha=0.5, clip_on=False)
            ax.axhline(l_mu, c='C0', ls='--')
            ax.axhline(g_mu, c='C1', ls='--')

            ax = fig.add_axes(rect=(0.8, 0.45, 0.15, 0.15))
            ax.scatter(np.arange(n_reps), r_gs - np.clip(r_ls, 0, None), color='dimgray')
            ax.axhline(surround_index, c='k', ls='--')
            ax.set(xlabel='reps', ylabel='r_g - r_l', title=f"surround index = {surround_index:.2f}")

            plt.show()

        return surround_index, r_mus

    def make(self, key, plot=False):
        surround_index, r_mus = self.compute_entry(key, plot=plot)

        self.insert1(dict(
            key,
            sinespot_surround_index=surround_index,
            response_mus=r_mus
        ))

    def plot1(self, key=None):
        key = get_primary_key(table=self, key=key)
        self.compute_entry(key, plot=True)
