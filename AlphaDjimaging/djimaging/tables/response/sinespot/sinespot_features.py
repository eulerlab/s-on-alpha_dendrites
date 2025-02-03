from abc import abstractmethod

import datajoint as dj
import numpy as np
from matplotlib import pyplot as plt

from djimaging.utils.dj_utils import get_primary_key


class SineSpotFeaturesTemplate(dj.Computed):
    database = ""

    _dt_base_before = 0.2
    _dt_base_after = 0.7
    _stim_names = ['sinespot']

    @property
    def definition(self):
        definition = '''
        # Compute SineSpot features
        -> self.averages_table
        ---
        response_spot_small : float  # Mean response to small spot
        response_spot_large : float  # Mean response to large spot
        other_spot_responses : blob  # Mean response to other spots
        surround_index = NULL : float  # r_large / r_small 
        offset_index = NULL : float  # r_ss / r_small,  where r_ss is strongest small spot response
        '''
        return definition

    @property
    @abstractmethod
    def averages_table(self):
        pass

    @property
    def key_source(self):
        try:
            return (self.averages_table & [dict(stim_name=stim_name) for stim_name in self._stim_names]).proj()
        except (AttributeError, TypeError):
            pass

    def _make_compute(self, key, plot=False):
        avg, avg_t, avg_tt = (self.averages_table & key).fetch1('average', 'average_times', 'triggertimes_rel')

        if plot:
            fig, ax = plt.subplots(1, 1, figsize=(8, 3))
        else:
            ax = False

        responses = compute_sinespot_responses(
            avg, avg_t, avg_tt, dt_base_before=self._dt_base_before, dt_base_after=self._dt_base_after, plot=ax)

        small_spot = responses[0]
        large_spot = responses[1]
        other_spots = responses[2:]

        surround_index = (large_spot - small_spot)
        offset_index = (np.max(other_spots) - small_spot)

        return small_spot, large_spot, other_spots, surround_index, offset_index

    def make(self, key, plot=False):
        small_spot, large_spot, other_spots, surround_index, offset_index = self._make_compute(key, plot=plot)

        self.insert1(dict(
            **key,
            surround_index=surround_index,
            offset_index=offset_index,
            response_spot_small=small_spot,
            response_spot_large=large_spot,
            other_spot_responses=other_spots,
        ))

    def plot1(self, key=None):
        key = get_primary_key(table=self, key=key)
        self._make_compute(key, plot=True)

    def plot(self, restriction=None):
        if restriction is None:
            restriction = dict()

        response_spot_small = (self & restriction).fetch("response_spot_small")
        response_spot_large = (self & restriction).fetch("response_spot_large")
        surround_index = (self & restriction).fetch("surround_index")
        offset_index = (self & restriction).fetch("offset_index")

        fig, axs = plt.subplots(1, 4, figsize=(12, 3))

        ax = axs[0]
        ax.hist(response_spot_small)
        ax.set(title="small")

        ax = axs[1]
        ax.hist(response_spot_large)
        ax.set(title="large")

        ax = axs[2]
        ax.hist(surround_index)
        ax.set(title="surround_index")

        ax = axs[3]
        ax.hist(offset_index)
        ax.set(title="offset_index")

        plt.tight_layout()
        plt.show()


def compute_sinespot_responses(
        avg, avg_t, avg_tt, dt_base_before=0.2, dt_base_after=0.7, atol=0.05, plot=False):
    from djimaging.utils.trace_utils import find_closest

    if plot:
        if isinstance(plot, plt.Axes):
            ax = plot
        else:
            fig, ax = plt.subplots(1, 1, figsize=(8, 3))
    else:
        ax = None

    if ax is not None:
        ax.plot(avg_t, avg, c='k')
        ax.vlines(avg_tt, ymin=avg.min(), ymax=avg.max(), color='r', ls='--')
        ax.set(xlabel='Time  [s]', ylabel='Response', title='SineSpot responses')

    responses = np.zeros(avg_tt.size)
    for i, tt in enumerate(avg_tt):
        t_idx_1 = find_closest(target=tt + dt_base_before, data=avg_t, atol=atol, as_index=True)
        t_idx_2 = find_closest(target=tt + dt_base_after, data=avg_t, atol=atol, as_index=True)

        response = np.mean(avg[t_idx_1:t_idx_2])
        responses[i] = response

        if ax is not None:
            ax.fill_between(avg_t[t_idx_1:t_idx_2], avg[t_idx_1:t_idx_2])
            ax.text(avg_t[(t_idx_1 + t_idx_2) // 2], avg.max(), f"{response:.2f}", ha='center', va='top')

    return responses
