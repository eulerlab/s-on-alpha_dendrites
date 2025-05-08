import warnings
from abc import abstractmethod

import datajoint as dj
import numpy as np
from djimaging.utils import filter_utils
from djimaging.utils.dj_utils import get_primary_key
from djimaging.utils.plot_utils import plot_trace_and_trigger, plot_signals_heatmap
from matplotlib import pyplot as plt


class DownsampledAveragesTemplate(dj.Computed):
    database = ""

    _fdownsample = 4

    @property
    def definition(self):
        definition = """
        # Averages of snippets

        -> self.averages_table
        ---
        average             :longblob  # array of snippet average (time)
        average_norm        :longblob  # normalized array of snippet average (time)
        average_times       :longblob  # array of average time, starting at t=0 (time)
        triggertimes_rel    :longblob  # array of relative triggertimes 
        """
        return definition

    @property
    def key_source(self):
        try:
            return self.averages_table.proj()
        except (AttributeError, TypeError):
            pass

    @property
    @abstractmethod
    def averages_table(self):
        pass

    def make(self, key):
        average, average_norm, average_times, triggertimes_rel = (self.averages_table() & key).fetch1(
            'average', 'average_norm', 'average_times', 'triggertimes_rel')

        average_times_downsampled, average_downsampled = filter_utils.downsample_trace(
            tracetime=average_times, trace=average, fdownsample=self._fdownsample)

        average_times_downsampled_norm, average_norm_downsampled = filter_utils.downsample_trace(
            tracetime=average_times, trace=average_norm, fdownsample=self._fdownsample)

        assert np.allclose(average_times_downsampled, average_times_downsampled_norm)

        self.insert1(dict(
            **key,
            average=average_downsampled,
            average_norm=average_norm_downsampled,
            average_times=average_times_downsampled,
            triggertimes_rel=triggertimes_rel,
        ))

    def plot1(self, key=None):
        key = get_primary_key(table=self, key=key)
        average, average_norm, average_times, triggertimes_rel = \
            (self & key).fetch1('average', 'average_norm', 'average_times', 'triggertimes_rel')

        plot_trace_and_trigger(
            time=average_times, trace=average, triggertimes=triggertimes_rel, trace_norm=average_norm, title=str(key))

    def plot(self, restriction=None):
        if restriction is None:
            restriction = dict()

        average = (self & restriction).fetch('average')

        sizes = [a.size for a in average]

        if np.unique(sizes).size > 1:
            warnings.warn('Traces do not have the same size. Are you plotting multiple stimuli?')
        min_size = np.min(sizes)

        ax = plot_signals_heatmap(signals=np.stack([a[:min_size] for a in average]))
        ax.set(title='Averages')
        plt.show()
