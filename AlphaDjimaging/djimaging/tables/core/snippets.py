import warnings
from abc import abstractmethod

import datajoint as dj
import numpy as np
from matplotlib import pyplot as plt

from djimaging.tables.core.stimulus import reformat_numerical_trial_info
from djimaging.utils import plot_utils
from djimaging.utils.dj_utils import get_primary_key
from djimaging.utils.scanm_utils import split_trace_by_reps, split_trace_by_group_reps


def get_aligned_snippets_times(snippets_times, raise_error=True, tol=1e-4):
    snippets_times = snippets_times - snippets_times[0, :]

    is_inconsistent = np.any(np.std(snippets_times, axis=1) > tol)
    if is_inconsistent:
        if raise_error:
            raise ValueError(f'Failed to snippet times: max_std={np.max(np.std(snippets_times, axis=1))}')
        else:
            warnings.warn(f'Snippet times are inconsistent: max_std={np.max(np.std(snippets_times, axis=1))}')

    aligned_times = np.mean(snippets_times, axis=1)
    return aligned_times


class SnippetsTemplate(dj.Computed):
    database = ""
    _pad_trace = None  # If True, chose snippet times always contain the trigger times
    _dt_base_rng = None  # dict of baseline time for each stimulus with stimulus based baseline correction
    _delay = None  # dict of delay for each stimulus, defaults to None

    @property
    def definition(self):
        definition = """
        # Snippets created from slicing traces using the triggertimes. 
        -> self.preprocesstraces_table
        ---
        snippets               :longblob          # array of snippets (time x repetitions)
        snippets_times         :longblob          # array of snippet times (time x repetitions)
        triggertimes_snippets  :longblob          # snippeted triggertimes (ntrigger_rep x repetitions)
        droppedlastrep_flag    :tinyint unsigned  # Was the last repetition incomplete and therefore dropped?
        """
        return definition

    @property
    @abstractmethod
    def preprocesstraces_table(self):
        pass

    @property
    @abstractmethod
    def stimulus_table(self):
        pass

    @property
    @abstractmethod
    def presentation_table(self):
        pass

    @property
    @abstractmethod
    def traces_table(self):
        pass

    @property
    def key_source(self):
        try:
            return (self.preprocesstraces_table() & (self.stimulus_table() & "isrepeated=1")).proj()
        except (AttributeError, TypeError):
            pass

    def make(self, key):
        stim_name = (self.stimulus_table() & key).fetch1('stim_name')
        ntrigger_rep = (self.stimulus_table() & key).fetch1('ntrigger_rep')
        triggertimes = (self.presentation_table() & key).fetch1('triggertimes')
        trace_times, trace = (self.preprocesstraces_table() & key).fetch1('preprocess_trace_times', 'preprocess_trace')

        dt_base_rng = None if self._dt_base_rng is None else self._dt_base_rng.get(stim_name, None)
        if self._pad_trace is None:
            pad_trace = False
        elif isinstance(self._pad_trace, bool):
            pad_trace = self._pad_trace
        elif isinstance(self._pad_trace, dict):
            pad_trace = self._pad_trace.get(stim_name, False)
        else:
            raise ValueError(f"pad_trace must be bool or dict, not {type(self._pad_trace)}")
        delay = 0 if self._delay is None else self._delay.get(stim_name, 0)

        snippets, snippets_times, triggertimes_snippets, droppedlastrep_flag = split_trace_by_reps(
            trace, trace_times, triggertimes, ntrigger_rep, allow_drop_last=True, delay=delay,
            pad_trace=pad_trace, dt_base_rng=dt_base_rng)[:4]

        self.insert1(dict(
            **key,
            snippets=snippets,
            snippets_times=snippets_times,
            triggertimes_snippets=triggertimes_snippets,
            droppedlastrep_flag=int(droppedlastrep_flag),
        ))

    def plot1(self, key=None, xlim=None, xlim_aligned=None):
        key = get_primary_key(table=self, key=key)
        snippets, snippets_times, triggertimes_snippets = (self & key).fetch1(
            "snippets", "snippets_times", "triggertimes_snippets")

        fig, axs = plt.subplots(3, 1, figsize=(10, 6))

        plot_utils.plot_trace_and_trigger(
            ax=axs[0], time=snippets_times, trace=snippets, triggertimes=triggertimes_snippets, title=str(key))
        axs[0].set(xlim=xlim)

        axs[1].plot(snippets_times - triggertimes_snippets[0], snippets, alpha=0.5)
        axs[1].set(ylabel='trace', xlabel='rel. to trigger', xlim=xlim_aligned)

        aligned_times = get_aligned_snippets_times(snippets_times=snippets_times)
        plot_utils.plot_traces(
            ax=axs[2], time=aligned_times, traces=snippets.T)
        axs[2].set(ylabel='trace', xlabel='aligned time', xlim=xlim_aligned)

        plt.tight_layout()


class GroupSnippetsTemplate(dj.Computed):
    database = ""

    @property
    def definition(self):
        definition = """
        # Snippets created from slicing traces using the triggertimes. 
        -> self.preprocesstraces_table
        ---
        snippets               :longblob          # dict of array of snippets (group: time [x repetitions])
        snippets_times         :longblob          # dict of array of snippet times (group: time [x repetitions])
        triggertimes_snippets  :longblob          # dict of array of triggertimes (group: time [x repetitions])
        droppedlastrep_flag    :tinyint unsigned  # Was the last repetition incomplete and therefore dropped?
        """
        return definition

    @property
    @abstractmethod
    def preprocesstraces_table(self):
        pass

    @property
    @abstractmethod
    def stimulus_table(self):
        pass

    @property
    @abstractmethod
    def presentation_table(self):
        pass

    @property
    @abstractmethod
    def traces_table(self):
        pass

    @property
    def key_source(self):
        try:
            return (self.preprocesstraces_table() & (self.stimulus_table() & "trial_info!='None'")).proj()
        except (AttributeError, TypeError):
            pass

    def make(self, key):
        trial_info, stim_dict = (self.stimulus_table() & key).fetch1('trial_info', 'stim_dict')
        triggertimes = (self.presentation_table() & key).fetch1('triggertimes')
        trace_times, traces = (self.preprocesstraces_table() & key).fetch1('preprocess_trace_times', 'preprocess_trace')

        if not isinstance(trial_info[0], dict):
            trial_info = reformat_numerical_trial_info(trial_info)

        delay = stim_dict.get('trigger_delay', 0.)

        snippets, snippets_times, triggertimes_snippets, droppedlastrep_flag = split_trace_by_group_reps(
            traces, trace_times, triggertimes, trial_info=trial_info, allow_incomplete=True, delay=delay,
            stack_kind='pad')

        self.insert1(dict(
            **key,
            snippets=snippets,
            snippets_times=snippets_times,
            triggertimes_snippets=triggertimes_snippets,
            droppedlastrep_flag=droppedlastrep_flag,
        ))

    def plot1(self, key=None, xlim=None):
        key = get_primary_key(table=self, key=key)
        snippets, snippets_times, triggertimes_snippets = (self & key).fetch1(
            "snippets", "snippets_times", "triggertimes_snippets")

        import matplotlib as mpl

        names = list(snippets.keys())

        colors = mpl.colormaps['jet'](np.linspace(0, 1, len(names)))
        name2color = {name: colors[i] for i, name in enumerate(names)}

        fig, axs = plt.subplot_mosaic([["all"]] + [[name] for name in names], figsize=(8, 2 * (len(names) + 1)))

        tt_min = np.min([np.nanmin(snippets[name]) for name in names])
        tt_max = np.max([np.nanmax(snippets[name]) for name in names])

        for i, name in enumerate(names):
            axs['all'].vlines(triggertimes_snippets[name], tt_min, tt_max, color='k', zorder=-100, lw=0.5, alpha=0.5,
                              label='trigger' if name == names[0] else '_')
            axs['all'].plot(snippets_times[name], snippets[name], color=name2color[name],
                            label=[name] + ['_'] * (snippets[name].shape[1] - 1), alpha=0.8)

            axs[name].set(title='trace', xlabel='absolute time')
            axs[name].plot(snippets_times[name] - triggertimes_snippets[name][0], snippets[name], lw=1)
            axs[name].set(title=name, xlabel='relative time')
            axs[name].title.set_color(name2color[name])
            axs[name].vlines(triggertimes_snippets[name] - triggertimes_snippets[name][0],
                             tt_min, tt_max, color='k', zorder=-100, lw=0.5, alpha=0.5,
                             label='trigger' if name == names[0] else '_')

        axs['all'].set(xlim=xlim)
        plt.tight_layout()
        axs['all'].legend(bbox_to_anchor=(1, 1), loc='upper left')

        return fig, axs
