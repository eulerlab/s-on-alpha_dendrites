import numpy as np
from djimaging.tables import receptivefield
from djimaging.tables.receptivefield.rf_utils import prepare_noise_data
from djimaging.utils.dj_utils import get_primary_key


class EventsPerSecondTemplate(receptivefield.DNoiseTraceTemplate):
    database = ""

    def make(self, key):
        stim = (self.stimulus_table() & key).fetch1("stim_trace")
        stimtime = (self.presentation_table() & key).fetch1('triggertimes')
        trace, tracetime = (self.traces_table() & key).fetch1('preprocess_trace', 'preprocess_trace_times')
        fupsample_trace, fupsample_stim, fit_kind, lowpass_cutoff, pre_blur_sigma_s, post_blur_sigma_s, ref_time = (
                self.params_table() & key).fetch1(
            "fupsample_trace", "fupsample_stim", "fit_kind", "lowpass_cutoff",
            "pre_blur_sigma_s", "post_blur_sigma_s", "ref_time")

        stim, trace, dt, t0, dt_rel_error = prepare_noise_data(
            trace=trace, tracetime=tracetime, stim=stim, triggertimes=stimtime,
            fupsample_trace=fupsample_trace, fupsample_stim=fupsample_stim, ref_time=ref_time,
            fit_kind=fit_kind, lowpass_cutoff=lowpass_cutoff,
            pre_blur_sigma_s=pre_blur_sigma_s, post_blur_sigma_s=post_blur_sigma_s)

        time = np.arange(trace.size) * dt + t0

        data_key = key.copy()
        data_key['dt'] = dt
        data_key['time'] = time
        data_key['trace'] = trace
        data_key['stim'] = np.mean(trace > 0)  # Let's use this slot (Quick and dirty)
        data_key['dt_rel_error'] = dt_rel_error
        self.insert1(data_key)

    def plot1(self, key=None, xlim=None):
        key = get_primary_key(table=self, key=key)

        from matplotlib import pyplot as plt

        raw_trace, raw_tracetime = (self.traces_table() & key).fetch1('preprocess_trace', 'preprocess_trace_times')

        time, trace = (self & key).fetch1('time', 'trace')
        assert time.shape[0] == trace.shape[0], (time.shape[0], trace.shape[0])

        fit_kind = (self.params_table() & key).fetch1('fit_kind')

        fig, axs = plt.subplots(2, 1, figsize=(10, 5), sharex='all')
        ax = axs[0]
        ax.plot(time, trace, label='output trace')
        ax.legend(loc='upper left')
        ax.set(xlabel='Time', title=fit_kind)
        ax.set_xlim(xlim)

        ax = axs[1]
        ax.plot(time, trace, label='output trace')
        ax.legend(loc='upper left')
        ax = ax.twinx()
        ax.plot(raw_tracetime, raw_trace, 'r-', label='input trace', alpha=0.5)
        ax.legend(loc='upper right')
        ax.set(xlabel='Time', title='Input trace')
        plt.tight_layout()
