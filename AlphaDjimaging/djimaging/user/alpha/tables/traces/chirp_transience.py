"""
Tables for Chirp response features from Franke et al. 2017

Example usage:

from djimaging.tables import response


@schema
class ChirpTransience(traces.ChirpTransienceTemplate):
    stimulus_table = Stimulus
    snippets_table = Snippets
    presentation_table = Presentation
"""

from abc import abstractmethod

import datajoint as dj
import numpy as np
from matplotlib import pyplot as plt

from djimaging.utils.dj_utils import get_primary_key


class ChirpTransienceTemplate(dj.Computed):
    database = ""

    @property
    def definition(self):
        definition = '''
        # Computes Chirp transience index for ON responses.
        -> self.snippets_table
        ---
        
        transience_index = NULL: float
        '''
        return definition

    @property
    @abstractmethod
    def stimulus_table(self):
        pass

    @property
    @abstractmethod
    def snippets_table(self):
        pass

    @property
    @abstractmethod
    def presentation_table(self):
        pass

    @property
    def key_source(self):
        try:
            return self.snippets_table().proj() & \
                (self.stimulus_table() & "stim_name = 'chirp' or stim_family = 'chirp'")
        except (AttributeError, TypeError):
            pass

    def compute_entry(self, key, plot=False):
        snips, snips_t, snips_tt = (self.snippets_table & key).fetch1(
            'snippets', 'snippets_times', 'triggertimes_snippets')

        bases = []
        rs_a = []
        rs_b = []

        for i, (snip, snip_t, snip_tt) in enumerate(zip(snips.T, snips_t.T, snips_tt.T)):
            tt0 = snip_tt[0]
            base = np.median(snip[(snip_t >= tt0) & (snip_t < tt0 + 2)])
            r_a = np.percentile(snip[(snip_t >= tt0 + 2) & (snip_t < tt0 + 3)], q=90) - base
            r_b = np.percentile(snip[(snip_t >= tt0 + 4) & (snip_t < tt0 + 5)], q=90) - base

            bases.append(base)
            rs_a.append(r_a)
            rs_b.append(r_b)

        rs_a = np.array(rs_a)
        rs_b = np.clip(np.array(rs_b), 0, None)

        if np.any(rs_a <= 0):
            tri = np.nan
        else:
            assert np.all(rs_a >= 0)
            assert np.all(rs_b >= 0)
            tri = np.median((rs_a - rs_b) / (rs_a + rs_b))

        if plot:
            fig, ax = plt.subplots(1, 1, figsize=(5, 4))

            for i, (snip, snip_t, snip_tt, base, r_a, r_b) in enumerate(zip(
                    snips.T, snips_t.T, snips_tt.T, bases, rs_a, rs_b)):
                tt0 = snip_tt[0]

                ax.plot(snip_t - tt0, snip - base, c=f"C{i}", alpha=0.3)

                ax.plot([2, 3], [r_a, r_a], 'x-', c=f"C{i}")
                ax.plot([4, 5], [r_b, r_b], 'x-', c=f"C{i}")

            ax.set_xlim(0, 7)
            ax.set_title(f"TRi={tri:.2f}")
            plt.tight_layout()
            plt.show()

        return tri

    def make(self, key, plot=False):
        transience_index = self.compute_entry(key, plot=plot)

        self.insert1(dict(
            key,
            transience_index=transience_index
        ))

    def plot1(self, key=None):
        key = get_primary_key(table=self, key=key)
        self.compute_entry(key, plot=True)
