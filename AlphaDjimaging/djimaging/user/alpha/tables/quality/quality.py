from abc import abstractmethod

import datajoint as dj
import matplotlib.pyplot as plt
import numpy as np


class QualityParamsTemplate(dj.Lookup):
    database = ""

    @property
    def definition(self):
        definition = f"""
        quality_params_id : tinyint unsigned
        ---
        min_qidx_gchirp : float
        min_qidx_lchirp : float
        """
        return definition

    def add(self,
            min_qidx_gchirp=0.35, min_qidx_lchirp=0.35,
            params_id=1, skip_duplicates=False):
        key = dict(
            min_qidx_gchirp=min_qidx_gchirp, min_qidx_lchirp=min_qidx_lchirp, quality_params_id=params_id)
        self.insert1(key, skip_duplicates=skip_duplicates)


class QualityIndexTemplate(dj.Computed):
    database = ""

    @property
    def definition(self):
        definition = f'''
        -> self.params_table
        -> self.chirp_qi_table().proj(gchirp_stim_name='stim_name')
        -> self.chirp_qi_table().proj(lchirp_stim_name='stim_name')
        ---
        q_tot : tinyint unsigned  # 1: Use this ROI, 0: Don't use it.
        q_gchirp : tinyint  # 1: Good response, 0: bad response, -1: No or bad recording
        q_lchirp : tinyint  # 1: Good response, 0: bad response, -1: No or bad recording
        '''
        return definition

    @property
    @abstractmethod
    def params_table(self):
        pass

    @property
    @abstractmethod
    def chirp_qi_table(self):
        pass

    @property
    def key_source(self):
        try:
            return (
                    (self.chirp_qi_table & dict(stim_name='gChirp')).proj(gchirp_stim_name='stim_name') *
                    (self.chirp_qi_table & dict(stim_name='lChirp')).proj(lchirp_stim_name='stim_name') *
                    self.params_table().proj()
            )
        except (AttributeError, TypeError):
            pass

    @staticmethod
    def get_quality(qidx, min_qidx):
        qidx = np.atleast_1d(qidx)
        assert len(qidx) <= 1

        if len(qidx) == 0:
            return -1
        elif qidx[0] >= min_qidx:
            return 1
        else:
            return 0

    @staticmethod
    def combine_qualities(q_qidx):
        if q_qidx == 1:
            return 1
        elif q_qidx == 0:
            return 0
        else:
            return -1

    def make(self, key):
        min_qidx_gchirp, min_qidx_lchirp = (self.params_table & key).fetch1(
            "min_qidx_gchirp", "min_qidx_lchirp")

        # Signal to Noise
        q_gchirp_qidx = self.get_quality(
            (self.chirp_qi_table() & "stim_name='gChirp'" & key).fetch("qidx"), min_qidx_gchirp)
        q_lchirp_qidx = self.get_quality(
            (self.chirp_qi_table() & "stim_name='lChirp'" & key).fetch("qidx"), min_qidx_lchirp)

        # Get stimuli with response
        q_gchirp = self.combine_qualities(q_gchirp_qidx)
        q_lchirp = self.combine_qualities(q_lchirp_qidx)

        q_tot = ((q_gchirp == 1) or (q_lchirp == 1))

        key = key.copy()
        key['q_tot'] = q_tot
        key['q_gchirp'] = q_gchirp
        key['q_lchirp'] = q_lchirp
        self.insert1(key)

    def plot(self):
        def plot_qidx(ax_, qidxs_, title):
            qidxs_ = np.asarray(qidxs_).astype(int)
            bins = np.arange(np.min(qidxs_) - 0.25, np.max(qidxs_) + 0.5, 0.5)
            ax_.hist(qidxs_, bins=bins)
            ax_.set_xticks(np.unique(qidxs_))
            ax_.set_title(title)

        fig, axs = plt.subplots(1, 3, figsize=(10, 2))
        plot_qidx(ax_=axs[0], qidxs_=self.fetch('q_tot'), title='q_tot')
        plot_qidx(ax_=axs[1], qidxs_=self.fetch('q_gchirp'), title='q_gchirp')
        plot_qidx(ax_=axs[2], qidxs_=self.fetch('q_lchirp'), title='q_lchirp')
        plt.show()
