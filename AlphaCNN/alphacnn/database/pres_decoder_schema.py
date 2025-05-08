import warnings

import datajoint as dj
import numpy as np
import seaborn as sns
from IPython.core.display_functions import clear_output
from matplotlib import pyplot as plt

from alphacnn.database.dataset_schema import DataNorm, DataSet, DataSplit
from alphacnn.database.dataset_utils import get_dataset_dict, combine_w_and_wo_flat
from alphacnn.decoder.decoder_utils import save_decoder, plot_decoder_loss
from alphacnn.decoder.pres_decoder import train_pres_decoder

assert len(DataSet().proj()) > 0
assert len(DataNorm().proj()) > 0
assert len(DataSplit().proj()) > 0

pres_decoder_schema = dj.Schema()


def connect_to_database(dj_config_file, schema_name, create_schema=True, create_tables=True):
    dj.config.load(dj_config_file)
    dj.config['schema_name'] = schema_name
    print("schema_name:", dj.config['schema_name'])
    dj.conn()

    pres_decoder_schema.activate(schema_name, create_schema=create_schema, create_tables=create_tables)


@pres_decoder_schema
class PresDecoderKind(dj.Manual):
    definition = """
    decoder_id : varchar(32)
    ---
    kind : varchar(191)
    params : longblob
    data_filter_params = NULL: longblob
    """

    def add(self, decoder_id, kind, params, data_filter_params=None, skip_duplicates=True):
        self.insert1(dict(decoder_id=decoder_id, kind=kind, params=params, data_filter_params=data_filter_params),
                     skip_duplicates=skip_duplicates)


@pres_decoder_schema
class PresDecoderPrediction(dj.Computed):
    definition = """
    -> PresDecoderKind
    -> DataNorm
    -> DataSplit
    split_kind : enum('train', 'dev', 'test')
    ---
    keys : mediumblob
    key_idx : longblob
    y : longblob
    d : longblob
    p : longblob
    p_pred : longblob
    """

    class PresDecoderInfo(dj.Part):
        definition = """
        -> master
        ---
        history = NULL : longblob
        model_path = NULL : varchar(512)
        """

    @property
    def key_source(self):
        return PresDecoderKind().proj() * DataNorm().proj() * DataSplit().proj()

    def populate(self, *restrictions, suppress_errors=False, return_exception_objects=False, reserve_jobs=False,
                 order="original", limit=None, max_calls=None, display_progress=False, processes=1, make_kwargs=None):
        if processes > 1:
            warnings.warn('Multiprocessing not supported for this table')
            processes = 1

        super().populate(*restrictions, suppress_errors=suppress_errors,
                         return_exception_objects=return_exception_objects,
                         reserve_jobs=reserve_jobs, order=order, limit=limit, max_calls=max_calls,
                         display_progress=display_progress, processes=processes, make_kwargs=make_kwargs)

    def make(self, key):
        print(key)
        data_set_file, split_id = (DataNorm * DataSplit() & key).fetch1('data_set_file', 'split_id')

        decoder_id, decoder_kind, params, data_filter_params = (PresDecoderKind & key).fetch1(
            'decoder_id', 'kind', 'params', 'data_filter_params')

        if data_filter_params is not None:
            warnings.warn(f'Ignoring data_filter_params={data_filter_params}')

        if 'ensemble' in decoder_id:
            decoder_name, is_ensemble, n_ensemble = decoder_id.split('_')
            n_ensemble = int(n_ensemble)
        else:
            n_ensemble = 1

        print('Get data')
        dat = get_dataset_dict(
            data_set_file=data_set_file, split_id=split_id,
            flat_x=False, augment_train=True, augment_dev=True, augment_test=False,
            get_wo=True, cricket_only=True, merge_wo=False, add_p=True, return_keys=True)

        print('Add pseudo-pairwise data')
        for kind in ['train', 'dev', 'test']:
            dat[f'x_{kind}_mixed'], dat[f'p_{kind}_mixed'], dat[f'shuffle_idx_{kind}'] = combine_w_and_wo_flat(
                dat[f'x_{kind}'][dat[f'p_{kind}']], dat[f'x_{kind}_wo'][dat[f'p_{kind}']])

        print('Train decoder')
        decoder, history = train_pres_decoder(
            n_ensemble=n_ensemble, kind=decoder_kind,
            X_train=dat['x_train_mixed'], p_train=dat['p_train_mixed'],
            X_dev=dat['x_dev_mixed'], p_dev=dat['p_dev_mixed'],
            params=params, norm_data=False, seed=42)

        print('Add data to table')
        self.add(decoder=decoder, decoder_id=decoder_id, data_set_file=data_set_file,
                 split_id=split_id, dat=dat, history=history)
        clear_output()

    def add(self, decoder, decoder_id, data_set_file, split_id, dat, history):

        for kind in ['train', 'dev', 'test']:
            dat[f'p_pred_{kind}'] = decoder.eval(X_test=dat[f'x_{kind}_mixed'])

        model_key = dict(decoder_id=decoder_id, data_set_file=data_set_file, split_id=split_id)

        print('Save decoder to file')
        model_path = save_decoder(model_key=model_key, decoder=decoder, decoder_kind='pres_decoder')

        print('Add table to database')
        for kind in ['train', 'dev', 'test']:
            keys = dat[f'keys_{kind}']
            x_lens_corrected = dat[f'x_{kind}_lens_corrected']
            shuffle_idx = dat[f'shuffle_idx_{kind}']

            key_idx = np.concatenate([np.full(x_len, i % len(keys), dtype=int)
                                      for i, x_len in enumerate(x_lens_corrected)])
            key_idx = np.concatenate([key_idx, key_idx])[shuffle_idx]

            y = np.vstack([dat[f'y_{kind}'], dat[f'y_{kind}']])[shuffle_idx]
            d = np.concatenate([dat[f'd_{kind}'], dat[f'd_{kind}']])[shuffle_idx]
            p = dat[f'p_{kind}_mixed']
            p_pred = dat[f'p_pred_{kind}']
    
            assert len(y) == len(d), f"{len(y)} != {len(d)}"
            assert len(y) == len(p), f"{len(y)} != {len(p)}"
            assert len(y) == len(p_pred), f"{len(y)} != {len(p_pred)}"
            assert len(y) == len(key_idx), f"{len(y)} != {len(key_idx)}"

            self.insert1(dict(
                **model_key, split_kind=kind,
                keys=keys,
                key_idx=key_idx,
                y=y,
                d=d,
                p=p,
                p_pred=p_pred,
            ))

        self.PresDecoderInfo().insert1(dict(**model_key, split_kind='train', history=history, model_path=model_path))

    def plot(self, decoder_id, data_set_file, split_id, split_kind=None):

        dat_tab = self & dict(decoder_id=decoder_id, data_set_file=data_set_file, split_id=split_id)

        dat = dict()
        for split_kind in ['train', 'dev', 'test']:
            p, p_pred, d, y = (dat_tab & dict(split_kind=split_kind)).fetch1('p', 'p_pred', 'd', 'y')
            dat[f'p_{split_kind}'] = p
            dat[f'p_pred_{split_kind}'] = p_pred
            dat[f'd_{split_kind}'] = d
            dat[f'y_{split_kind}'] = y

        fig, axs = plt.subplots(4, 3, figsize=(12, 8), sharey='all')

        for ax_col, kind in zip(axs.T, ['train', 'dev', 'test']):
            d = dat[f'd_{kind}']
            xy = dat[f'y_{kind}']
            p_true = dat[f'p_{kind}'].astype(float)
            p_pred = dat[f'p_pred_{kind}'].astype(float)

            error = np.abs(p_true - p_pred)
            error_binary = np.abs(p_true - (p_pred > 0.5))

            d_bins = np.linspace(0, 1, 11)
            d_bins_error = [error_binary[(d >= i1) & (d <= i2)] for i1, i2 in zip(d_bins[:-1], d_bins[1:])]

            x_bins = np.linspace(np.nanmin(xy[:, 0]), np.nanmax(xy[:, 0]), 11)
            x_bins_error = [error_binary[(xy[:, 0] >= i1) & (xy[:, 0] <= i2)]
                            for i1, i2 in zip(x_bins[:-1], x_bins[1:])]

            y_bins = np.linspace(np.nanmin(xy[:, 1]), np.nanmax(xy[:, 1]), 11)
            y_bins_error = [error_binary[(xy[:, 1] >= i1) & (xy[:, 1] <= i2)]
                            for i1, i2 in zip(y_bins[:-1], y_bins[1:])]

            ax = ax_col[0]
            ax.set(title=kind, ylabel='error', xlabel='normalized distance', xlim=(0, 1))
            sns.regplot(ax=ax, x=d, y=error, order=4, scatter_kws=dict(s=0.3, color='gray'), line_kws=dict(color='r'),
                        ci=None)
            ax.axhline(0.5, c='k')

            ax_col[1].set_xlabel('norm dist bin')
            ax_col[2].set_xlabel('x bin')
            ax_col[3].set_xlabel('y bin')

            for ax, bins, errors in zip(
                    ax_col[1:], [d_bins, x_bins, y_bins], [d_bins_error, x_bins_error, y_bins_error]):
                ax.axhline(0.5, c='k')
                sns.barplot(ax=ax, data=errors, color='gray')
                ax.set_xticklabels(
                    [(np.around(i1, 1), np.around(i2, 1)) for i1, i2 in zip(bins[:-1], bins[1:])],
                    rotation=90)
                ax.set(ylabel='error (binary)')
                ax.set(ylim=(-0.05, 1.05))

        plt.tight_layout()
        plt.show()

    def plot_loss(self, decoder_id, data_set_file, split_id, split_kind=None):
        key = dict(decoder_id=decoder_id, data_set_file=data_set_file, split_id=split_id)
        plot_decoder_loss((self.PresDecoderInfo & key).fetch1('history'))
