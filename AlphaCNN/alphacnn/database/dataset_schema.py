import warnings

import datajoint as dj
import numpy as np
from matplotlib import pyplot as plt

from alphacnn.database.dataset_utils import flatten_datapoints, split_flat_datapoints
from alphacnn.database.dataset_utils import load_raw_data_set, y_to_p, get_split_indexes_equal_lengths

dataset_schema = dj.Schema()


def connect_to_database(dj_config_file, schema_name, create_schema=True, create_tables=True):
    dj.config.load(dj_config_file)
    dj.config['schema_name'] = schema_name
    print("schema_name:", dj.config['schema_name'])
    dj.conn()

    dataset_schema.activate(schema_name, create_schema=create_schema, create_tables=create_tables)


@dataset_schema
class DataSet(dj.Manual):
    definition = """
    data_set_file: varchar(256)
    ---
    n_videos : int
    n_samples_per_video : int
    pixel_size_um : float
    video_width : int
    video_height : int
    """

    class DataPoint(dj.Part):
        definition = """
        -> master
        stimulus_id : varchar(32)
        bc_noise_sample: int
        rgc_noise_sample: int
        --- 
        y_full : blob
        d_full : blob
        p_full : blob
        p_sum : int
        x_len : int
        x : mediumblob
        x_wo = null : mediumblob
        """

    def add(self, data_set_file, skip_duplicates=False, src_dir='database'):
        if len(self & dict(data_set_file=data_set_file)) > 0:
            if not skip_duplicates:
                raise ValueError(f"Data set {data_set_file} already exists in database.")

        data_set, n_videos, n_samples_per_video, video_width, video_height, pixel_size_um = load_raw_data_set(
            src_dir=src_dir, data_set_file=data_set_file)

        self.insert1(dict(data_set_file=data_set_file, n_videos=n_videos, n_samples_per_video=n_samples_per_video,
                          video_width=video_width, video_height=video_height, pixel_size_um=pixel_size_um),
                     skip_duplicates=skip_duplicates)

        for stimulus_id, group in data_set.groupby('stimulus_id'):
            for _, row in group.iterrows():
                bc_noise_sample = row.bc_noise_sample if 'bc_noise_sample' in row else 0
                rgc_noise_sample = row.rgc_noise_sample if 'rgc_noise_sample' in row else 0

                y_full = row.target_pos['xy']
                d_full = row.target_pos['d']
                p_full = y_to_p(y_full)
                p_sum = p_full.sum()
                x_len = row.response.shape[0]
                x = row.response
                x_wo = row.response_wo if "response_wo" in row else None

                if x_wo is not None:
                    if x.shape != x_wo.shape:
                        warnings.warn(f"{stimulus_id} x and x_wo don't have the same shape. Use hard coded cropping")
                        x_wo = x_wo[-x.shape[0]:]

                assert x.shape == x_wo.shape, "x and x_wo don't have the same shape"

                self.DataPoint().insert1(
                    dict(data_set_file=data_set_file, stimulus_id=stimulus_id,
                         bc_noise_sample=bc_noise_sample, rgc_noise_sample=rgc_noise_sample,
                         y_full=y_full, d_full=d_full, p_full=p_full, p_sum=p_sum, x_len=x_len, x=x, x_wo=x_wo),
                    skip_duplicates=skip_duplicates)

    @staticmethod
    def plot1(key=None, frame_i=None):
        if key is None:
            key = np.random.choice(DataSet.DataPoint().proj().fetch(as_dict=True))
        y_full, d_full, x, x_wo = (DataSet().DataPoint() & key).fetch1('y_full', 'd_full', 'x', 'x_wo')
        if frame_i is None:
            frame_i = np.random.randint(x.shape[0])
        plot_w_wo_and_diff(x, x_wo, y_full, d_full, frame_i=frame_i)


def plot_w_wo_and_diff(x, x_wo, y_full, d_full, frame_i=30):
    vmax = np.nanmax([np.max(x[frame_i]), np.max(x_wo[frame_i]) if x_wo is not None else np.nan])
    vmin = np.nanmin([np.min(x[frame_i]), np.min(x_wo[frame_i]) if x_wo is not None else np.nan])

    diff = x_wo[frame_i] - x[frame_i]
    vabsmax = np.nanmax(np.abs(diff))

    fig, axs = plt.subplots(ncols=3, figsize=(15, 4))
    fig.suptitle(f'xy: {y_full[frame_i]}\nd: {d_full[frame_i]}')

    ax = axs[0]
    im = ax.imshow(x[frame_i], vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax)
    ax.set_title('w')

    ax = axs[1]
    im = ax.imshow(x_wo[frame_i], vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax)
    ax.set_title('wo')

    ax = axs[2]
    im = ax.imshow(diff, vmin=-vabsmax, vmax=vabsmax, cmap='bwr')
    plt.colorbar(im, ax=ax, label='diff')
    ax.set_title(f'mean diff={np.mean(diff):.2f}')
    plt.show()


@dataset_schema
class DataSplit(dj.Manual):
    definition = """
    -> DataSet
    split_id : int
    ---
    f_train : float
    f_dev : float
    f_test : float
    seed : int
    """

    class SplitPoint(dj.Part):
        definition = """
        -> master
        -> DataSet.DataPoint
        ---
        split_kind : enum('train', 'dev', 'test', 'none')
        """

    def add_length_stratified(self, data_set_file, split_id, f_train, f_dev=0, seed=42, skip_duplicates=False):
        data_points = (DataSet.DataPoint & dict(data_set_file=data_set_file))
        n_samples_per_video = (DataSet & dict(data_set_file=data_set_file)).fetch1('n_samples_per_video')
        stimulus_ids, stimulus_lengths = data_points.fetch('stimulus_id', 'x_len')

        stimulus_ids = stimulus_ids[::n_samples_per_video]
        stimulus_lengths = stimulus_lengths[::n_samples_per_video]

        stimulus_ids_train, stimulus_ids_dev, stimulus_ids_test = get_split_indexes_equal_lengths(
            stimulus_ids, stimulus_lengths, f_train=f_train, f_dev=f_dev, seed=seed)

        self.insert1(dict(data_set_file=data_set_file, split_id=split_id, f_train=f_train, f_dev=f_dev,
                          f_test=1. - f_train - f_dev, seed=seed),
                     skip_duplicates=skip_duplicates)

        for kind, ids in zip(['train', 'dev', 'test'], [stimulus_ids_train, stimulus_ids_dev, stimulus_ids_test]):
            for data_point in (data_points & [{"stimulus_id": i} for i in ids]).proj():
                self.SplitPoint().insert1(dict(**data_point, split_id=split_id, split_kind=kind),
                                          skip_duplicates=skip_duplicates)

    def add_distance_stratified(self, data_set_file, split_id, seed=42, skip_duplicates=False):
        data_points = (DataSet.DataPoint & dict(data_set_file=data_set_file))
        df_data_points = flatten_datapoints(data_points.fetch(format='frame'), inplace=True)
        idxs_train, idxs_dev, idxs_test = split_flat_datapoints(
            df=df_data_points, n_train=7, n_dev=1, n_test=2, col_dist='d', q_dist=10, seed=seed, col_groups='trial_id')

        assert len(idxs_train) > 0
        assert len(idxs_dev) > 0
        assert len(idxs_test) > 0

        split = np.full(df_data_points.shape[0], 'loss', dtype=object)
        split[idxs_train] = 'train'
        split[idxs_dev] = 'dev'
        split[idxs_test] = 'test'

        df_data_points['split'] = split

        fig, ax = plt.subplots(1, 1, figsize=(4, 3))
        ax.boxplot(df_data_points.iloc[idxs_train].d, positions=[0])
        ax.boxplot(df_data_points.iloc[idxs_dev].d, positions=[1])
        ax.boxplot(df_data_points.iloc[idxs_test].d, positions=[2])
        plt.show()

        split_kinds = dict()
        for (index, stimulus_id), group in df_data_points.groupby(['index', 'stimulus_id']):
            assert np.unique(group.split).size == 1
            split_kinds[stimulus_id] = group.split.values[0]

        self.insert1(dict(
            data_set_file=data_set_file, split_id=split_id, f_train=0.7, f_dev=0.1, f_test=0.2, seed=seed),
            skip_duplicates=skip_duplicates)

        for data_point in data_points.proj():
            # set to none for non-cricket videos
            split_kind = split_kinds.get(data_point["stimulus_id"], 'none')

            self.SplitPoint().insert1(dict(
                **data_point, split_id=split_id, split_kind=split_kind), skip_duplicates=skip_duplicates)


@dataset_schema
class DataNorm(dj.Computed):
    definition = """
    -> DataSet
    ---
    x_mean : float
    x_std : float
    d_min : float
    d_max : float
    y_center : blob
    y_scale : blob
    """

    class NormPoint(dj.Part):
        definition = """
        -> master
        -> DataSet.DataPoint
        ---
        y_norm : blob
        d_norm : blob
        x_norm : mediumblob
        x_norm_wo: mediumblob
        """

    @property
    def key_source(self):
        return DataSet().proj()

    def make(self, key):
        dataset_tab = DataSet.DataPoint & key

        x = np.vstack(dataset_tab.fetch('x'))
        x_mean = np.nanmean(x)
        x_std = np.nanstd(x)

        d = np.concatenate(dataset_tab.fetch('d_full'))
        d_min = np.nanmin(d)
        d_max = np.nanmax(d)

        video_width, video_height, pixel_size_um = (DataSet & key).fetch1(
            'video_width', 'video_height', 'pixel_size_um')

        y_center = 0.5 * np.array([video_width, video_height])
        y_scale = np.array([video_width, video_height])

        self.insert1(dict(**key, x_mean=x_mean, x_std=x_std,
                          y_center=y_center, y_scale=y_scale,
                          d_min=d_min, d_max=d_max))

        for point_key in dataset_tab.proj():
            y_full = (dataset_tab & point_key).fetch1('y_full')
            y_norm = (y_full - y_center) / (y_scale / 2.)

            d_full = (dataset_tab & point_key).fetch1('d_full')
            d_norm = (d_full - d_min) / (d_max - d_min)

            x, x_wo = (dataset_tab & point_key).fetch1('x', 'x_wo')
            x_norm = (x - x_mean) / x_std
            x_norm_wo = (x_wo - x_mean) / x_std if x_wo is not None else None

            self.NormPoint().insert1(dict(
                **point_key, y_norm=y_norm, d_norm=d_norm, x_norm=x_norm, x_norm_wo=x_norm_wo))

    @staticmethod
    def plot1(key=None, frame_i=None):
        if key is None:
            key = np.random.choice(DataNorm.NormPoint().proj().fetch(as_dict=True))
        x, x_wo, p_full, y_full, d_full, d_norm = (DataSet.DataPoint() * DataNorm.NormPoint() & key).fetch1(
            'x_norm', 'x_norm_wo', 'p_full', 'y_full', 'd_full', 'd_norm')

        if frame_i is None:
            frame_i = np.random.randint(x.shape[0])

        plot_w_wo_and_diff(x, x_wo, y_full, d_full, frame_i=frame_i)
