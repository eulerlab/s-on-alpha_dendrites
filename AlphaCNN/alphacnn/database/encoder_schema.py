import gc
import os

import datajoint as dj
import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tqdm.notebook import tqdm

from alphacnn import paths
from alphacnn.model.cnn_model import prepare_gpu, parametrized_sigmoid
from alphacnn.utils.data_utils import load_var

prepare_gpu()

encoder_schema = dj.Schema()

def connect_to_database(dj_config_file, schema_name, create_schema=True, create_tables=True):
    dj.config.load(dj_config_file)
    dj.config['schema_name'] = schema_name
    print("schema_name:", dj.config['schema_name'])
    dj.conn()

    encoder_schema.activate(schema_name, create_schema=create_schema, create_tables=create_tables)


def predict_from_model(model, model_inputs, batch_size_frames=64, verbose=0):
    keras.backend.clear_session()
    gc.collect()
    split_idxs = np.cumsum([len(model_input) for model_input in model_inputs[:-1]])
    model_inputs = tf.convert_to_tensor(np.concatenate(model_inputs, axis=0), dtype=tf.float32)
    model_outputs = model.predict(model_inputs, batch_size=batch_size_frames, verbose=verbose).squeeze()
    model_outputs = np.split(model_outputs, split_idxs, axis=0)
    gc.collect()
    return model_outputs


@encoder_schema
class StimulusConfig(dj.Manual):
    @property
    def definition(self):
        definition = """
        stimulus_config_id : tinyint unsigned
        ---
        stimulus_config_file : varchar(191)
        stimulus_dict : longblob
        """
        return definition

    def add_stim(self, stimulus_config_id, stimulus_dict, stimulus_config_file="", skip_duplicates=True):
        self.insert1(dict(stimulus_config_id=stimulus_config_id,
                          stimulus_dict=stimulus_dict,
                          stimulus_config_file=stimulus_config_file),
                     skip_duplicates=skip_duplicates)


@encoder_schema
class StimulusIDs(dj.Computed):
    @property
    def definition(self):
        definition = """
        -> StimulusConfig
        stimulus_id : tinyint unsigned
        wo_cricket : tinyint unsigned
        ---
        stimulus_file : varchar(191)
        video_dict : longblob
        """
        return definition

    @property
    def key_source(self):
        return StimulusConfig().proj()

    def make(self, key):
        stimulus_dict = (StimulusConfig & key).fetch1('stimulus_dict')
        self.make_stim(key, stimulus_dict=stimulus_dict)

    def make_stim(self, key, stimulus_dict):
        video_dir = stimulus_dict['source']['path']
        if not os.path.isdir(video_dir):
            video_dir = os.path.join(paths.get_video_in_path(), video_dir)
        assert os.path.isdir(video_dir), video_dir

        if stimulus_dict['files'][0] == 'all':
            files = [f for f in os.listdir(video_dir) if str(f).endswith('left.mp4') or str(f).endswith('right.mp4')]
        else:
            files = stimulus_dict['files']

        create_videos_wo_cricket = 'path_wo_cricket' in stimulus_dict['source']

        if create_videos_wo_cricket:
            video_dir_wo = stimulus_dict['source']['path_wo_cricket']
            if not os.path.isdir(video_dir_wo):
                video_dir_wo = os.path.join(paths.get_video_in_path(), video_dir_wo)
            assert os.path.isdir(video_dir_wo)
        else:
            video_dir_wo = None

        for i, file in enumerate(tqdm(files)):
            self.insert1(dict(key, stimulus_id=i, stimulus_file=file,
                              video_dict=dict(video_path=os.path.join(video_dir, file)), wo_cricket=False))

            if create_videos_wo_cricket:
                if os.path.isfile(os.path.join(video_dir_wo, file)):
                    self.insert1(dict(key, stimulus_id=i, stimulus_file=file,
                                      video_dict=dict(video_path=os.path.join(video_dir_wo, file)), wo_cricket=True))
                else:
                    print(f'File {file} not found in {video_dir_wo}')


@encoder_schema
class Stimulus(dj.Computed):
    output_name = 'video'

    @property
    def definition(self):
        definition = f"""
        -> StimulusConfig
        -> StimulusIDs
        ---
        {self.output_name} : longblob
        target_pos : longblob
        """
        return definition

    @property
    def key_source(self):
        return StimulusConfig().proj() * StimulusIDs().proj()

    def make(self, key):
        video_dict = (StimulusIDs & key).fetch1('video_dict')

        file_dat = np.load(video_dict['video_path'])
        video = file_dat['video']
        target_pos = file_dat['cricket_xy']
        target_dist = file_dat['cricket_d']

        if video.nbytes * 1e-9 > 1:
            raise MemoryError(f'Video size is {video.nbytes * 1e-9:.2f} GB. This might take a while.')

        self.insert1(dict(key, video=video, target_pos=dict(xy=target_pos, d=target_dist)))

    @classmethod
    def plot1(cls, key=None, n_rows=4, n_cols=4, drop_first_n=0, **kwargs):
        key = ((cls & (key or dict())).proj()).fetch(format='frame').sample(1)
        stimulus_id, video, target_pos = (cls & key).fetch1('stimulus_id', cls.output_name, 'target_pos')

        from alphacnn.visualize import plot_stimulus

        axs, fis = plot_stimulus.plot_video_frames(video[drop_first_n:], n_rows=n_rows, n_cols=n_cols, **kwargs)
        plot_stimulus.plot_target_positions(target_pos['xy'][drop_first_n:], axs=axs, fis=fis, c='darkred')
        plt.tight_layout()
        plt.show()


def load_rf_dict(file, stimulus_config_id=None):
    path = os.path.join(paths.VIDEO_ROOT, 'cluster_strfs', file)
    rf_dict = load_var(path)
    if stimulus_config_id is not None:
        stimulus_dict = (StimulusConfig & dict(stimulus_config_id=stimulus_config_id)).fetch1('stimulus_dict')
        assert stimulus_dict['stimulus']['pixel_size'] == rf_dict['pixel_size']

    return rf_dict

@encoder_schema
class BCsRfConfig(dj.Manual):

    @property
    def definition(self):
        definition = """
        -> StimulusConfig
        bc_srf_config_id : tinyint unsigned
        ---
        bc_srf_config_name : varchar(191)
        bc_cdist : int
        bc_srf : longblob
        bc_srf_bias : longblob
        """
        return definition

    def add_from_file(self, bc_cdist, file, bc_srf_config_id, stimulus_config_id, bc_srf_config_name, skip_duplicates=True):
        rf_dict = load_rf_dict(file, stimulus_config_id)
        srf = rf_dict['srf'].copy().astype("float32")
        bias = np.float32(rf_dict['bias'])
        self.insert1(dict(stimulus_config_id=stimulus_config_id, bc_srf_config_id=bc_srf_config_id,
                          bc_srf_config_name=bc_srf_config_name,
                          bc_cdist=bc_cdist, bc_srf=srf, bc_srf_bias=bias), skip_duplicates=skip_duplicates)

    @classmethod
    def plot(cls, restriction=None):
        id_list, srf_list, = (cls & (restriction or dict())).fetch('bc_srf_config_id', 'bc_srf')

        from alphacnn.response.receptive_fields import plot_srf
        n_rfs = len(srf_list)
        fig, axs = plt.subplots(1, n_rfs, figsize=(12, 3), sharex='col', sharey='col')
        for ax, name, srf in zip(axs, id_list, srf_list):
            ax.set_title(name)
            plot_srf(srf, ax=ax, cb=True)


@encoder_schema
class BCSpatialRFOutput(dj.Computed):
    input_table = Stimulus
    output_name = 'bc_srf_output'

    @property
    def definition(self):
        definition = f"""
        -> StimulusConfig
        -> BCsRfConfig
        -> self.input_table
        ---
        {self.output_name} : longblob
        """
        return definition

    @property
    def key_source(self):
        return StimulusConfig().proj() * BCsRfConfig().proj()

    def make(self, key, batch_size=1, batch_size_frames=64):
        stimulus_dict = (StimulusConfig & key).fetch1('stimulus_dict')
        bc_cdist, srf, bc_srf_bias = (BCsRfConfig & key).fetch1('bc_cdist', 'bc_srf', 'bc_srf_bias')
        input_keys = (self.input_table & key).proj().fetch(as_dict=True)
        video = (self.input_table & input_keys[0]).fetch1(self.input_table.output_name)
        if video.ndim == 3:
            video = video[:, :, :, np.newaxis]

        pixel_size = stimulus_dict['stimulus']['pixel_size']
        rf_dia_n = srf.shape[0]

        assert bc_cdist % pixel_size == 0
        bc_cdist_n = bc_cdist // pixel_size

        model = self.init_model(input_shape=video.shape, rf_dia_n=rf_dia_n, cdist_n=bc_cdist_n)
        self.init_model_weights(model, srf, bc_srf_bias)

        bar = tqdm(total=len(input_keys), desc=f"Batch_size={batch_size}")
        batches = np.array_split(input_keys, np.ceil(len(input_keys) / batch_size))

        for batch_keys in batches:
            batch_inputs = []

            for input_key in batch_keys:
                video = (self.input_table & input_key).fetch1(self.input_table.output_name)
                if video.ndim == 3:
                    video = video[:, :, :, np.newaxis]
                batch_inputs.append(video)

            batch_outputs = predict_from_model(
                model=model, model_inputs=batch_inputs, batch_size_frames=batch_size_frames)

            for input_key, batch_output in zip(batch_keys, batch_outputs):
                input_key = input_key.copy()
                input_key.update(key)
                self.insert1({**input_key, self.output_name: batch_output})
                bar.update(1)

    @staticmethod
    def init_model(input_shape, rf_dia_n, cdist_n):
        keras.backend.clear_session()

        x_in = keras.layers.Input(shape=(input_shape[1], input_shape[2], input_shape[3]))
        x_out = keras.layers.Conv2D(
            name='BC_sRF', filters=1, kernel_size=(rf_dia_n, rf_dia_n), strides=(cdist_n, cdist_n),
            activation=None, trainable=False, padding='valid', use_bias=True)(x_in)

        model = keras.models.Model(x_in, x_out)

        return model

    @staticmethod
    def init_model_weights(model, srf, bias):
        model.get_layer('BC_sRF').set_weights([srf[:, :, np.newaxis, np.newaxis], np.array([bias])])

    @classmethod
    def plot1(cls, key=None, n_rows=1, n_cols=4, **kwargs):
        key = ((cls & (key or dict())).proj()).fetch(format='frame').sample(1)
        data_output = (cls & key).fetch1(cls.output_name)

        from alphacnn.visualize import plot_stimulus

        plot_stimulus.plot_video_frames(data_output, n_rows=n_rows, n_cols=n_cols, **kwargs)
        plt.tight_layout()
        plt.show()


@encoder_schema
class BCRectConfig(dj.Manual):

    @property
    def definition(self):
        definition = """
        -> StimulusConfig
        bc_rect_config_id : tinyint unsigned
        ---
        bc_rect_config_name : varchar(191)
        bc_nl : longblob
        """
        return definition

    def add_from_data(self, nl, bc_rect_config_id, stimulus_config_id, bc_rect_config_name, skip_duplicates=True):
        key = dict(stimulus_config_id=stimulus_config_id, bc_rect_config_id=bc_rect_config_id,
                   bc_rect_config_name=bc_rect_config_name)
        self.insert1(dict(**key, bc_nl=nl), skip_duplicates=skip_duplicates)

    @classmethod
    def plot(cls, restriction=None):
        bc_config_id_list, nl_list, = (cls & (restriction or dict())).fetch('bc_config_id', 'nl_list')
        n_rfs = len(nl_list)
        fig, axs = plt.subplots(n_rfs, 1, figsize=(12, 3 * n_rfs), sharex='col', sharey='col')
        for ax, bc_nl in zip(axs, nl_list):
            if isinstance(bc_nl, dict):
                ax.text(0.5, 0.5, f"{bc_nl['name']}:\n{bc_nl['params']}")
            else:
                sigmoid_x = np.linspace(-2, 5, 200).astype("float32")
                ax.plot(sigmoid_x, parametrized_sigmoid(sigmoid_x, *bc_nl.astype("float32")))


@encoder_schema
class BCRectOutput(dj.Computed):
    input_table = BCSpatialRFOutput
    input_config_table = BCsRfConfig
    output_name = 'bc_rect_output'

    @property
    def definition(self):
        definition = f"""
        -> BCRectConfig
        -> self.input_table
        ---
        {self.output_name} : longblob
        """
        return definition

    @property
    def key_source(self):
        return BCRectConfig().proj() * self.input_config_table().proj()

    def make(self, key, batch_size=1, batch_size_frames=64):
        nl = (BCRectConfig & key).fetch1('bc_nl')
        input_keys = (self.input_table * BCRectConfig & key).proj().fetch(as_dict=True)
        self.compute_and_insert(input_keys, nl, batch_size=batch_size, batch_size_frames=batch_size_frames)

    @classmethod
    def populate_missing(cls, make_kwargs=None):
        make_kwargs = make_kwargs or dict()
        for key in tqdm(cls().key_source):
            nl = (BCRectConfig & key).fetch1('bc_nl')
            input_keys = ((cls.input_table * BCRectConfig & key).proj() - cls().proj()).fetch(as_dict=True)
            if len(input_keys) == 0:
                continue
            cls().compute_and_insert(input_keys, nl, allow_direct_insert=True, **make_kwargs)

    def compute_and_insert(self, input_keys, nl, batch_size=1, batch_size_frames=64, allow_direct_insert=None):
        input_shape = (self.input_table & input_keys[0]).fetch1(self.input_table.output_name).shape

        model = self.init_model_and_weights(input_shape=input_shape, nl=nl)

        bar = tqdm(total=len(input_keys), desc=f"Batch_size={batch_size}")
        batches = np.array_split(input_keys, np.ceil(len(input_keys) / batch_size))

        for batch_keys in batches:
            batch_inputs = [(self.input_table & input_key).fetch1(self.input_table.output_name) for input_key in
                            batch_keys]

            batch_outputs = predict_from_model(
                model=model, model_inputs=batch_inputs, batch_size_frames=batch_size_frames)

            for input_key, batch_output in zip(batch_keys, batch_outputs):
                self.insert1({**input_key, self.output_name: batch_output}, allow_direct_insert=allow_direct_insert)
                bar.update(1)

    @staticmethod
    def init_model_and_weights(input_shape, nl):
        keras.backend.clear_session()
        from alphacnn.model.cnn_model import parametrized_sigmoid
        from functools import partial

        x_in = keras.layers.Input(shape=input_shape[1:])

        if isinstance(nl, dict):
            nl_name = nl['name']
            if nl_name in ['lin', 'none', 'linear']:
                x_out = x_in
            else:
                x_out = keras.layers.Activation(nl_name, name='BC_rect', **nl['params'])(x_in)
        else:
            # Assume parametrized_sigmoid per default
            if len(nl) == 4:
                k, q, b, v = nl.astype("float32")
                nl_fun = partial(parametrized_sigmoid, k=k, q=q, b=b, v=v)
            elif len(nl) == 5:
                k, q, b, v, d = nl.astype("float32")
                nl_fun = partial(parametrized_sigmoid, k=k, q=q, b=b, v=v, d=d)
            else:
                raise NotImplementedError

            x_out = keras.layers.Lambda(lambda x: nl_fun(x), name='BC_rect')(x_in)

        model = keras.models.Model(x_in, x_out)

        return model

    @classmethod
    def plot1(cls, key=None, n_rows=1, n_cols=4, **kwargs):
        from alphacnn.visualize import plot_stimulus

        key = ((cls & (key or dict())).proj()).fetch(format='frame').sample(1)
        data_output = (cls & key).fetch1(cls.output_name)
        plot_stimulus.plot_video_frames(data_output, n_rows=n_rows, n_cols=n_cols, **kwargs)
        plt.tight_layout()
        plt.show()


@encoder_schema
class BCNoiseConfigCore(dj.Manual):
    input_config_table = BCsRfConfig

    @property
    def definition(self):
        definition = """
        -> StimulusConfig
        -> self.input_config_table
        bc_noise_id : tinyint unsigned
        ---
        bc_noise_name : varchar(32)
        bc_noise_dict : longblob
        bc_core_seed : int
        """
        return definition

    def add(self, stimulus_config_id, noise_id, noise_name, noise_dict, core_seed, skip_duplicates=True, **kwargs):
        self.insert1(dict(bc_noise_id=noise_id, bc_noise_name=noise_name, stimulus_config_id=stimulus_config_id,
                          bc_noise_dict=noise_dict, bc_core_seed=core_seed, **kwargs),
                     skip_duplicates=skip_duplicates)


@encoder_schema
class BCNoiseConfig(dj.Computed):

    @property
    def definition(self):
        definition = """
        -> BCNoiseConfigCore
        -> StimulusIDs
        ---
        bc_sub_seed : int
        """
        return definition

    @property
    def key_source(self):
        return StimulusConfig().proj() * BCNoiseConfigCore().proj()

    def make(self, key):
        bc_core_seed = (BCNoiseConfigCore & key).fetch1('bc_core_seed')
        stimulus_ids, wo_crickets = (StimulusIDs & key).fetch('stimulus_id', 'wo_cricket')

        # Add different seeds for all movies
        np.random.seed(bc_core_seed)
        seeds = np.atleast_1d(np.random.randint(0, np.iinfo(np.int32).max, size=len(stimulus_ids), dtype=np.int32))

        for stimulus_id, wo_cricket, seed in zip(stimulus_ids, wo_crickets, seeds):
            self.insert1(dict(**key, stimulus_id=stimulus_id, wo_cricket=wo_cricket, bc_sub_seed=seed))


@encoder_schema
class BCNoiseSeeds(dj.Manual):

    @property
    def definition(self):
        definition = """
        -> BCNoiseConfig
        bc_noise_sample : int
        ---
        bc_noise_seed : int
        """
        return definition

    def add_samples(self, stimulus_config_id, bc_srf_config_id, bc_noise_id, n_samples_tot: int):
        main_key = dict(stimulus_config_id=stimulus_config_id, bc_srf_config_id=bc_srf_config_id,
                        bc_noise_id=bc_noise_id)

        stimulus_ids, wo_crickets = (StimulusIDs & main_key).fetch('stimulus_id', 'wo_cricket')

        for stimulus_id, wo_cricket in zip(stimulus_ids, wo_crickets):
            key = dict(**main_key, stimulus_id=stimulus_id, wo_cricket=wo_cricket)
            bc_sub_seed = (BCNoiseConfig & key).fetch1('bc_sub_seed')
            n_samples_old = len((self & key).fetch())

            if n_samples_old > n_samples_tot:
                raise ValueError(f"Fewer samples {n_samples_tot} requested than present {n_samples_old}.")

            np.random.seed(bc_sub_seed)
            seeds = np.atleast_1d(np.random.randint(0, np.iinfo(np.int32).max, size=n_samples_tot, dtype=np.int32))

            for sample, seed in enumerate(seeds):
                sub_key = dict(**key, bc_noise_sample=sample, bc_noise_seed=seed)
                self.insert1(sub_key, skip_duplicates=True)


@encoder_schema
class BCNoiseOutput(dj.Computed):
    input_table = BCRectOutput
    input_config_table = BCRectConfig
    output_name = 'bc_noise_output'

    @property
    def definition(self):
        definition = f"""
        -> BCNoiseConfig
        -> BCNoiseSeeds
        -> self.input_table
        ---
        {self.output_name} : longblob
        """
        return definition

    @property
    def key_source(self):
        return StimulusConfig().proj() * BCNoiseConfigCore().proj() * self.input_config_table().proj()

    def make(self, key):
        noise_dict = (BCNoiseConfigCore & key).fetch1('bc_noise_dict')

        input_tab = BCNoiseSeeds() * self.input_table() * StimulusIDs & key
        input_keys = input_tab.proj().fetch(as_dict=True)

        if len(input_keys) == 0:
            return

        input_shape = (input_tab & input_keys[0]).fetch1(self.input_table.output_name).shape

        model = self.init_model(input_shape=input_shape, stddev=noise_dict['bc_stddev'])

        for input_key in tqdm(input_keys):
            model_input = (self.input_table & input_key).fetch1(self.input_table.output_name)
            seed = (BCNoiseSeeds & input_key).fetch1('bc_noise_seed')

            keras.utils.set_random_seed(int(seed))
            data_output = model(model_input).numpy().squeeze()

            input_key = input_key.copy()
            input_key.update(key)

            self.insert1({**input_key, self.output_name: data_output})

    @classmethod
    def plot1(cls, key=None, n_rows=1, n_cols=4, **kwargs):
        from alphacnn.visualize import plot_stimulus

        key = ((cls & (key or dict())).proj()).fetch(format='frame').sample(1)
        data_output = (cls & key).fetch1(cls.output_name)

        plot_stimulus.plot_video_frames(data_output, n_rows=n_rows, n_cols=n_cols, **kwargs)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def init_model(input_shape, stddev):
        from alphacnn.layers.noise_layers import ActiveGaussianNoise
        keras.backend.clear_session()

        x_in = keras.layers.Input(shape=input_shape[1:])
        x_out = ActiveGaussianNoise(stddev=stddev, seed=None, trainable=False, name='Noise')(x_in)
        model = keras.models.Model(x_in, x_out)

        return model


@encoder_schema
class RGCSynapticWeights(dj.Manual):

    @property
    def definition(self):
        definition = """
        rgc_id : tinyint unsigned
        bc_srf_config_id_1 : tinyint unsigned
        bc_rect_config_id_1 : tinyint unsigned
        bc_srf_config_id_2 : tinyint unsigned
        bc_rect_config_id_2 : tinyint unsigned
        ---
        rgc_name : varchar(32)
        bc_srf_config_name_1 : varchar(32)
        bc_rect_config_name_1 : varchar(32)
        bc_srf_config_name_2 : varchar(32)
        bc_rect_config_name_2 : varchar(32)
        rgc_cdist : int
        rgc_rf_dia : int
        config_weights_1 : longblob
        rgc_synaptic_weights_1 : longblob
        config_weights_2 : longblob
        rgc_synaptic_weights_2 : longblob
        """
        return definition

    def add(self, rgc_id, rgc_name, rgc_cdist, rgc_rf_dia,
            bc_srf_config_id_1, bc_rect_config_id_1, config_weights_1,
            bc_srf_config_id_2, bc_rect_config_id_2, config_weights_2):
        from alphacnn.response.synapses import w_syn_truncated_normal, compute_rgc_weights

        bc_key1 = dict(bc_srf_config_id=bc_srf_config_id_1, bc_rect_config_id=bc_rect_config_id_1)
        bc_key2 = dict(bc_srf_config_id=bc_srf_config_id_2, bc_rect_config_id=bc_rect_config_id_2)

        bc_cdists = (BCsRfConfig & [bc_key1, bc_key2]).fetch('bc_cdist')
        assert np.all(bc_cdists == bc_cdists[0]), 'Incomplete config. BCs have different cdist.'
        bc_cdist = bc_cdists[0]

        bc_srf_config_name_1 = (BCsRfConfig & bc_key1).fetch1('bc_srf_config_name')
        bc_rect_config_name_1 = (BCRectConfig & bc_key1).fetch1('bc_rect_config_name')
        bc_srf_config_name_2 = (BCsRfConfig & bc_key2).fetch1('bc_srf_config_name')
        bc_rect_config_name_2 = (BCRectConfig & bc_key2).fetch1('bc_rect_config_name')

        assert rgc_rf_dia % bc_cdist == 0
        rgc_rf_dia_n = rgc_rf_dia // bc_cdist

        weights_1 = compute_rgc_weights(
            rgc_rf_dia_n, bc_cdist, weight_fun=w_syn_truncated_normal, weight_kws=config_weights_1)
        weights_2 = compute_rgc_weights(
            rgc_rf_dia_n, bc_cdist, weight_fun=w_syn_truncated_normal, weight_kws=config_weights_2)

        self.insert1(dict(
            rgc_id=rgc_id, rgc_name=rgc_name, rgc_cdist=rgc_cdist, rgc_rf_dia=rgc_rf_dia,
            bc_srf_config_id_1=bc_srf_config_id_1, bc_rect_config_id_1=bc_rect_config_id_1,
            bc_srf_config_id_2=bc_srf_config_id_2, bc_rect_config_id_2=bc_rect_config_id_2,
            bc_srf_config_name_1=bc_srf_config_name_1, bc_rect_config_name_1=bc_rect_config_name_1,
            bc_srf_config_name_2=bc_srf_config_name_2, bc_rect_config_name_2=bc_rect_config_name_2,
            config_weights_1=config_weights_1, config_weights_2=config_weights_2,
            rgc_synaptic_weights_1=weights_1, rgc_synaptic_weights_2=weights_2,
        ), skip_duplicates=True)

@encoder_schema
class RGCSynapticInputs(dj.Computed):
    input_table = BCNoiseOutput
    threshold = 0.

    @property
    def definition(self):
        definition = """
        -> RGCSynapticWeights
        -> self.input_table.proj(bc_srf_config_id_1='bc_srf_config_id', bc_rect_config_id_1='bc_rect_config_id')
        -> self.input_table.proj(bc_srf_config_id_2='bc_srf_config_id', bc_rect_config_id_2='bc_rect_config_id')
        ---
        rgc_synaptic_inputs : longblob
        """
        return definition

    @property
    def key_source(self):
        return (RGCSynapticWeights().proj() * StimulusConfig().proj() *
                BCNoiseConfigCore().proj(bc_srf_config_id_1='bc_srf_config_id'))

    def make(self, key, batch_size=1, batch_size_frames=64):
        bc_cdists = (BCsRfConfig & key).fetch('bc_cdist')
        assert np.all(bc_cdists == bc_cdists[0])
        bc_cdist = bc_cdists[0]

        rgc_cdist = (RGCSynapticWeights & key).fetch1('rgc_cdist')
        rgc_synaptic_weights = np.stack((RGCSynapticWeights & key).fetch1(
            'rgc_synaptic_weights_1', 'rgc_synaptic_weights_2'), axis=-1)

        bc_srf_config_id_1, bc_rect_config_id_1 = (RGCSynapticWeights & key).fetch1(
            'bc_srf_config_id_1', 'bc_rect_config_id_1')
        bc_srf_config_id_2, bc_rect_config_id_2 = (RGCSynapticWeights & key).fetch1(
            'bc_srf_config_id_2', 'bc_rect_config_id_2')

        bc_key1 = dict(bc_srf_config_id=bc_srf_config_id_1, bc_rect_config_id=bc_rect_config_id_1)
        bc_key2 = dict(bc_srf_config_id=bc_srf_config_id_2, bc_rect_config_id=bc_rect_config_id_2)

        map1 = dict(bc_srf_config_id_1='bc_srf_config_id', bc_rect_config_id_1='bc_rect_config_id',
                    bc_rect_output_1=self.input_table.output_name)

        map2 = dict(bc_srf_config_id_2='bc_srf_config_id', bc_rect_config_id_2='bc_rect_config_id',
                    bc_rect_output_2=self.input_table.output_name)

        input_tab = (self.input_table & bc_key1).proj(**map1) * (self.input_table & bc_key2).proj(**map2)

        input_keys = (input_tab & key).proj().fetch(as_dict=True)

        if len(input_keys) == 0:
            raise ValueError(f'No input keys found for key={key}')

        input_shape = (input_tab & input_keys[0]).fetch1('bc_rect_output_1').shape

        assert rgc_cdist % bc_cdist == 0
        rgc_cdist_n = rgc_cdist // bc_cdist

        model = self.init_model(input_shape=input_shape, rf_dia_n=rgc_synaptic_weights.shape[0], cdist_n=rgc_cdist_n,
                                threshold=self.threshold)
        self.init_model_weights(model=model, weights=rgc_synaptic_weights)

        bar = tqdm(total=len(input_keys), desc=f"Batch_size={batch_size}")
        batches = np.array_split(input_keys, np.ceil(len(input_keys) / batch_size))

        for batch_keys in batches:
            batch_inputs = [np.stack((input_tab & input_key).fetch1(
                'bc_rect_output_1', 'bc_rect_output_2'), axis=-1)[:, :, :, :, np.newaxis] for input_key in batch_keys]

            batch_outputs = predict_from_model(
                model=model, model_inputs=batch_inputs, batch_size_frames=batch_size_frames)

            for input_key, model_output in zip(batch_keys, batch_outputs):
                self.insert1(dict(**input_key, rgc_id=key['rgc_id'], rgc_synaptic_inputs=model_output))
                bar.update()

    @staticmethod
    def init_model(input_shape, rf_dia_n, cdist_n, threshold=0.):
        keras.backend.clear_session()
        x_in = keras.layers.Input(shape=(input_shape[1], input_shape[2], 2, 1))
        x_threshold = keras.layers.ReLU(trainable=False, threshold=threshold)(x_in)
        x_out = keras.layers.Conv3D(
            name='RGC_in', filters=1, kernel_size=(rf_dia_n, rf_dia_n, 2), strides=(cdist_n, cdist_n, 1),
            activation=None, trainable=False, padding='valid', use_bias=False)(x_threshold)

        model = keras.models.Model(x_in, x_out)

        return model

    @staticmethod
    def init_model_weights(model, weights):
        model.get_layer('RGC_in').set_weights([weights[:, :, :, np.newaxis, np.newaxis]])

    @classmethod
    def plot1(cls, key=None, n_rows=1, n_cols=4, **kwargs):
        key = ((cls & (key or dict())).proj()).fetch(format='frame').sample(1)
        rgc_synaptic_inputs = (cls & key).fetch1('rgc_synaptic_inputs')

        from alphacnn.visualize import plot_stimulus

        plot_stimulus.plot_video_frames(rgc_synaptic_inputs, n_rows=n_rows, n_cols=n_cols, **kwargs)
        plt.tight_layout()
        plt.show()