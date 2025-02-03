import numpy as np
from joblib import dump
from keras.src.initializers.initializers import Constant
from sklearn.linear_model import LinearRegression, RidgeCV, LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import StandardScaler
from tqdm.keras import TqdmCallback

from alphacnn.database.dataset_utils import y_to_p
from alphacnn.decoder.decoder_utils import EarlyStoppingWarmup

_BATCH_SIZE = 16384
_EPOCHS = 1000
_MIN_EPOCHS = 200
_PATIENCE = 10


#  Debug values
# EPOCHS = 5
# MIN_EPOCHS = 2
# PATIENCE = 1


def train_frame_decoder(
        n_ensemble, dpos_kind, dpos_params, dpres_kind, dpres_params,
        X_train, y_train, d_train, X_dev=None, y_dev=None, d_dev=None, norm_data=False, seed=42):
    if n_ensemble > 1:
        print(f'Get frame decoder ensemble: n={n_ensemble}')
        decoder = FrameDecoderEnsemble(
            n_ensemble=n_ensemble, dpos_kind=dpos_kind, dpres_kind=dpres_kind,
            X_train=X_train, y_train=y_train, d_train=d_train,
            X_dev=X_dev, y_dev=y_dev, d_dev=d_dev,
            dpos_params=dpos_params, dpres_params=dpres_params, norm_data=norm_data, core_seed=seed)
        history = [(d.history_pos, d.history_pres) for d in decoder.decoders]
    else:
        print('Get frame decoder')
        decoder = FrameDecoder(
            dpos_kind=dpos_kind, dpres_kind=dpres_kind,
            X_train=X_train, y_train=y_train, d_train=d_train,
            X_dev=X_dev, y_dev=y_dev, d_dev=d_dev,
            dpos_params=dpos_params, dpres_params=dpres_params, norm_data=norm_data, seed=seed)
        history = [(decoder.history_pos, decoder.history_pres)]

    return decoder, history


class FrameDecoder:
    def __init__(self, dpos_kind, dpres_kind, X_train, y_train, d_train, X_dev=None, y_dev=None, d_dev=None,
                 dpos_params=None, dpres_params=None, norm_data=True, seed=42):
        self.dpos_kind = dpos_kind
        self.dpres_kind = dpres_kind

        self.X_train = X_train
        self.y_train = y_train
        self.d_train = d_train

        self.X_dev = X_dev
        self.y_dev = y_dev
        self.d_dev = d_dev

        self.x_scaler = StandardScaler() if norm_data else None
        self.y_scaler = StandardScaler() if norm_data else None
        self.d_scaler = StandardScaler() if norm_data else None

        self.pos_decoder = get_pos_decoder_and_params(
            kind=dpos_kind, params=dpos_params, X_shape=self.X_train.shape)

        if np.all(y_to_p(self.y_train)):
            self.pres_decoder = DummyPresenceDecoder()
        else:
            self.pres_decoder = get_pres_decoder_and_params(
                kind=dpres_kind, params=dpres_params, p_train=y_to_p(self.y_train), X_shape=self.X_train.shape)

        np.random.seed(seed)
        if dpos_kind == 'cnn' or dpres_kind == 'cnn':
            import tensorflow as tf
            tf.random.set_seed(int(seed))

        self.history_pos, self.history_pres = self.train()

    def train(self):
        p_train = y_to_p(self.y_train)
        p_dev = y_to_p(self.y_dev)

        print('Training POS decoder')
        history_pos = self._train_pos(
            X_train=self.X_train[p_train], y_train=self.y_train[p_train], d_train=self.d_train[p_train],
            X_dev=self.X_dev[p_dev], y_dev=self.y_dev[p_dev], d_dev=self.d_dev[p_dev])
        print('Training PRES decoder')
        history_pres = self._train_pres(
            X_train=self.X_train, p_train=p_train, d_train=self.d_train,
            X_dev=self.X_dev, p_dev=p_dev, d_dev=self.d_dev)

        return history_pos, history_pres

    def _train_pos(self, X_train, y_train, d_train, X_dev=None, y_dev=None, d_dev=None):
        X_scaled = self.x_scaler.fit_transform(X_train) if self.x_scaler is not None else X_train
        y_scaled = self.y_scaler.fit_transform(y_train) if self.y_scaler is not None else y_train

        if X_dev is None:
            assert not isinstance(self.pos_decoder, CNNModel)
            history = self.pos_decoder.fit(X=X_scaled, y=y_scaled)
        else:
            X_dev_scaled = self.x_scaler.transform(X_dev) if self.x_scaler is not None else X_dev
            y_dev_scaled = self.y_scaler.transform(y_dev) if self.y_scaler is not None else y_dev

            if isinstance(self.pos_decoder, CNNModel):
                # d_scaled = self.d_scaler.fit_transform(d_train) if self.d_scaler is not None else d_train
                # d_dev_scaled = self.d_scaler.transform(d_dev) if self.d_scaler is not None else d_dev
                history = self.pos_decoder.fit(X=X_scaled, y=y_scaled, X_dev=X_dev_scaled, y_dev=y_dev_scaled)
            else:
                history = self.pos_decoder.fit(X=np.concatenate([X_scaled, X_dev_scaled], axis=0),
                                               y=np.concatenate([y_scaled, y_dev_scaled], axis=0))
        return history

    def _train_pres(self, X_train, p_train, d_train, X_dev=None, p_dev=None, d_dev=None):
        assert np.all(p_train == p_train.astype(bool))
        X_scaled = self.x_scaler.fit_transform(X_train) if self.x_scaler is not None else X_train

        if np.unique(p_train).size == 1:
            self.pres_decoder = DummyPresenceDecoder()
            history = dict()
        else:
            if X_dev is None:
                history = self.pres_decoder.fit(X=X_scaled, y=p_train.astype(bool))
            else:
                X_dev_scaled = self.x_scaler.transform(X_dev) if self.x_scaler is not None else X_dev
                # d_scaled = self.d_scaler.fit_transform(d_train) if self.d_scaler is not None else d_train
                # d_dev_scaled = self.d_scaler.transform(d_dev) if self.d_scaler is not None else d_dev
                if isinstance(self.pres_decoder, CNNModel):
                    history = self.pres_decoder.fit(
                        X=X_scaled, y=p_train.astype(bool), X_dev=X_dev_scaled, y_dev=p_dev.astype(bool))
                else:
                    history = self.pres_decoder.fit(
                        X=np.concatenate([X_scaled, X_dev_scaled], axis=0),
                        y=np.concatenate([p_train.astype(bool), p_dev.astype(bool)], axis=0))

        return history

    def eval(self, X_test, y_test=None):
        assert X_test.ndim == 2 or X_test.ndim == 3, X_test.ndim
        if y_test is not None:
            assert y_test.ndim == 2, y_test.ndim
            assert X_test.shape[0] == y_test.shape[0]

        X_scaled = self.x_scaler.transform(X_test) if self.x_scaler is not None else X_test
        y_pred = self.pos_decoder.predict(X_scaled)
        if self.y_scaler is not None:
            y_pred = self.y_scaler.inverse_transform(y_pred)

        p_pred = self.pres_decoder.predict(X_scaled)

        if y_test is not None:
            y_errors = np.sum((y_test - y_pred) ** 2, axis=1) ** 0.5
            p_errors = np.abs(y_to_p(y_test).astype(int) - (p_pred > 0.5).astype(int))
            return y_pred, y_errors, p_pred, p_errors
        else:
            return y_pred, p_pred

    def eval_and_summarize(self, X_test, y_test):
        y_pred_train, y_errors_train, p_pred_train, p_errors_train = self.eval(X_test=self.X_train, y_test=self.y_train)
        y_pred_test, y_errors_test, p_pred_test, p_errors_test = self.eval(X_test=X_test, y_test=y_test)

        decoder_summary = dict()
        decoder_summary['X_train'] = self.X_train
        decoder_summary['X_test'] = X_test

        decoder_summary['y_train'] = self.y_train
        decoder_summary['y_test'] = y_test

        decoder_summary['y_pred_train'] = y_pred_train
        decoder_summary['y_errors_train'] = y_errors_train
        decoder_summary['p_pred_train'] = p_pred_train
        decoder_summary['p_errors_train'] = p_errors_train

        decoder_summary['y_pred_test'] = y_pred_test
        decoder_summary['y_errors_test'] = y_errors_test
        decoder_summary['p_pred_test'] = p_pred_test
        decoder_summary['p_errors_test'] = p_errors_test

        return decoder_summary

    def save_to_file(self, model_path):
        if self.dpos_kind == 'cnn':
            self.pos_decoder.model.save(filepath=model_path + '_pos')
        else:
            dump(self.pos_decoder, model_path + '_pos.joblib')

        if self.dpres_kind == 'cnn':
            self.pres_decoder.model.save(filepath=model_path + '_pres')
        else:
            dump(self.pres_decoder, model_path + '_pres.joblib')

    def load_from_file(self, model_path):
        raise NotImplementedError()


class FrameDecoderEnsemble:
    def __init__(self, dpos_kind, dpres_kind, X_train, y_train, d_train, X_dev=None, y_dev=None, d_dev=None,
                 dpos_params=None, dpres_params=None, norm_data=True, n_ensemble=3, core_seed=42):
        np.random.seed(core_seed)
        self.seeds = np.random.randint(0, np.iinfo(np.int32).max, size=n_ensemble, dtype=np.int32)
        self.decoders = []
        for seed in self.seeds:
            self.decoders.append(
                FrameDecoder(dpos_kind=dpos_kind, dpres_kind=dpres_kind,
                             X_train=X_train, y_train=y_train, d_train=d_train, X_dev=X_dev, y_dev=y_dev, d_dev=d_dev,
                             dpos_params=dpos_params, dpres_params=dpres_params, norm_data=norm_data,
                             seed=seed))

    def eval(self, X_test, y_test=None):
        y_preds = []
        p_preds = []

        for decoder in self.decoders:
            y_pred_i, p_pred_i = decoder.eval(X_test=X_test)
            y_preds.append(y_pred_i[np.newaxis, :])
            p_preds.append(p_pred_i[np.newaxis, :])

        y_pred = np.mean(np.concatenate(y_preds, axis=0), axis=0)
        p_pred = np.mean(np.concatenate(p_preds, axis=0), axis=0)

        if y_test is not None:
            y_errors = np.sum((y_test - y_pred) ** 2, axis=1) ** 0.5
            p_errors = np.abs(y_to_p(y_test).astype(int) - (p_pred > 0.5).astype(int))
            return y_pred, y_errors, p_pred, p_errors
        else:
            return y_pred, p_pred

    def save_to_file(self, model_path):
        for i, decoder in enumerate(self.decoders):
            decoder.save_to_file(model_path=model_path + f'_{i}')

    def load_from_file(self, filepath):
        raise NotImplementedError()


class DummyPresenceDecoder:
    def fit(self, *args, **kwargs):
        history = dict()
        return history

    def predict(self, X, *args, **kwargs):
        y_pred = np.ones(X.shape[0], dtype=bool)
        return y_pred


class DummyFrameDecoder:
    def __init__(self):
        self.mean_y = np.array([np.nan, np.nan], dtype=np.float32)

    def fit(self, X=None, y=None, *args, **kwargs):
        self.mean_y[:] = np.mean(y, axis=0)
        history = dict()
        return history

    def predict(self, X, *args, **kwargs):
        y_pred = np.tile(self.mean_y[np.newaxis, :], (X.shape[0], 1))
        return y_pred


class CNNModel:
    def __init__(self, x_shape, kind='pos', first_conv_size=3, other_conv_size=3,
                 padding='same', pool_padding='valid', first_conv_nfilt=2, other_conv_nfilt=2,
                 w_l2=0.001, w_l2_conv=0., n_convs=2, n_dense=4, pos_loss='mae', pres_loss='binary_crossentropy',
                 pos_out_bias=None, pres_out_bias=0., p_drop=0., **kwargs):
        from keras.models import Sequential
        from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
        from keras.regularizers import l2

        if len(kwargs) > 0:
            print('Unused kwargs:', kwargs)

        self.kind = kind

        input_shape = x_shape[1:] + (1,) if len(x_shape) == 3 else x_shape[1:]

        model = Sequential()

        model.add(Conv2D(first_conv_nfilt, (first_conv_size, first_conv_size),
                         activation='relu', input_shape=input_shape,
                         padding=padding, kernel_regularizer=l2(w_l2_conv)))
        model.add(MaxPooling2D(pool_size=(2, 2), padding=pool_padding))
        for i in range(n_convs - 1):
            model.add(Conv2D(other_conv_nfilt, (other_conv_size, other_conv_size), activation='relu',
                             padding=padding, kernel_regularizer=l2(w_l2_conv)))
            model.add(MaxPooling2D(pool_size=(2, 2), padding=pool_padding))
        model.add(Flatten())
        model.add(Dense(n_dense, activation='relu', kernel_regularizer=l2(w_l2)))

        if kind == 'pos':
            model.add(Dense(2, activation=None,
                            bias_initializer=Constant(pos_out_bias) if pos_out_bias is not None else None))
            model.compile(optimizer='adam', loss=pos_loss)
        elif kind == 'pres':
            if p_drop > 0.:
                model.add(Dropout(p_drop))
            model.add(Dense(1, activation='sigmoid',
                            bias_initializer=Constant(pres_out_bias) if pres_out_bias is not None else None))
            model.compile(optimizer='adam', loss=pres_loss, metrics=['accuracy'])
        else:
            raise NotImplementedError(kind)

        self.model = model

    @staticmethod
    def dist_to_weight(d, pw=2.):
        d[~np.isfinite(d)] = 1.
        assert np.all(d >= 0) and np.all(d <= 1)
        return ((1. - d) ** pw).flatten()

    def fit(self, X, y, X_dev, y_dev, d=None, d_dev=None, use_early_stopping=True,
            batch_size=_BATCH_SIZE, epochs=_EPOCHS, start_from_epoch=_MIN_EPOCHS, patience=_PATIENCE, **kwargs):
        if use_early_stopping:
            early_stopping = EarlyStoppingWarmup(
                monitor='val_loss', patience=patience, min_delta=0.001, restore_best_weights=True,
                start_from_epoch=start_from_epoch)
            callbacks = [TqdmCallback(verbose=1), early_stopping]
        else:
            callbacks = [TqdmCallback(verbose=1)]

        val_kws = dict()
        if X_dev is not None:
            if d_dev is not None:
                assert d is not None
                sample_weight_dev = self.dist_to_weight(d_dev)
                val_kws['validation_data'] = (X_dev, y_dev, sample_weight_dev)
            else:
                val_kws['validation_data'] = (X_dev, y_dev)
        else:
            val_kws['validation_split'] = 0.2

        if self.kind == 'pres':
            print('Initial accuracy:', accuracy_score(y_true=y, y_pred=self.predict(X) > 0.5))
            pass

        rnd_idxs_train = np.arange(len(y))
        np.random.shuffle(rnd_idxs_train)

        sample_weight = self.dist_to_weight(d)[rnd_idxs_train] if d is not None else None

        history = self.model.fit(
            x=X[rnd_idxs_train], y=y[rnd_idxs_train], sample_weight=sample_weight,
            batch_size=batch_size, epochs=epochs, callbacks=callbacks, verbose=0, **val_kws)
        return history.history

    def predict(self, X, *args, **kwargs):
        return self.model.predict(X, verbose=0).squeeze()


def get_pres_decoder_and_params(kind, params=None, p_train=None, X_shape=None):
    if params is None:
        params = dict()

    if kind == 'baseline':
        pres_decoder = DummyPresenceDecoder()
    elif kind == 'linear':
        pres_decoder = LogisticRegression()
    elif kind == 'ridge':
        pres_decoder = RidgeClassifier()
    elif kind == 'tiniest_mlp':
        default_params = dict(solver='adam', hidden_layer_sizes=(4, 4, 4, 4), early_stopping=True, max_iter=1000)
        default_params.update(params)
        pres_decoder = MLPClassifier(**default_params)
    elif kind == 'tiny_mlp':
        default_params = dict(solver='adam', hidden_layer_sizes=(32, 16,), early_stopping=True, max_iter=1000)
        default_params.update(params)
        pres_decoder = MLPClassifier(**default_params)
    elif kind == 'small_mlp':
        default_params = dict(solver='adam', hidden_layer_sizes=(64, 64, 32, 32), early_stopping=True,
                              max_iter=1000)
        default_params.update(params)
        pres_decoder = MLPClassifier(**default_params)
    elif kind == 'medium_mlp':
        default_params = dict(solver='adam', hidden_layer_sizes=(256, 256, 128, 128), early_stopping=True,
                              max_iter=1000)
        default_params.update(params)
        pres_decoder = MLPClassifier(**default_params)
    elif kind == 'large_mlp':
        default_params = dict(solver='adam', hidden_layer_sizes=(512, 512, 256, 256), early_stopping=True,
                              max_iter=1000)
        default_params.update(params)
        pres_decoder = MLPClassifier(**default_params)
    elif kind == 'cnn':
        default_params = dict(w_l2=0.001, n_convs=2)
        default_params.update(params)
        out_bias = float(np.log(np.sum(p_train) / np.sum(~p_train)))
        pres_decoder = CNNModel(x_shape=X_shape, kind='pres', pres_out_bias=out_bias, **default_params)
    else:
        raise NotImplementedError(kind)
    return pres_decoder


def get_pos_decoder_and_params(kind, params=None, X_shape=None):
    if params is None:
        params = dict()

    if kind == 'baseline':
        pos_decoder = DummyFrameDecoder()
    elif kind == 'linear':
        pos_decoder = LinearRegression()
    elif kind == 'ridge':
        default_params = dict(alphas=10 ** np.arange(-10, 10, 0.5))
        pos_decoder = RidgeCV(**default_params)
    elif kind == 'tiniest_mlp':
        default_params = dict(solver='adam', hidden_layer_sizes=(4, 4, 4, 4), early_stopping=True, max_iter=1000)
        default_params.update(params)
        pos_decoder = MLPRegressor(**default_params)
    elif kind == 'tiny_mlp':
        default_params = dict(solver='adam', hidden_layer_sizes=(32, 16,), early_stopping=True, max_iter=1000)
        default_params.update(params)
        pos_decoder = MLPRegressor(**default_params)
    elif kind == 'small_mlp':
        default_params = dict(solver='adam', hidden_layer_sizes=(64, 64, 32, 32), early_stopping=True,
                              max_iter=1000)
        default_params.update(params)
        pos_decoder = MLPRegressor(**default_params)
    elif kind == 'medium_mlp':
        default_params = dict(solver='adam', hidden_layer_sizes=(256, 256, 128, 128), early_stopping=True,
                              max_iter=1000)
        default_params.update(params)
        pos_decoder = MLPRegressor(**default_params)
    elif kind == 'large_mlp':
        default_params = dict(solver='adam', hidden_layer_sizes=(512, 512, 256, 256), early_stopping=True,
                              max_iter=1000)
        default_params.update(params)
        pos_decoder = MLPRegressor(**default_params)
    elif kind == 'cnn':
        default_params = dict(w_l2=0.001, n_convs=3)
        default_params.update(params)
        out_bias = 0.
        pos_decoder = CNNModel(x_shape=X_shape, kind='pos', pos_out_bias=out_bias, **default_params)
    else:
        raise NotImplementedError(kind)

    return pos_decoder
