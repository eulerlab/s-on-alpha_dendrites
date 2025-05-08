import numpy as np
from joblib import dump
from sklearn.preprocessing import StandardScaler

from alphacnn.decoder.frame_decoder import CNNModel, get_pres_decoder_and_params

_BATCH_SIZE = 16384
_EPOCHS = 1000
_MIN_EPOCHS = 200
_PATIENCE = 10


# #  Debug values
# EPOCHS = 5
# MIN_EPOCHS = 2
# PATIENCE = 1


def train_pres_decoder(n_ensemble, kind, params, X_train, p_train, X_dev=None, p_dev=None, norm_data=False, seed=42):
    if n_ensemble > 1:
        print(f'Get frame decoder ensemble: n={n_ensemble}')
        decoder = PresDecoderEnsemble(
            n_ensemble=n_ensemble, kind=kind, X_train=X_train, p_train=p_train, X_dev=X_dev, p_dev=p_dev,
            params=params, norm_data=norm_data, core_seed=seed)
        history = [d.history_pres for d in decoder.decoders]
    else:
        print('Get frame decoder')
        decoder = PresDecoder(
            kind=kind, X_train=X_train, p_train=p_train,
            X_dev=X_dev, p_dev=p_dev, params=params, norm_data=norm_data, seed=seed)
        history = [decoder.history_pres]

    return decoder, history


class PresDecoder:
    def __init__(self, kind, X_train, p_train, X_dev=None, p_dev=None, params=None, norm_data=True, seed=42):
        self.kind = kind

        self.X_train = X_train
        self.p_train = p_train

        self.X_dev = X_dev
        self.p_dev = p_dev

        self.x_scaler = StandardScaler() if norm_data else None

        self.decoder = get_pres_decoder_and_params(
            kind=kind, params=params, p_train=self.p_train, X_shape=self.X_train.shape)

        np.random.seed(seed)
        if kind == 'cnn':
            import tensorflow as tf
            tf.random.set_seed(int(seed))

        self.history_pres = self.train()

    def train(self):
        print('Training PRES decoder')
        history_pres = self._train_pres(X_train=self.X_train, p_train=self.p_train, X_dev=self.X_dev, p_dev=self.p_dev)
        return history_pres

    def _train_pres(self, X_train, p_train, X_dev=None, p_dev=None):
        assert np.all(p_train == p_train.astype(bool))
        X_scaled = self.x_scaler.fit_transform(X_train) if self.x_scaler is not None else X_train

        if X_dev is None:
            history = self.decoder.fit(X=X_scaled, y=p_train.astype(bool))
        else:
            X_dev_scaled = self.x_scaler.transform(X_dev) if self.x_scaler is not None else X_dev
            if isinstance(self.decoder, CNNModel):
                history = self.decoder.fit(
                    X=X_scaled, y=p_train.astype(bool), X_dev=X_dev_scaled, y_dev=p_dev.astype(bool))
            else:
                history = self.decoder.fit(
                    X=np.concatenate([X_scaled, X_dev_scaled], axis=0),
                    y=np.concatenate([p_train.astype(bool), p_dev.astype(bool)], axis=0))

        return history

    def eval(self, X_test, p_test=None):
        assert X_test.ndim == 2 or X_test.ndim == 3, X_test.ndim
        if p_test is not None:
            assert X_test.shape[0] == p_test.shape[0]

        X_scaled = self.x_scaler.transform(X_test) if self.x_scaler is not None else X_test
        p_pred = self.decoder.predict(X_scaled)

        if p_test is not None:
            p_errors = np.abs(p_test.astype(int) - (p_pred > 0.5).astype(int))
            return p_pred, p_errors
        else:
            return p_pred

    def eval_and_summarize(self, X_test, p_test):
        p_pred_train, p_errors_train = self.eval(X_test=self.X_train, p_test=self.p_train)
        p_pred_test, p_errors_test = self.eval(X_test=X_test, p_test=p_test)

        decoder_summary = dict()
        decoder_summary['X_train'] = self.X_train
        decoder_summary['X_test'] = X_test

        decoder_summary['p_train'] = self.p_train
        decoder_summary['p_test'] = p_test

        decoder_summary['p_pred_train'] = p_pred_train
        decoder_summary['p_errors_train'] = p_errors_train

        decoder_summary['p_pred_test'] = p_pred_test
        decoder_summary['p_errors_test'] = p_errors_test

        return decoder_summary

    def save_to_file(self, model_path):
        if self.kind == 'cnn':
            self.decoder.model.save(filepath=model_path + '_pres')
        else:
            dump(self.decoder, model_path + '_pres.joblib')

    def load_from_file(self, model_path):
        raise NotImplementedError()


class PresDecoderEnsemble:
    def __init__(self, kind, X_train, p_train, X_dev=None, p_dev=None,
                 params=None, norm_data=True, n_ensemble=3, core_seed=42):
        np.random.seed(core_seed)
        self.seeds = np.random.randint(0, np.iinfo(np.int32).max, size=n_ensemble, dtype=np.int32)
        self.decoders = []
        for seed in self.seeds:
            self.decoders.append(
                PresDecoder(kind=kind, X_train=X_train, p_train=p_train, X_dev=X_dev, p_dev=p_dev,
                            params=params, norm_data=norm_data, seed=seed))

    def eval(self, X_test, p_test=None):
        p_preds = []

        for decoder in self.decoders:
            p_pred_i = decoder.eval(X_test=X_test)
            p_preds.append(p_pred_i[np.newaxis, :])

        p_pred = np.mean(np.concatenate(p_preds, axis=0), axis=0)

        if p_test is not None:
            p_errors = np.abs(p_test.astype(int) - (p_pred > 0.5).astype(int))
            return p_pred, p_errors
        else:
            return p_pred

    def save_to_file(self, model_path):
        for i, decoder in enumerate(self.decoders):
            decoder.save_to_file(model_path=model_path + f'_{i}')

    def load_from_file(self, filepath):
        raise NotImplementedError()
