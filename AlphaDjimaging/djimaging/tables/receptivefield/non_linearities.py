import warnings
from abc import abstractmethod
from copy import deepcopy

import datajoint as dj
import numpy as np
from djimaging.utils.dj_utils import get_primary_key
from matplotlib import pyplot as plt
from scipy.optimize import minimize

from djimaging.tables.receptivefield import rf_utils


class FitSigmoidNonLinearityTemplate(dj.Computed):
    database = ""

    @property
    def definition(self):
        definition = """
        -> self.fit_rf_table
        ---
        sigmoid_params : blob # Sigmoid parameters
        """
        return definition

    class DataSet(dj.Part):
        @property
        def definition(self):
            definition = """
            -> master
            kind : enum('train', 'dev', 'test')  # Data set kind
            ---
            y_pred : longblob # predicted output
            cc : float  # Correlation
            mse : float  # Mean Squared Error
            """
            return definition

    @property
    def key_source(self):
        try:
            return self.fit_rf_table.proj()
        except (AttributeError, TypeError):
            pass

    @property
    @abstractmethod
    def rf_table(self):
        pass

    @property
    @abstractmethod
    def split_rf_table(self):
        pass

    @property
    @abstractmethod
    def fit_rf_table(self):
        pass

    def fetch1_rf_time(self, key):
        try:
            rf_time = (self.rf_table & key).fetch1('rf_time')
        except dj.DataJointError:
            model_dict = (self.rf_table & key).fetch1('model_dict')

            if "rf_time" in model_dict:
                rf_time = model_dict['rf_time']
            else:
                dt, rf = (self.rf_table & key).fetch1('dt', 'rf')
                n_t = rf.shape[0]
                shift = model_dict['shift']
                rf_time = -(np.arange(n_t) * dt + shift['stimulus'] * dt)[::-1]

        return rf_time

    def fetch1_shift(self, key):
        try:
            shift = (self.rf_table & key).fetch1('shift')
        except dj.DataJointError:
            shift = (self.rf_table & key).fetch1('model_dict')['shift']['stimulus']
        return shift

    def make(self, key):
        shift = self.fetch1_shift(key)
        rf_time = self.fetch1_rf_time(key)
        trf = (self.split_rf_table() & key).fetch1("trf")
        srf_fit = (self.fit_rf_table & key).fetch1('srf_fit')
        x_train, y_train, burn_in = (self.rf_table.DataSet() & key & "kind='train'").fetch1('x', 'y', 'burn_in')

        trf_fit = trf.copy()
        trf_fit[rf_time > 0] = 0.
        strf_fit = rf_utils.normalize_strf(rf_utils.merge_strf(srf=srf_fit, trf=trf_fit))

        x_train_dm = rf_utils.build_design_matrix(X=x_train, n_lag=strf_fit.shape[0], shift=shift)[burn_in:]
        y_train_pred_lin = x_train_dm @ strf_fit.flatten()
        sigmoid_params = fit_sigmoid(y_train[burn_in:], y_train_pred_lin, alpha=0.5)

        y_train_pred = apply_sigmoid(y_train_pred_lin, *sigmoid_params)
        cc_train = np.corrcoef(y_train[burn_in:], y_train_pred)[0, 1]
        mse_train = np.mean((y_train[burn_in:] - y_train_pred) ** 2) ** 0.5

        # Save
        fit_key = deepcopy(key)
        self.insert1(dict(**fit_key, sigmoid_params=sigmoid_params))
        self.DataSet().insert1(dict(**fit_key, kind='train', y_pred=y_train_pred, cc=cc_train, mse=mse_train))

        for kind in set((self.rf_table.DataSet() & key).fetch('kind')) - {'train'}:
            x_i, y_i, burn_in_i = (self.rf_table.DataSet() & key & f"kind='{kind}'").fetch1('x', 'y', 'burn_in')
            x_i_dm = rf_utils.build_design_matrix(X=x_i, n_lag=strf_fit.shape[0], shift=shift)[burn_in_i:]
            y_i_pred_lin = x_i_dm @ strf_fit.flatten()
            y_i_pred = apply_sigmoid(y_i_pred_lin, *sigmoid_params)
            cc_i = np.corrcoef(y_train[burn_in_i:], y_i_pred)[0, 1]
            mse_i = np.mean((y_train[burn_in_i:] - y_i_pred) ** 2) ** 0.5
            self.DataSet().insert1(dict(**fit_key, kind=kind, y_pred=y_i_pred, cc=cc_i, mse=mse_i))

    def plot1(self, key=None):
        key = get_primary_key(table=self, key=key)

        sigmoid_params = (self & key).fetch1('sigmoid_params')
        dataset_kinds = (self.rf_table.DataSet() & key).fetch('kind')
        dt = (self.rf_table & key).fetch('dt')

        fig, axs = plt.subplots(1, 1 + len(dataset_kinds), figsize=(5 * (1 + len(dataset_kinds)), 3))
        ax = axs[0]
        x_plot = np.linspace(-2, 2, 101)
        y_plot = apply_sigmoid(x_plot, *sigmoid_params)
        ax.plot(x_plot, y_plot)
        ax.set(xlabel='input', ylabel='output')

        for ax, kind in zip(axs[1:], dataset_kinds):
            y, burn_in = (self.rf_table.DataSet() & key & f"kind='{kind}'").fetch1('y', 'burn_in')
            y_pred = (self.DataSet() & key & f"kind='{kind}'").fetch1('y_pred')
            time = np.arange(y_pred.size) * dt
            ax.plot(time, y[burn_in:], label='data')
            ax.plot(time, y_pred, ls='--', label='pred')
            ax.set(title=kind, xlabel='Time', ylabel='y')
            ax.legend(loc='upper right')

        plt.show()

    def plot(self, restriction=None):
        if restriction is None:
            restriction = dict()

        tab = (self & restriction)
        sigmoid_params = tab.fetch('sigmoid_params')
        cc_train = (self.DataSet() & tab & "kind='train'").fetch('cc')
        mse_train = (self.DataSet() & tab & "kind='train'").fetch('mse')

        fig, axs = plt.subplots(1, 3, figsize=(9, 3))

        ax = axs[0]
        x_plot = np.linspace(-2, 2, 101)
        for sigmoid_params_i in sigmoid_params:
            y_plot = apply_sigmoid(x_plot, *sigmoid_params_i)
            ax.plot(x_plot, y_plot)
        ax.set(xlabel='input', ylabel='output')

        ax = axs[1]
        ax.hist(cc_train)
        ax.set(xlim=(-1, 1), xlabel='cc(train)')

        ax = axs[2]
        ax.hist(mse_train)
        ax.set(xlabel='mse(train)')

        plt.show()


def fit_sigmoid(y, y_pred_lin, alpha=0.5):
    k0 = np.percentile(y, q=90)
    q0 = 1.
    b0 = 1.
    v0 = 1.

    x0 = np.array([k0, q0, b0, v0])

    bounds = [(np.min(y), np.max(y)),
              (0.01, 10),
              (0.01, 10),
              (-3, 3)]

    warnings.simplefilter(action='ignore', category=RuntimeWarning)
    fit_result = minimize(fun=fit_sigmoid_min_fun, x0=x0, args=(y, y_pred_lin, alpha,),
                          method='Nelder-Mead', bounds=bounds)

    for i in range(9):
        x0_i = np.array([np.random.uniform(low, high) for low, high in bounds])

        fit_result_i = minimize(fun=fit_sigmoid_min_fun, x0=x0_i, args=(y, y_pred_lin, alpha,),
                                method='Nelder-Mead', bounds=bounds)
        if fit_result_i.fun < fit_result.fun:
            fit_result = fit_result_i

    sigmoid_params = fit_result.x
    return sigmoid_params


def apply_sigmoid(y, k=1., q=1., b=1., v=1., d=0.):
    return k / (1. + q * np.exp(-b * (y - d))) ** (1. / np.exp(v))


def loss_diff(y, y_pred, q=2.):
    return np.abs(np.mean((y - y_pred) ** q)) ** 1. / q


def loss_corr(y, y_pred):
    return 1. - np.corrcoef(y, y_pred)[0, 1]


def fit_sigmoid_loss_fun(y, y_pred, alpha=0.5):
    l_diff = loss_diff(y, y_pred, q=2.)
    l_corr = loss_corr(y, y_pred)
    return alpha * l_diff + (1. - alpha) * l_corr


def fit_sigmoid_min_fun(x, y, y_pred_lin, alpha):
    trace_pred = apply_sigmoid(y_pred_lin, *x)
    loss = fit_sigmoid_loss_fun(y=y, y_pred=trace_pred, alpha=alpha)
    return loss
