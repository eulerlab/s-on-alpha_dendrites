import numpy as np
import shapely.geometry
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from tqdm.notebook import tqdm

from . import grid
from ..response.distributions import loss_normal, loss_truncated_normal, normal, truncated_normal


def compute_linesegement_in_polygon(line, plg, auto_convert=False):
    if auto_convert:
        if not isinstance(line, shapely.geometry.LineString):
            line = shapely.geometry.LineString(line)
        if not isinstance(plg, shapely.geometry.polygon.Polygon):
            plg = shapely.geometry.polygon.Polygon(plg)
    return line.intersection(plg).length


def center_grid(cxys, hxys):
    center_bc_idx = np.argmin(np.sum(cxys ** 2, axis=1) ** 0.5)

    center_offset = hxys[center_bc_idx][:, 0].copy()

    cxys -= center_offset
    hxys = (hxys.swapaxes(1, 2) - center_offset).swapaxes(1, 2)

    return cxys, hxys


class Morphology:
    def __init__(self, df_paths, soma_xyz, cxys, hxys, n_jobs=40):
        self.df_paths = df_paths
        self.soma_xyz = soma_xyz
        self.n_jobs = n_jobs
        self.cxys, self.hxys = center_grid(cxys.copy(), hxys.copy())

        # Lines
        self.lines = None

        self.lines_xs = None
        self.lines_ys = None

        self.lines_xlim = None
        self.lines_ylim = None

        self.line_lengths = None

        # Sympy
        self.sp_hexs = None
        self.sp_lines = None

        # Results
        self.dist_to_soma = np.sum(self.cxys ** 2, axis=1) ** 0.5
        self.length_per_hex = None

        self.opt_fit = None
        self.fit_kind = None

        # Init
        self.init_lines()
        self.init_line_summary()
        self.init_shapely_object()

    def init_lines(self):
        lines = []

        for path in self.df_paths.path:
            for p1, p2 in zip(path[:-1], path[1:]):
                line = (p1[:2] - self.soma_xyz[:2], p2[:2] - self.soma_xyz[:2])
                lines.append(line)

        self.lines = np.stack(lines)

    def init_line_summary(self):
        self.lines_xs = np.stack(self.lines)[:, :, 0]
        self.lines_ys = np.stack(self.lines)[:, :, 1]

        self.lines_xlim = (np.min(self.lines_xs, axis=1), np.max(self.lines_xs, axis=1))
        self.lines_ylim = (np.min(self.lines_ys, axis=1), np.max(self.lines_ys, axis=1))

        self.line_lengths = np.sum((self.lines[:, 0, :] - self.lines[:, 1, :]) ** 2, axis=1) ** 0.5

    def init_shapely_object(self):
        self.sp_lines = Parallel(n_jobs=self.n_jobs)(delayed(shapely.geometry.LineString)(line)
                                                     for line in self.lines)
        self.sp_hexs = Parallel(n_jobs=self.n_jobs)(delayed(shapely.geometry.polygon.Polygon)(bc_hxy.T)
                                                    for bc_hxy in self.hxys)

    def compute_length_per_area(self):
        self.length_per_hex = np.zeros(self.hxys.shape[0], dtype=float)

        for i, (bc_hxy, sp_hex) in enumerate(tqdm(zip(self.hxys, self.sp_hexs), total=len(self.sp_hexs))):
            hex_xlim = (np.min(bc_hxy[0, :]), np.max(bc_hxy[0, :]))
            hex_xlim_inner = (bc_hxy[0, 0], bc_hxy[0, 1])
            hex_ylim = (np.min(bc_hxy[1, :]), np.max(bc_hxy[1, :]))

            is_outside = (self.lines_xlim[0] > hex_xlim[1]) | \
                         (self.lines_xlim[1] < hex_xlim[0]) | \
                         (self.lines_ylim[0] > hex_ylim[1]) | \
                         (self.lines_ylim[1] < hex_ylim[0])

            is_inside = (self.lines_xlim[0] > hex_xlim_inner[0]) & \
                        (self.lines_xlim[1] < hex_xlim_inner[1]) & \
                        (self.lines_ylim[0] > hex_ylim[0]) & \
                        (self.lines_ylim[1] < hex_ylim[1])

            might_intersect = np.array(~(is_inside | is_outside))

            sp_lines_intersect = [sp_line for might_intersect_i, sp_line in zip(might_intersect, self.sp_lines) if
                                  might_intersect_i]

            lengths = Parallel(n_jobs=self.n_jobs)(delayed(compute_linesegement_in_polygon)(sp_line, sp_hex)
                                                   for sp_line in sp_lines_intersect)
            self.length_per_hex[i] = np.sum(lengths) + np.sum(self.line_lengths[is_inside])

    def _fit(self, loss, theta0_list):
        from scipy.optimize import minimize

        best_opt_fit = None

        for theta0 in theta0_list:
            opt_fit_i = minimize(loss, x0=theta0, args=(self.dist_to_soma, self.length_per_hex), method='Nelder-Mead')

            if best_opt_fit is None:
                best_opt_fit = opt_fit_i

            if float(opt_fit_i.fun) < float(best_opt_fit.fun):
                best_opt_fit = opt_fit_i

        return best_opt_fit

    def fit_normal(self):

        np.random.seed(42)

        theta0 = np.array([
            np.max(self.length_per_hex),
            self.dist_to_soma[self.length_per_hex < 0.5 * np.max(self.length_per_hex)].min()
        ])

        theta0_list = [np.array([theta0[0] * np.random.uniform(0.9, 1.1),
                                 theta0[1] * np.random.uniform(0.9, 1.1)])
                       for _ in range(10)]

        self.opt_fit = self._fit(loss=loss_normal, theta0_list=theta0_list)
        self.fit_kind = 'normal'

    def fit_truncated_normal(self):

        np.random.seed(42)

        theta0 = np.array([
            np.max(self.length_per_hex),
            self.dist_to_soma[self.length_per_hex < 0.5 * np.max(self.length_per_hex)].min(),
            0.25 * np.max(self.dist_to_soma),
        ])

        theta0_list = [np.array([theta0[0] * np.random.uniform(0.9, 1.1),
                                 theta0[1] * np.random.uniform(0.9, 1.1),
                                 theta0[2] * np.random.uniform(0.5, 3)])
                       for _ in range(10)]

        self.opt_fit = self._fit(loss=loss_truncated_normal, theta0_list=theta0_list)
        self.fit_kind = 'truncated_normal'

    def plot_lines(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))

        ax.plot(self.lines.T[0], self.lines.T[1])
        ax.set_aspect('equal', 'box')
        return ax

    def plot(self, ax=None):

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(7, 5))

        grid.plot_grid(self.cxys, self.hxys, ax=ax,
                       scatter_kws=dict(alpha=0, color='k', s=7),
                       hex_kws=dict(c='dimgray', zorder=-100))
        ax.plot(self.lines.T[0], self.lines.T[1], c='k', lw=0.8, zorder=100, alpha=1.0,
                solid_joinstyle='round', solid_capstyle='round')
        ax.set_aspect('equal', 'box')

        if self.length_per_hex is not None:
            plt.scatter(self.cxys[:, 0], self.cxys[:, 1], c=self.length_per_hex, cmap='Reds', zorder=50, s=35,
                        alpha=0.7, marker='H', edgecolor='none')
            plt.colorbar(label='dendritic length per hex [um]')

        return ax

    def get_yfit(self, x=None):

        if x is None:
            x = np.sort(self.dist_to_soma)

        if self.fit_kind == 'normal':
            yfit = normal(x, *self.opt_fit.x)
        elif self.fit_kind == 'truncated_normal':
            yfit = truncated_normal(x, *self.opt_fit.x)
        else:
            raise NotImplementedError(self.fit_kind)

        return yfit

    def plot_fit(self, ax=None):
        import seaborn as sns

        yfit = self.get_yfit()

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(6, 2))

        sns.scatterplot(x=self.dist_to_soma, y=self.length_per_hex, s=1, color='k', ax=ax)
        sns.regplot(x=self.dist_to_soma, y=self.length_per_hex, order=4, label='poly, order 4', ax=ax, scatter=False)
        ax.plot(np.sort(self.dist_to_soma), yfit, label=self.fit_kind)
        ax.set_xlabel('Soma-Dist [um]')
        ax.set_ylabel('L/Hex [um]')
        ax.legend()
