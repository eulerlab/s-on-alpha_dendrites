from scipy.spatial import ConvexHull
import numpy as np
import datajoint as dj
import pandas as pd
from abc import abstractmethod
from matplotlib import pyplot as plt

from djimaging.utils.dj_utils import get_primary_key


def compute_convex_hull_features(points):
    hull = ConvexHull(points)
    hull_polygon = np.vstack([hull.points[hull.vertices], hull.points[hull.vertices[0]]])
    hull_area = hull.volume
    hull_cdiameter = 2 * np.sqrt(hull_area / np.pi)
    return hull_polygon, hull_area, hull_cdiameter


class ConvexHullTemplate(dj.Computed):
    database = ""

    @property
    def definition(self):
        definition = """
            -> self.morph_table
            ---
            hull_points : mediumblob
            hull_area_um2 : float
            hull_cdia_um : float
            """
        return definition

    @property
    @abstractmethod
    def morph_table(self):
        pass

    @property
    def key_source(self):
        try:
            return self.morph_table.proj()
        except (AttributeError, TypeError):
            pass

    def make(self, key):
        df_paths = pd.DataFrame((self.morph_table & key).fetch1('df_paths'))
        polygon, area, cdia = compute_convex_hull_features(points=np.vstack(df_paths.path)[:, :2])
        self.insert1(dict(key, hull_points=polygon, hull_area_um2=area, hull_cdia_um=cdia))

    def plot1(self, key=None):
        key = get_primary_key(table=self, key=key)
        df_paths = pd.DataFrame((self.morph_table & key).fetch1('df_paths'))
        polygon, area, cdia = (self & key).fetch1('hull_points', 'hull_area_um2', 'hull_cdia_um')

        fig, ax = plt.subplots(1, 1, figsize=(6, 6))

        ax.set_title(f"cdia={cdia:.1f}")
        ax.plot(*polygon.T, c='r')

        for path in df_paths.path:
            ax.plot(path[:, 0], path[:, 1], color='black', lw=1)

        return fig, ax
