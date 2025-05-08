from abc import abstractmethod

import numpy as np
import datajoint as dj
from djimaging.utils.dj_utils import get_primary_key
from matplotlib import pyplot as plt

from djimaging.utils.scanm_utils import get_rel_roi_pos


class RoiOffsetTemplate(dj.Computed):
    database = ""

    @property
    def definition(self):
        definition = """
        # location of roi relative to field center
        -> self.presentation_table
        -> self.roi_table
        ---
        roi_dx_um: float
        roi_dy_um: float
        roi_d_um: float
        """
        return definition

    @property
    def key_source(self):
        try:
            return self.roi_table.proj() * self.presentation_table.proj()
        except (AttributeError, TypeError):
            pass

    @property
    @abstractmethod
    def presentation_table(self):
        pass

    @property
    @abstractmethod
    def roi_table(self):
        pass

    @property
    @abstractmethod
    def roi_mask_table(self):
        pass

    def make(self, key):
        roi_id = (self.roi_table & key).fetch1('roi_id')
        roi_mask = (self.roi_mask_table & key).fetch1('roi_mask')
        pixel_size_um, scan_type = (self.presentation_table & key).fetch1('pixel_size_um', 'scan_type')
        ang_deg = (self.presentation_table.ScanInfo() & key).fetch1('angle_deg')

        # Add Roi offset to field offset
        if scan_type == 'xy':
            dx_um, dy_um = get_rel_roi_pos(roi_id, roi_mask, pixel_size_um, ang_deg=ang_deg)
            d_um = np.sqrt(dx_um ** 2 + dy_um ** 2)
        else:
            raise NotImplementedError(scan_type)

        roi_key = key.copy()
        roi_key['roi_dx_um'] = dx_um
        roi_key['roi_dy_um'] = dy_um
        roi_key['roi_d_um'] = d_um
        self.insert1(roi_key)

    def plot(self, key=None):
        if key is None:
            key = get_primary_key(self.presentation_table)
        roi_dx_um, roi_dy_um = (self & key).fetch('roi_dx_um', 'roi_dy_um')
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.plot(roi_dx_um, roi_dy_um, 'o', label='roi', c='C0')
        ax.set_aspect(aspect="equal", adjustable="datalim")
        ax.set(xlabel='dx (um)', ylabel='dy (um)')
        plt.show()
