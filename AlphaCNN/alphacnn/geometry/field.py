import numpy as np


class Field:

    def __init__(self, field_xlim: tuple, field_ylim: tuple, pixelsize: float):
        # Input parameters
        self.field_xlim = field_xlim
        self.field_ylim = field_ylim
        self.pixelsize = pixelsize

        # Derived parameters
        self.stim_dims = (int(np.ceil((field_xlim[1] - field_xlim[0]) / pixelsize)),
                          int(np.ceil((field_ylim[1] - field_ylim[0]) / pixelsize)))

        self.stim_xlim = np.array([0, self.stim_dims[0] * pixelsize]) + self.field_xlim[0]
        self.stim_ylim = np.array([0, self.stim_dims[1] * pixelsize]) + self.field_ylim[0]

        if self.field_xlim[0] < 0 < self.field_xlim[1] and self.field_ylim[0] < 0 < self.field_ylim[1]:
            self.stim_xlim = self.stim_xlim
            self.stim_ylim = self.stim_ylim

        self.stim_extent = (self.stim_xlim[0], self.stim_xlim[1], self.stim_ylim[0], self.stim_ylim[1])

    def __eq__(self, other):
        if not isinstance(other, Field):
            return NotImplemented

        is_equal = np.allclose(other.field_xlim, self.field_xlim) & \
            np.allclose(other.field_ylim, self.field_ylim) & \
            np.allclose(other.pixelsize, self.pixelsize) & \
            np.allclose(other.stim_dims, self.stim_dims) & \
            np.allclose(other.stim_xlim, self.stim_xlim) & \
            np.allclose(other.stim_ylim, self.stim_ylim) & \
            np.allclose(other.stim_extent, self.stim_extent)

        return is_equal
