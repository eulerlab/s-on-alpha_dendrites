import os
import warnings
from abc import abstractmethod

import datajoint as dj
import numpy as np
import pandas as pd
from djimaging.user.alpha.tables.morphology.morphology_utils import compute_df_paths_and_density_maps, \
    compute_density_map_extent, compute_density_center, get_linestack
from djimaging.utils.data_utils import load_h5_data
from djimaging.utils.datafile_utils import find_folders_with_file_of_type


class SWCTemplate(dj.Computed):
    database = ""
    _swc_folder = 'SWC'

    @property
    def definition(self):
        definition = """
        # SWC for experiment assumes single cell per exp_num
        -> self.field_table
        ---
        swc_path : varchar(191)
        """
        return definition

    @property
    @abstractmethod
    def field_table(self):
        pass

    @property
    @abstractmethod
    def experiment_table(self):
        pass

    @property
    def key_source(self):
        try:
            return (self.field_table & "z_stack_flag=1").proj()
        except (AttributeError, TypeError):
            pass

    def make(self, key):
        header_path = (self.experiment_table() & key).fetch1('header_path')

        swc_folder = find_folders_with_file_of_type(header_path, 'swc', ignore_hidden=True)

        if len(swc_folder) > 1:
            swc_folder = [swc for swc in swc_folder if swc.endswith(self._swc_folder)]

        if len(swc_folder) == 0:
            warnings.warn(f'Did not find swc file for key={key}')
            return
        swc_folder = swc_folder[0]

        swc_file = [f for f in os.listdir(swc_folder) if f.endswith('swc')]
        assert len(swc_file) == 1, f'Found multiple SWC files for key={key}'
        swc_file = swc_file[0]

        swc_key = key.copy()
        swc_key['swc_path'] = os.path.join(swc_folder, swc_file)
        self.insert1(swc_key)


class MorphPathsTemplate(dj.Computed):
    database = ""

    @property
    def definition(self):
        definition = """
        # SWC for experiment assumes single cell per exp_num
        -> self.swc_table
        ---
        soma_xyz: blob
        df_paths : longblob
        density_map : longblob
        density_map_extent : blob
        density_center: blob        
        """
        return definition

    @property
    def key_source(self):
        try:
            return self.swc_table.proj()
        except (AttributeError, TypeError):
            pass

    @property
    @abstractmethod
    def field_table(self):
        pass

    @property
    @abstractmethod
    def swc_table(self):
        pass

    def make(self, key):
        swc_path = (self.swc_table() & key).fetch1('swc_path')
        pixel_size_um = (self.field_table() & key).fetch1('pixel_size_um')

        df_paths, density_maps = compute_df_paths_and_density_maps(swc_path=swc_path, pixel_size_um=pixel_size_um)

        # Update cell parameters
        soma_xyz = df_paths[df_paths.type == 1].path[0].flatten()
        density_map = density_maps[1]
        density_map_extent = compute_density_map_extent(paths=df_paths.path.iloc[1:], soma=soma_xyz)
        density_center = compute_density_center(df_paths.path.iloc[1:], soma_xyz, density_map)

        paths_key = key.copy()
        paths_key['soma_xyz'] = np.asarray(soma_xyz).astype(np.float32)
        paths_key['df_paths'] = df_paths.to_dict()
        paths_key['density_map'] = np.asarray(density_map).astype(np.float32)
        paths_key['density_map_extent'] = np.asarray(density_map_extent).astype(np.float32)
        paths_key['density_center'] = np.asarray(density_center).astype(np.float32)

        self.insert1(paths_key)


class LineStackTemplate(dj.Computed):
    database = ""

    @property
    def definition(self):
        definition = """
        # Linestack for experiment assumes single cell per exp_num
        -> self.morph_table
        ---
        linestack: longblob
        """
        return definition

    @property
    def key_source(self):
        try:
            return self.morph_table.proj()
        except (AttributeError, TypeError):
            pass

    @property
    @abstractmethod
    def morph_table(self):
        pass

    @property
    @abstractmethod
    def field_table(self):
        pass

    def make(self, key):
        zstack_file = (self.field_table() & key).fetch1('fromfile')
        df_paths = pd.DataFrame((self.morph_table() & key).fetch1('df_paths'))

        data_stack = load_h5_data(zstack_file, lower_keys=True)
        stack_shape = data_stack.get('line_stack_warped', data_stack['wdatach0']).shape

        linestack = get_linestack(df_paths, stack_shape)

        paths_key = key.copy()
        paths_key['linestack'] = np.asarray(linestack)

        self.insert1(paths_key)
