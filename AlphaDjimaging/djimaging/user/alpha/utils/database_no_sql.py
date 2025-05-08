import os

import pandas as pd
from djimaging.user.alpha.utils.pandas_dj_wrapper import DataJointTable


class DatabaseInterface:
    def __init__(self, indicator):
        folder_suffix = f"sONa_{indicator}"

        self.database_export_root = os.path.abspath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         '..', '..', '..', '..', '..', '..', 'database-exports', 'Ran', folder_suffix))

        print(f'Loading {indicator} data from:', self.database_export_root)
        assert os.path.isdir(self.database_export_root), f"Database directory for {indicator} not found"

    def _get_tab(self, prefix):
        return DataJointTable(pd.read_hdf(os.path.join(self.database_export_root, f'{prefix}_tab.h5')))


    def get_experiment_tab(self):
        return self._get_tab('experiment')


    def get_location_tab(self):
        return self._get_tab('location')


    def get_field_tab(self):
        return self._get_tab('field')


    def get_roi_pos_tab(self):
        return self._get_tab('roi_pos')


    def get_sinespot_tab(self, quality_filter=True):
        return self._get_tab('sinespot_qfilt' if quality_filter else 'sinespot')


    def get_gchirp_tab(self, quality_filter=True):
        return self._get_tab('gchirp_qfilt' if quality_filter else 'gchirp')


    def get_lchirp_tab(self, quality_filter=True):
        return self._get_tab('lchirp_qfilt' if quality_filter else 'lchirp')


    def get_roi_tab(self):
        return self._get_tab('roi')


    def get_morph_tab(self, quality_filter=True, include_linestack=False):
        table_name = 'morph'
        if quality_filter:
            table_name += '_qfilt'
        tab = self._get_tab(table_name)
        if not include_linestack:
            tab.df = tab.df.drop(columns=['linestack', 'fromfile_flat', 'linestack_flat'])

        return tab


    def get_averages_tab(self, quality_filter=True):
        return self._get_tab('averages_qfilt' if quality_filter else 'averages')


    def get_clustering_tab(self):
        return self._get_tab('clustering')


    def get_clustering_params_tab(self):
        return self._get_tab('clustering_params')


    def get_clustering_features_tab(self):
        return self._get_tab('clustering_features')


    def get_roi_rf_tab(self):
        return self._get_tab('roi_rf')


    def get_field_rf_tab(self, quality_filter=True):
        return self._get_tab('field_rf_qfilt' if quality_filter else 'field_rf')


    def get_soma_rf_tab(self):
        return self._get_tab('soma_rf')


    def get_field_stack_pos_tab(self):
        return self._get_tab('field_stack_pos')


    def get_roi_rf_morph_tab(self):
        return self._get_tab('roi_rf_morph'), self._get_tab('roi_rf'), self._get_tab('morph')


    def get_soma_rf_morph_tab(self):
        return self._get_tab('soma_rf_morph'), self._get_tab('soma_rf'), self._get_tab('morph')


    def get_pres_tab(self):
        return self._get_tab('pres')


    def get_paths_tab(self):
        return self._get_tab('paths')


    def get_chirp_surround_tab(self):
        return self._get_tab('chirp_surround')

    def get_field_avg_offset(self):
        # TODO: merge with database function?
        field_rf_tab = self.get_field_rf_tab()

        df_field_rfs = (
            field_rf_tab.proj(field_rf_dx_um='rf_dx_um', field_rf_dy_um='rf_dy_um', field_rf_d_um='rf_d_um')
        ).fetch(format='frame').reset_index()

        df_field_rfs['field'] = df_field_rfs['field'].apply(lambda x: x.replace('FieldROI', ''))
        df_field_rfs = df_field_rfs.drop('roi_id', axis=1)
        df_field_rfs = df_field_rfs.set_index(['date', 'exp_num', 'field'])

        field_avg_dx = df_field_rfs.groupby(['date', 'exp_num']).field_rf_dx_um.median()
        field_avg_dy = df_field_rfs.groupby(['date', 'exp_num']).field_rf_dy_um.median()

        return df_field_rfs, field_avg_dx, field_avg_dy

    def get_tri_df(self):
        df = pd.DataFrame(pd.read_hdf(os.path.join(self.database_export_root, 'tri_tab.h5')))
        df['cell_id'] = df['cell_id'].astype('category')
        df['field_id'] = df['field_id'].astype('category')
        df['group'] = df['group'].astype('category')
        return df
