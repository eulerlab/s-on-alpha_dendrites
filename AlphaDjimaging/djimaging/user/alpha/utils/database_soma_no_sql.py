import os

import pandas as pd
from djimaging.user.alpha.utils.pandas_dj_wrapper import DataJointTable


class DatabaseInterface:
    def __init__(self, indicator):
        assert indicator == 'soma'

        self.database_export_root = os.path.abspath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         '..', '..', '..', '..', '..', '..', 'database-exports', 'Oesterle', 'sONa_somas'))

        print(f'Loading {indicator} data from:', self.database_export_root)
        assert os.path.isdir(self.database_export_root), f"Database directory for {indicator} not found"

    def _get_tab(self, prefix):
        return DataJointTable(pd.read_hdf(os.path.join(self.database_export_root, f'{prefix}_tab.h5')))


    def get_soma_spots_tab(self):
        return self._get_tab('soma_spots')

    @staticmethod
    def get_avg_fs():
        return 500