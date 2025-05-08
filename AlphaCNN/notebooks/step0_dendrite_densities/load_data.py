import argparse

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("indicator")
parser.add_argument("data_folder")
args = parser.parse_args()


def run(indicator, data_folder):
    from alphacnn.utils import data_utils
    from djimaging.user.alpha.utils.database import connect_dj
    from djimaging.user.alpha.schemas.alpha_schema import MorphPaths, RetinalFieldLocationCat

    assert len(indicator) > 0, 'Enter valid username'
    assert len(data_folder) > 0, 'Enter valid username'

    connect_dj(indicator, create_tables=False, create_schema=False)

    df = pd.DataFrame(MorphPaths() * RetinalFieldLocationCat())
    df.drop(['table_hash', 'field', 'density_map', 'density_map_extent', 'density_center', 'nt_side', 'vd_side'],
            axis=1, inplace=True)

    df['indicator'] = indicator
    df['n_tvd_side'] = df.ntvd_side.apply(lambda x: 'n' if x in ['nd', 'nv'] else x)
    df['df_paths'] = df.df_paths.apply(pd.DataFrame)

    data_utils.save_var(df, f'{data_folder}/df_{indicator}.pkl')


if __name__ == "__main__":
    run(indicator=args.indicator, data_folder=args.data_folder)
