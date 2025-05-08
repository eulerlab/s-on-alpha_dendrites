from djimaging.user.alpha.schemas.alpha_somas_schema import *
from djimaging.user.alpha.utils.populate_alpha import SCHEMA_PREFIX, CONFIG_FILE


def load_alpha_config(schema_name):
    dj.config.load(CONFIG_FILE)
    dj.config['schema_name'] = schema_name
    dj.conn()

    print("schema_name:", dj.config['schema_name'])


def load_alpha_schema(create_schema=False, create_tables=False):
    from djimaging.utils.dj_utils import activate_schema
    activate_schema(schema=schema, create_schema=create_schema, create_tables=create_tables)


def connect_dj(create_tables=False, create_schema=False) -> None:
    schema_name = SCHEMA_PREFIX + 'soma'
    load_alpha_config(schema_name=schema_name)
    load_alpha_schema(create_schema=create_schema, create_tables=create_tables)


def get_soma_spots_tab(n_reps_min=2, q_thresh=0.5):
    spot_loc_q_tab = (
            (
                    (WbgSpots & dict(condition='control'))
                    * RetinalFieldLocation
                    * Presentation.ScanInfo().proj('scan_frequency')
                    * SizeRoi
                    * (Averages & dict(stim_name='gChirp', condition='control') & (ChirpQI & "qidx>0.35") & dict(
                condition='control')).proj('average', chirp='stim_name')
            )
            & f"w_qidx>{q_thresh}"
            & [f"g_qidx>{q_thresh}", f"b_qidx>{q_thresh}"] & f"n_reps>={n_reps_min}"
    )
    return spot_loc_q_tab


def get_avg_fs():
    return Averages._f_resample
