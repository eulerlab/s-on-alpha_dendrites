import os

from djimaging.user.alpha.schemas.alpha_schema import *
from djimaging.user.alpha.utils.populate_alpha import SCHEMA_PREFIX, CONFIG_FILE, get_dataset, __cell_position_table
from djimaging.utils.dj_utils import get_secondary_keys

__rf_kind = {'calcium': 'glm', 'glutamate': 'glm'}

__quality_id = 1

__glm_dnoise_params_id = 1
__glm_params_id = {'calcium': 10, 'glutamate': 10}
__q_rf_glm_split_min = 0.35
__lcontour_ratio_min = 0.8

__soma_roi_max_dist = 50

# Remove some weird fits
__rf_cdia_um_range = {'calcium': (25, 500), 'glutamate': (25, 200)}  # Likely just a random noise fit when outside
__rf_max_lag = {'calcium': 0.3, 'glutamate': 0.2}  # Likely wrong sign for tRF when larger

# Clustering
__features_id = {'calcium': 1, 'glutamate': 1}
__clustering_id = {'calcium': 1, 'glutamate': 1}

FLAGGED_NOISE_FILES = {
    'calcium': [],
    'glutamate': [],
}


def load_alpha_config(schema_name):
    dj.config.load(CONFIG_FILE)
    dj.config['schema_name'] = schema_name
    dj.conn()

    print("schema_name:", dj.config['schema_name'])
    print("dataset:", get_dataset())


def load_alpha_schema(create_schema=False, create_tables=False):
    from djimaging.utils.dj_utils import activate_schema
    from djimaging.tables.location.location_from_table import prepare_dj_config_location_from_table
    prepare_dj_config_location_from_table(input_folder=os.path.split(__cell_position_table)[0])
    activate_schema(schema=schema, create_schema=create_schema, create_tables=create_tables)


def connect_dj(indicator: str, create_tables=False, create_schema=False) -> None:
    if indicator == 'calcium':
        schema_name = SCHEMA_PREFIX + 'ca'
    elif indicator == 'glutamate':
        schema_name = SCHEMA_PREFIX + 'glu'
    else:
        raise NotImplementedError(f"Unknown indicator: {indicator}")

    load_alpha_config(schema_name=schema_name)
    load_alpha_schema(create_schema=create_schema, create_tables=create_tables)


def get_roi_kind_filter(roi_kind: str):
    if roi_kind == 'soma':
        # Exclude old way to define soma ROIs
        roi_kind_filter = ((RoiKind & "roi_kind='field'") &
                           (FieldRoiPosMetrics() & f"d_dist_to_soma<{__soma_roi_max_dist}"))
    elif roi_kind == 'field':
        roi_kind_filter = (RoiKind & "roi_kind='field'")
    elif roi_kind == 'true_soma':
        roi_kind_filter = (RoiKind & "roi_kind='soma'")
    else:
        roi_kind_filter = (RoiKind & "roi_kind='roi'")

    return roi_kind_filter


def get_roi_tab(quality_filter=True, roi_kind='roi') -> dj.Table:
    """Get ROI table with all ROIs that have minimum response quality"""
    if quality_filter and roi_kind == 'roi':
        q_filter = (
                (QualityIndex & f"q_tot=1" & f"quality_params_id={__quality_id}") &
                (FieldCalibratedStackPos().RoiCalibratedStackPos() & "success_cal_flag=1")
        )
    elif roi_kind == 'field':
        q_filter = dict()
    else:
        q_filter = dict()

    return Roi & q_filter & get_roi_kind_filter(roi_kind=roi_kind)


def get_experiment_tab(quality_filter=False, roi_kind='any') -> dj.Table:
    if roi_kind == 'any':
        if quality_filter:
            raise ValueError("Quality filter not implemented for any ROI kind")
        return Experiment() * CellTags()
    else:
        return (Experiment * CellTags) & get_roi_tab(quality_filter=quality_filter, roi_kind=roi_kind)


def get_field_tab() -> dj.Table:
    return Field()


def get_location_tab() -> dj.Table:
    quality_filter = True
    roi_kind = 'roi'
    location_tab = RetinalFieldLocationFromTable & get_roi_kind_filter(roi_kind=roi_kind)

    if get_dataset() == 'calcium':
        location_tab *= RetinalFieldLocationWing().proj(group="wing_side")
    elif get_dataset() == 'glutamate':
        location_tab *= RetinalFieldLocationCat().proj(group="nt_side")
    else:
        raise NotImplementedError(f"Unknown indicator: {get_dataset()}")

    return location_tab & get_experiment_tab(quality_filter=quality_filter, roi_kind=roi_kind)


def get_morph_tab(quality_filter=True, include_linestack=False) -> dj.Table:
    roi_kind = 'roi'

    if get_dataset() == 'calcium':
        loc_tab = RetinalFieldLocationWing().proj(group="wing_side")
    elif get_dataset() == 'glutamate':
        loc_tab = RetinalFieldLocationCat().proj(group="nt_side")
    else:
        raise NotImplementedError(f"Unknown indicator: {get_dataset()}")

    if include_linestack:
        morph_tab = MorphPaths * RetinalFieldLocationFromTable * SWC * ConvexHull * CellTags * loc_tab * LineStack
    else:
        morph_tab = MorphPaths * RetinalFieldLocationFromTable * SWC * ConvexHull * CellTags * loc_tab

    morph_tab & get_experiment_tab(quality_filter=quality_filter, roi_kind=roi_kind).proj()
    proj = {k: k for k in get_secondary_keys(morph_tab)}
    proj['field_stack'] = 'field'
    morph_tab = morph_tab.proj(**proj)

    return morph_tab


def get_averages_tab(quality_filter=True) -> dj.Table:
    avg_tab = Averages
    return avg_tab & get_roi_tab(quality_filter=quality_filter, roi_kind='roi')


def get_lchirp_tab(quality_filter=True) -> dj.Table:
    avg_tab = get_averages_tab(quality_filter=quality_filter)
    return avg_tab & "stim_name='lChirp'"


def get_gchirp_tab(quality_filter=True) -> dj.Table:
    avg_tab = get_averages_tab(quality_filter=quality_filter)
    return avg_tab & "stim_name='gChirp'"


def get_sinespot_tab(quality_filter=True) -> dj.Table:
    avg_tab = get_averages_tab(quality_filter=quality_filter)
    return (avg_tab & "stim_name='sinespot'") * SineSpotSurroundIndex() * SineSpotQI()


def get_clustering_tab() -> dj.Table:
    clust_tab = (Clustering.RoiCluster &
                 f"features_id={__features_id[get_dataset()]}" &
                 f"clustering_id={__clustering_id[get_dataset()]}")
    return clust_tab & get_roi_tab(quality_filter=True, roi_kind='roi')


def get_clustering_params_tab() -> dj.Table:
    return (FeaturesParams().proj(
        'ncomps', 'stim_names', 'norm_trace', feature_kind='kind', feature_params_dict='params_dict')
            * ClusteringParameters) & f"clustering_id={__clustering_id[get_dataset()]}"


def get_clustering_features_tab() -> dj.Table:
    return Features & f"features_id={__features_id[get_dataset()]}"


def get_roi_pos_tab() -> dj.Table:
    return (
            (FieldStackPos
             * FieldPosMetrics.RoiPosMetrics
             * (RelativeRoiPos & [dict(stim_name='noise_1500'), dict(stim_name='noise_2500')]).proj(
                        'roi_dx_um', 'roi_dy_um', 'roi_d_um', pos_stim_name='stim_name')
             * FieldPathPos
             * FieldCalibratedStackPos.RoiCalibratedStackPos
             ) & get_roi_tab(quality_filter=True, roi_kind='roi'))


def get_roi_rf_tab() -> dj.Table:
    return __get_rf_tab(roi_kind='roi', quality_filter=True, rf_quality_filter=True, reject_tags=('none',))


def get_field_rf_tab(quality_filter=True) -> dj.Table:
    return __get_rf_tab(roi_kind='field', quality_filter=quality_filter, rf_quality_filter=True)


def get_soma_rf_tab() -> dj.Table:
    return __get_rf_tab(roi_kind='soma', only_one_soma_rf=True, quality_filter=True, rf_quality_filter=True)


def __get_rf_tab(roi_kind='roi', quality_filter=True, rf_quality_filter=None, only_one_soma_rf=True,
                 reject_tags=('none',)) -> dj.Table:
    if rf_quality_filter is None:
        rf_quality_filter = quality_filter

    from djimaging.user.alpha.schemas.alpha_schema import RFGLM, SplitRFGLM, TempRFGLMProperties

    glm_dnoise_params_id = __glm_dnoise_params_id
    glm_params_id = __glm_params_id[get_dataset()]
    lcontour_ratio_min = __lcontour_ratio_min
    q_rf_split_min = __q_rf_glm_split_min
    rf_cdia_um_range = __rf_cdia_um_range[get_dataset()]
    max_main_peak_lag = __rf_max_lag[get_dataset()]

    rf_tab = (
            (RFGLM & f"dnoise_params_id={glm_dnoise_params_id}" & f"rf_glm_params_id={glm_params_id}")
            * SplitRFGLM
            * TempRFGLMProperties
            * GLMContours
            * GLMContourMetrics
            * GLMContourOffset
    )

    if rf_quality_filter:
        rf_tab = (
                rf_tab
                & f"rf_cdia_um>={rf_cdia_um_range[0]}"
                & f"rf_cdia_um<={rf_cdia_um_range[1]}"
                & f"split_qidx>{q_rf_split_min}"
                & f"largest_contour_ratio>={lcontour_ratio_min}"
                & f"main_peak_lag<={max_main_peak_lag}"
        )

    rf_tab = rf_tab & get_roi_tab(quality_filter=quality_filter, roi_kind=roi_kind)

    if roi_kind == 'soma' and only_one_soma_rf:
        rf_tab &= get_single_soma_rf_filter(tab=rf_tab, reject_tags=reject_tags)

    if len(FLAGGED_NOISE_FILES[get_dataset()]) > 0:
        rf_tab &= (Presentation & [f"h5_header!='{f}'" for f in FLAGGED_NOISE_FILES[get_dataset()]])

    return rf_tab


def get_single_soma_rf_filter(tab, reject_tags=('none',)):
    # Restrict to one RF per cell if soma
    exp_keys = (Experiment & tab).fetch('KEY')
    best_rf_keys = []
    for exp_key in exp_keys:
        exp_rf_tab = tab & exp_key

        rf_cdia_um = exp_rf_tab.fetch('rf_cdia_um')  # Get largest RF
        best_rf_key = exp_rf_tab.proj().fetch(as_dict=True)[np.argmax(rf_cdia_um)]

        if (CellTags & best_rf_key).fetch1('cell_tag') in reject_tags:
            print(f"Rejecting {best_rf_key}")
            continue

        best_rf_keys.append(best_rf_key)

    return best_rf_keys


def get_field_stack_pos_tab():
    return FieldStackPos()


def get_roi_rf_morph_tab():
    return __get_rf_and_morph_tab('roi')


def get_soma_rf_morph_tab():
    return __get_rf_and_morph_tab('soma')


def __get_rf_and_morph_tab(roi_kind):
    if roi_kind == 'roi':
        rf_tab = get_roi_rf_tab()
    elif roi_kind == 'soma':
        rf_tab = get_soma_rf_tab()
    else:
        raise ValueError(f'Unknown roi_kind: {roi_kind}')

    morph_tab = get_morph_tab(quality_filter=False)
    rf_morph_tab = rf_tab * morph_tab

    return rf_morph_tab, rf_tab, morph_tab


def get_field_avg_offset():
    field_rf_tab = get_field_rf_tab()

    df_field_rfs = (
        field_rf_tab.proj(field_rf_dx_um='rf_dx_um', field_rf_dy_um='rf_dy_um', field_rf_d_um='rf_d_um')
    ).fetch(format='frame').reset_index()

    df_field_rfs['field'] = df_field_rfs['field'].apply(lambda x: x.replace('FieldROI', ''))
    df_field_rfs = df_field_rfs.drop('roi_id', axis=1)
    df_field_rfs['date'] = df_field_rfs['date']
    df_field_rfs = df_field_rfs.set_index(['date', 'exp_num', 'field'])

    field_avg_dx = df_field_rfs.groupby(['date', 'exp_num']).field_rf_dx_um.median()
    field_avg_dy = df_field_rfs.groupby(['date', 'exp_num']).field_rf_dy_um.median()

    return df_field_rfs, field_avg_dx, field_avg_dy


def get_pres_tab():
    pres_tab = Presentation * Presentation.ScanInfo * Presentation.RoiMask * (
            Presentation.StackAverages & dict(ch_name='wDataCh0'))
    return pres_tab


def get_paths_tab():
    return FieldPathPos * MorphPaths().proj('soma_xyz', stack='field')

def get_chirp_surround_tab():
    roi_tab = get_roi_tab(quality_filter=True)
    chirp_surround_tab = ChirpSurroundIndex() & roi_tab
    return chirp_surround_tab


def prep_df(df):
    id_cols = [col for col in df.columns if
               (col.endswith('_id') and not col in ['roi_id']) or col.endswith('_hash') or col in ['condition',
                                                                                                   'stim_name']]
    for id_col in id_cols:
        if df[id_col].nunique() == 1:
            df.drop(id_col, axis=1, inplace=True)
    df['cell_id'] = (df['date'].astype(str) + '_' + df['exp_num'].astype(str))
    df['field_id'] = (df['date'].astype(str) + '_' + df['exp_num'].astype(str) + '_' + df['field'].astype(str))
    df['group'] = df['group']
    return df


def get_tri_df(categorize=True):
    roi_tab = get_roi_tab(quality_filter=True)

    if get_dataset() == 'calcium':
        loc_tab = RetinalFieldLocationWing().proj(group="wing_side")
    else:
        loc_tab = RetinalFieldLocationCat().proj(group="nt_side")

    chirp_tab = (
                        (ChirpTransienceV2 & "stim_name='lChirp'").proj(chirp_tri='transience_index')
                        * ChirpQI()
                        * ChirpSurroundIndex().proj('chirp_surround_index')
                        * Averages().proj(chirp_avg='average_norm')
                        * FieldPosMetrics.RoiPosMetrics().proj(soma_dist='d_dist_to_soma')
                        * loc_tab
                ) & roi_tab.proj()

    rf_tab = (
            get_roi_rf_tab().proj(
                'trf', 'dt', rf_size='rf_cdia_um', surround_index='full_surround_index', rf_tri='transience_idx')
            * FieldPosMetrics.RoiPosMetrics().proj(soma_dist='d_dist_to_soma')
            * loc_tab
    )

    assert len((Roi & rf_tab).proj() - (Roi & chirp_tab).proj()) == 0

    scan_frequency = 31.25
    t_on_idx = int(np.round(2 * scan_frequency))
    t_off_idx = int(np.round(5 * scan_frequency))

    df_chirp = chirp_tab.fetch(format='frame').reset_index()
    df_chirp = prep_df(df_chirp).set_index(
        ['experimenter', 'date', 'exp_num', 'field', 'cond1', 'roi_id', 'cell_id', 'field_id'])
    df_chirp['chirp_is_on'] = df_chirp.chirp_avg.apply(
        lambda avg: np.mean(avg[t_on_idx:t_off_idx]) > 2 * np.std(np.mean(avg[:t_on_idx])))
    df_chirp.loc[~df_chirp.chirp_is_on, 'chirp_tri'] = np.nan

    df_rf = rf_tab.fetch(format='frame').reset_index()
    df_rf = prep_df(df_rf).set_index(
        ['experimenter', 'date', 'exp_num', 'field', 'cond1', 'roi_id', 'cell_id', 'field_id'])

    df = df_chirp.merge(df_rf[['trf', 'dt', 'rf_tri', 'rf_size', 'surround_index']], how='left', left_index=True,
                        right_index=True).reset_index()

    if categorize:
        df['cell_id'] = df['cell_id'].astype('category')
        df['field_id'] = df['field_id'].astype('category')
        df['group'] = df['group'].astype('category')

    return df
