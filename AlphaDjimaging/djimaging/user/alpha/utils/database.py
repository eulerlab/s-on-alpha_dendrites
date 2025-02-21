import os

import pandas as pd
from djimaging.user.alpha.schemas.alpha_schema import *
from djimaging.user.alpha.utils.populate_alpha import SCHEMA_PREFIX, CONFIG_FILE, get_dataset, __cell_position_table
from djimaging.utils.dj_utils import get_secondary_keys

__rf_kind = {'calcium': 'glm', 'glutamate': 'glm'}

__rf_fit_kind = {'calcium': 'contour', 'glutamate': 'contour'}

__quality_id = 1

__sta_dnoise_params_id = 1
__sta_params_id = 1
__q_rf_split_min = 0.
__q_rf_fit_min = 0.35

__glm_dnoise_params_id = 1
__glm_params_id = {'calcium': 10, 'glutamate': 10}
__q_rf_glm_min_cc = -2
__q_rf_glm_split_min = 0.35
__q_rf_glm_fit_min = 0.35
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


def get_roi_tab(quality_filter, roi_kind='roi') -> dj.Table:
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


def get_field_tab(quality_filter=False, roi_kind='any') -> dj.Table:
    if roi_kind == 'any':
        if quality_filter:
            raise ValueError("Quality filter not implemented for any ROI kind")
        return Field()
    else:
        return Field & get_roi_tab(quality_filter=quality_filter, roi_kind=roi_kind)


def get_location_tab(quality_filter=True, roi_kind='roi') -> dj.Table:
    location_tab = RetinalFieldLocationFromTable & get_roi_kind_filter(roi_kind=roi_kind)

    if get_dataset() == 'calcium':
        location_tab *= RetinalFieldLocationWing().proj(group="wing_side")
    elif get_dataset() == 'glutamate':
        location_tab *= RetinalFieldLocationCat().proj(group="nt_side")

    return location_tab & get_experiment_tab(quality_filter=quality_filter, roi_kind=roi_kind)


def get_morph_tab(quality_filter=True, roi_kind='roi', rename=True, include_linestack=False) -> dj.Table:
    morph_tab = MorphPaths * RetinalFieldLocationFromTable * SWC * ConvexHull * CellTags

    if include_linestack:
        morph_tab *= LineStack

    if get_dataset() == 'calcium':
        morph_tab *= RetinalFieldLocationWing().proj(group="wing_side")
    elif get_dataset() == 'glutamate':
        morph_tab *= RetinalFieldLocationCat().proj(group="nt_side")

    morph_tab & get_experiment_tab(quality_filter=quality_filter, roi_kind=roi_kind).proj()

    if rename:
        proj = {k: k for k in get_secondary_keys(morph_tab)}
        proj['field_stack'] = 'field'
        morph_tab = morph_tab.proj(**proj)

    return morph_tab


def get_averages_tab(quality_filter=True, downsampled=False, roi_kind='roi') -> dj.Table:
    if downsampled:
        avg_tab = DownsampledAverages
    else:
        avg_tab = Averages
    return avg_tab & get_roi_tab(quality_filter=quality_filter, roi_kind=roi_kind)


def get_lchirp_tab(quality_filter=True, downsampled=False, roi_kind='roi') -> dj.Table:
    avg_tab = get_averages_tab(quality_filter=quality_filter, downsampled=downsampled, roi_kind=roi_kind)
    return avg_tab & "stim_name='lChirp'"


def get_gchirp_tab(quality_filter=True, downsampled=False, roi_kind='roi') -> dj.Table:
    avg_tab = get_averages_tab(quality_filter=quality_filter, downsampled=downsampled, roi_kind=roi_kind)
    return avg_tab & "stim_name='gChirp'"


def get_lchirp_gchirp_tab(quality_filter=True, downsampled=False, roi_kind='roi') -> dj.Table:
    secondary_keys = get_secondary_keys(Averages)

    gchirp_tab = get_gchirp_tab(quality_filter=quality_filter, downsampled=downsampled, roi_kind=roi_kind).proj(
        gchirp='stim_name', **{f"gchirp_{k}": k for k in secondary_keys})
    lchirp_tab = get_lchirp_tab(quality_filter=quality_filter, downsampled=downsampled, roi_kind=roi_kind).proj(
        lchirp='stim_name', **{f"lchirp_{k}": k for k in secondary_keys})

    return gchirp_tab * lchirp_tab


def get_sinespot_tab(quality_filter=True, downsampled=False, roi_kind='roi') -> dj.Table:
    avg_tab = get_averages_tab(quality_filter=quality_filter, downsampled=downsampled, roi_kind=roi_kind)
    return (avg_tab & "stim_name='sinespot'") * SineSpotSurroundIndex() * SineSpotQI()


def get_clustering_tab(quality_filter=True) -> dj.Table:
    clust_tab = (Clustering.RoiCluster &
                 f"features_id={__features_id[get_dataset()]}" &
                 f"clustering_id={__clustering_id[get_dataset()]}")
    return clust_tab & get_roi_tab(quality_filter=quality_filter, roi_kind='roi')


def get_clustering_params_tab() -> dj.Table:
    return (FeaturesParams().proj(
        'ncomps', 'stim_names', 'norm_trace', feature_kind='kind', feature_params_dict='params_dict')
            * ClusteringParameters) & f"clustering_id={__clustering_id[get_dataset()]}"


def get_clustering_features_tab() -> dj.Table:
    return Features & f"features_id={__features_id[get_dataset()]}"


def get_roi_pos_tab(quality_filter=True, roi_kind='roi') -> dj.Table:
    return (
            (FieldStackPos
             * FieldPosMetrics.RoiPosMetrics
             * (RelativeRoiPos & [dict(stim_name='noise_1500'), dict(stim_name='noise_2500')]).proj(
                        'roi_dx_um', 'roi_dy_um', 'roi_d_um', pos_stim_name='stim_name')
             * FieldPathPos
             * FieldCalibratedStackPos.RoiCalibratedStackPos
             ) & get_roi_tab(quality_filter=quality_filter, roi_kind=roi_kind))


def get_rf_tab(
        roi_kind='roi', kind='glm', fit_kind=None,
        q_rf_split_min=None, q_rf_fit_min=None, rf_cdia_um_range=None,
        quality_filter=True, rf_quality_filter=None, only_one_soma_rf=True,
        reject_tags=('none',)  # ROI is outside RF, this is not a good somatic RF proxy
) -> dj.Table:
    kind = kind if kind is not None else __rf_kind[get_dataset()]
    fit_kind = fit_kind if fit_kind is not None else __rf_fit_kind[get_dataset()]

    if kind == 'glm':
        rf_tab = get_rf_glm_tab(
            roi_kind=roi_kind, fit_kind=fit_kind,
            quality_filter=quality_filter, rf_quality_filter=rf_quality_filter,
            q_rf_split_min=q_rf_split_min, q_rf_fit_min=q_rf_fit_min,
            rf_cdia_um_range=rf_cdia_um_range)
    else:
        raise ValueError(f"Unknown RF kind: {kind}")

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

        if len(exp_rf_tab.proj()) > 0:
            print(f'Found {len(exp_rf_tab.proj())} RFs for {exp_key}')

        rf_cdia_um = exp_rf_tab.fetch('rf_cdia_um')  # Get largest RF
        best_rf_key = exp_rf_tab.proj().fetch(as_dict=True)[np.argmax(rf_cdia_um)]

        if (CellTags & best_rf_key).fetch1('cell_tag') in reject_tags:
            print(f"Rejecting {best_rf_key}")
            continue

        best_rf_keys.append(best_rf_key)

    return best_rf_keys


def get_rf_glm_tab(
        roi_kind='roi', fit_kind=None,
        quality_filter=True, q_rf_split_min=None, q_rf_fit_min=None, rf_cdia_um_range=None,
        rf_quality_filter=None, glm_dnoise_params_id=None, glm_params_id=None,
        lcontour_ratio_min=None) -> dj.Table:
    if rf_quality_filter is None:
        rf_quality_filter = quality_filter

    from djimaging.user.alpha.schemas.alpha_schema import RFGLM, SplitRFGLM, TempRFGLMProperties

    lcontour_ratio_min = lcontour_ratio_min if lcontour_ratio_min is not None else __lcontour_ratio_min
    fit_kind = fit_kind if fit_kind is not None else __rf_fit_kind[get_dataset()]
    q_rf_split_min = q_rf_split_min if q_rf_split_min is not None else __q_rf_glm_split_min
    q_rf_fit_min = q_rf_fit_min if q_rf_fit_min is not None else __q_rf_glm_fit_min
    rf_cdia_um_range = rf_cdia_um_range if rf_cdia_um_range is not None else __rf_cdia_um_range[get_dataset()]
    glm_dnoise_params_id = glm_dnoise_params_id if glm_dnoise_params_id is not None else __glm_dnoise_params_id
    glm_params_id = glm_params_id if glm_params_id is not None else __glm_params_id[get_dataset()]
    max_main_peak_lag = __rf_max_lag[get_dataset()]

    rf_tab = (
            (RFGLM & f"dnoise_params_id={glm_dnoise_params_id}" & f"rf_glm_params_id={glm_params_id}") *
            SplitRFGLM * TempRFGLMProperties)

    if fit_kind == 'pos_dog':
        rf_tab = rf_tab * FitPosDoG2DRFGLM * GlmPosDogOffset
    elif fit_kind == 'dog':
        rf_tab = rf_tab * FitDoG2DRFGLM * GlmDogOffset
    elif fit_kind == 'gauss':
        rf_tab = rf_tab * FitGauss2DRFGLM * GlmGaussOffset
    elif fit_kind == 'contour':
        rf_tab = rf_tab * GLMContours * GLMContourMetrics * GLMContourOffset
    elif fit_kind == 'none':
        pass
    else:
        raise ValueError(f"Unknown fit kind: {fit_kind}")

    if rf_quality_filter:
        rf_tab = rf_tab & f"main_peak_lag<={max_main_peak_lag}"

        if fit_kind != 'none':
            rf_tab = (rf_tab &
                      f"rf_cdia_um>={rf_cdia_um_range[0]}" &
                      f"rf_cdia_um<={rf_cdia_um_range[1]}" &
                      f"split_qidx>{q_rf_split_min}")

        if fit_kind == 'contour':
            rf_tab = rf_tab & f"largest_contour_ratio>={lcontour_ratio_min}"
        elif fit_kind != 'none':
            rf_tab = rf_tab & f"rf_qidx>{q_rf_fit_min}"

    return rf_tab & get_roi_tab(quality_filter=quality_filter, roi_kind=roi_kind)


def drop_single_value_df_column(df, column):
    try:
        df.reset_index(column, inplace=True)
    except KeyError:
        pass

    if column not in df.columns:
        raise KeyError(f"{column} not in {df.columns}")

    if df[column].apply(lambda x: x.lower()).nunique() != 1:
        raise KeyError(f"{column} should have only one value but has {df[column].nunique()}: {df[column].unique()}")
    df.drop(column, axis=1, inplace=True)


def get_morph_df() -> pd.DataFrame:
    df_morph = get_morph_tab().fetch(format='frame')
    drop_single_value_df_column(df_morph, 'experimenter')
    drop_single_value_df_column(df_morph, 'field_stack')
    drop_single_value_df_column(df_morph, 'table_hash')
    return df_morph


def get_field_stack_pos_tab():
    return FieldStackPos


def get_loc_df(rf_kind=None, rf_fit_kind=None, annotate_cells=None, roi_kind='roi') -> pd.DataFrame:
    loc_tab = get_location_tab()

    df_cell_location = loc_tab.fetch(format='frame')
    df_cell_location = df_cell_location.droplevel((0, 3, 4)).drop_duplicates()

    if roi_kind == 'soma':
        rf_tab = get_rf_tab(kind=rf_kind, fit_kind=rf_fit_kind, quality_filter=False, roi_kind=roi_kind)
        df_soma_rf = (loc_tab * rf_tab).fetch(format='frame').droplevel((0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12))
        df_cell_location['pd-RF'] = False
        df_cell_location.loc[df_soma_rf.index, 'pd-RF'] = True

    if annotate_cells is not None:
        df_cell_location['annotate'] = False
        df_cell_location.loc[annotate_cells.index, 'annotate'] = True

    return df_cell_location


def get_rf_and_morph_tab(roi_kind, rf_kind=None, rf_fit_kind=None, quality_filter=True):
    rf_tab = get_rf_tab(kind=rf_kind, fit_kind=rf_fit_kind, quality_filter=quality_filter,
                        roi_kind=roi_kind)
    morph_tab = get_morph_tab(quality_filter=False, rename=True, roi_kind='roi')  # Always use roi here
    rf_morph_tab = rf_tab * morph_tab

    if roi_kind == 'soma':
        cells = rf_morph_tab.proj('group').fetch(format='frame').reset_index()
        assert len(cells) == np.unique(
            cells['experimenter'].astype(str) + cells['date'].astype(str) + cells['exp_num'].astype(str)).size
    else:
        cells = None

    return rf_morph_tab, rf_tab, morph_tab, cells


def get_field_avg_offset(rf_kind=None, rf_fit_kind=None, rf_quality_filter=True):
    rf_kind = rf_kind if rf_kind is not None else __rf_kind[get_dataset()]
    rf_fit_kind = rf_fit_kind if rf_fit_kind is not None else __rf_fit_kind[get_dataset()]

    field_rf_tab = get_rf_tab(
        roi_kind='field', kind=rf_kind, fit_kind=rf_fit_kind, rf_quality_filter=rf_quality_filter)

    df_field_rfs = (
        field_rf_tab.proj(field_rf_dx_um='rf_dx_um', field_rf_dy_um='rf_dy_um', field_rf_d_um='rf_d_um')
    ).fetch(format='frame').reset_index()

    df_field_rfs['field'] = df_field_rfs['field'].apply(lambda x: x.replace('FieldROI', ''))
    df_field_rfs = df_field_rfs.drop('roi_id', axis=1)
    df_field_rfs['date'] = df_field_rfs['date'].astype(str)
    df_field_rfs = df_field_rfs.set_index(['date', 'exp_num', 'field'])

    field_avg_dx = df_field_rfs.groupby(['date', 'exp_num']).field_rf_dx_um.median()
    field_avg_dy = df_field_rfs.groupby(['date', 'exp_num']).field_rf_dy_um.median()

    return df_field_rfs, field_avg_dx, field_avg_dy


def get_default_rf_and_rf_fit_kind():
    rf_kind = __rf_kind[get_dataset()]
    rf_fit_kind = __rf_fit_kind[get_dataset()]
    return rf_kind, rf_fit_kind


def get_pres_tab():
    pres_tab = Presentation * Presentation.ScanInfo * Presentation.RoiMask * (
            Presentation.StackAverages & dict(ch_name='wDataCh0'))
    return pres_tab


def get_paths_tab():
    return FieldPathPos * MorphPaths().proj('soma_xyz', stack='field')
