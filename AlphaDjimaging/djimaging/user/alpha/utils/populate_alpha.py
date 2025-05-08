import os
import time
import warnings
from datetime import datetime
from pathlib import Path

from djimaging.user.alpha.schemas.alpha_schema import *
from matplotlib import pyplot as plt

# Set config
HOME_DIR = os.path.expanduser("~")
USERNAME = os.path.split(HOME_DIR)[-1]
CONFIG_FILE = os.path.join(HOME_DIR, "datajoint", f"dj_{USERNAME}_conf.json")  # Change if necessary
SCHEMA_PREFIX = f'ageuler_{USERNAME}_alpha_'  # Change if necessary

if not os.path.isfile(CONFIG_FILE):
    raise FileNotFoundError(f"Please enter the path to the config file in CONFIG_FILE")

if SCHEMA_PREFIX == 'enter_prefix_here_':
    raise ValueError("Please enter the schema prefix in SCHEMA_PREFIX")

# Specific to user
PROJECT_ROOT = os.path.abspath(Path(__file__).resolve().parents[6])

__cell_position_table = os.path.join(PROJECT_ROOT, "data/Ran/cell_positions_22deg_rot.csv")
__stim_noise_file = os.path.join(PROJECT_ROOT, "data/Stimulus/noise.h5")
__stim_noise_long_file = os.path.join(PROJECT_ROOT, "data/Stimulus/noise_2500.h5")
__roi_match_fig_folder = os.path.join(PROJECT_ROOT, "code/djimaging/djimaging/user/alpha/notebooks/figures_match_rois")

# Hyper parameters
__soma_roi_max_dist = 50
__soma_roi_min_area = 0

__dataset_dirs = {
    'calcium': os.path.join(PROJECT_ROOT, 'data/Ran/'),
    'glutamate': os.path.join(PROJECT_ROOT, 'data/Ran/'),

}

__dataset_experiments = {
    'calcium': [
        ('Ran', '20180610', 'C1'),
        ('Ran', '20200511', 'C2'),
        ('Ran', '20200708', 'C1'),
        ('Ran', '20200705', 'C2'),
        ('Ran', '20200511', 'C1'),
        ('Ran', '20200510', 'C1'),
        ('Ran', '20200608', 'C1'),
        ('Ran', '20200608', 'C2'),
        ('Ran', '20200607', 'C1'),
        ('Ran', '20200510', 'C2'),
        ('Ran', '20200621', 'C1'),
        ('Ran', '20200713', 'C1'),
        ('Ran', '20200823', 'C1'),
        ('Ran', '20200824', 'C1'),
        ('Ran', '20200824', 'C2'),
        ('Ran', '20200829', 'C1'),
        ('Ran', '20200831', 'C1'),
    ],
    'glutamate': [
        ('Ran', '20201103', 'C1'),
        ('Ran', '20201103', 'C2'),

        ('Ran', '20201110', 'C2'),
        ('Ran', '20201110', 'C3'),

        ('Ran', '20201111', 'C1'),

        ('Ran', '20201127', 'C1'),
        ('Ran', '20201127', 'C2'),

        ('Ran', '20201104', 'C1'),
        ('Ran', '20201104', 'C2'),
        ('Ran', '20201104', 'C3'),
        ('Ran', '20201104', 'C5'),
        ('Ran', '20201104', 'C6'),

    ],
}

__problem_files = [
    (os.path.join(PROJECT_ROOT, "data/Ran/sONa_glutamate/20201104/5/Pre/SMP_C5_d4_sinespot.h5"), 1768),
    (os.path.join(PROJECT_ROOT, "data/Ran/sONa_glutamate/20201127/2/Pre/SMP_C2_d5_Chirp.h5"), 3535),
    (os.path.join(PROJECT_ROOT, "data/Ran/sONa_calcium/20200510/2/Pre/SMP_C2_d1_lChirp.h5"), 3625),
]

__dataset_delete_unmatched_keys = {
    'calcium': [
    ],
    'glutamate': [
        {'experimenter': 'Ran', 'date': "20201127", 'exp_num': 1, 'field': 'd4'},
    ],
    'soma':
        []
}

__roi_stack_pos_params = dict(pad_more=150, pad_scale=1.1, dist_score_factor=2e-4)

__dataset_sta_dnoise_params_list = {
    'calcium': [
        dict(dnoise_params_id=1, fupsample_trace=1, fupsample_stim=0, lowpass_cutoff=3, ref_time='trace',
             fit_kind='gradient', pre_blur_sigma_s=0., post_blur_sigma_s=0.),
    ],
    'glutamate': [
        dict(dnoise_params_id=1, fupsample_trace=1, fupsample_stim=0, lowpass_cutoff=5, ref_time='trace',
             fit_kind='gradient', pre_blur_sigma_s=0., post_blur_sigma_s=0.),
    ],
}

__dataset_glm_dnoise_params_list = {
    'calcium': [
        dict(dnoise_params_id=1, fupsample_trace=1, fupsample_stim=0, lowpass_cutoff=3, ref_time='trace',
             fit_kind='gradient', pre_blur_sigma_s=0., post_blur_sigma_s=0.),
    ],
    'glutamate': [
        dict(dnoise_params_id=1, fupsample_trace=1, fupsample_stim=0, lowpass_cutoff=5, ref_time='trace',
             fit_kind='gradient', pre_blur_sigma_s=0., post_blur_sigma_s=0.),
    ]
}

__dataset_glm_params_list = {
    'calcium': [
        {'rf_glm_params_id': 10,
         'filter_dur_s_past': 1.2,
         'filter_dur_s_future': 0.2,
         'df_ts': (10,),
         'df_ws': (9,),
         'betas': (0.005,),
         'kfold': 0,
         'metric': 'mse',
         'output_nonlinearity': 'none',
         'other_params_dict': {
             'frac_test': 0,
             'min_iters': 100,
             'max_iters': 2000,
             'step_size': 0.1,
             'tolerance': 5,
             'alphas': (1.0,),
             'verbose': 100,
             'n_perm': 20,
             'min_cc': 0.2,
             'seed': 42,
             'fit_R': False,
             'fit_intercept': True,
             'init_method': 'random',
             'atol': 1e-05,
             'distr': 'gaussian',
             'step_size_finetune': 0.03}
         },
    ],
    'glutamate': [
        {'rf_glm_params_id': 10,
         'filter_dur_s_past': 1.2,
         'filter_dur_s_future': 0.2,
         'df_ts': (10,),
         'df_ws': (12,),
         'betas': (0.005,),
         'kfold': 0,
         'metric': 'mse',
         'output_nonlinearity': 'none',
         'other_params_dict': {
             'min_iters': 100,
             'max_iters': 2000,
             'step_size': 0.1,
             'tolerance': 5,
             'alphas': (1.0,),
             'verbose': 100,
             'n_perm': 0,
             'min_cc': 0.2,
             'seed': 42,
             'fit_R': False,
             'fit_intercept': True,
             'init_method': 'random',
             'atol': 1e-05,
             'distr': 'gaussian',
             'frac_test': 0,
             'step_size_finetune': 0.03}}
    ],
}

__glm_quality_params = dict(
    glm_quality_params_id=1, min_corrcoef=0.1, max_mse=-1., perm_alpha=0.05
)

__split_rf_params = dict(split_rf_params_id=1, blur_std=0., blur_npix=0, peak_nstd=0.5, npeaks_max=2)

__sta_params = dict(sta_params_id=1, filter_dur_s_past=1.2, filter_dur_s_future=0.2,
                    frac_train=1., frac_test=0., rf_method='sta', )


def get_dataset():
    if 'soma' in dj.config['schema_name']:
        dataset = 'soma'
    elif 'ca' in dj.config['schema_name']:
        dataset = 'calcium'
    elif 'glu' in dj.config['schema_name']:
        dataset = 'glutamate'
    else:
        return None
    return dataset


def populate_user(verbose=False):
    if verbose:
        print('Populate UserInfo')

    userinfo = {
        'experimenter': 'Ran',
        'data_dir': __dataset_dirs[get_dataset()],
        'datatype_loc': 0,
        'animal_loc': 10,
        'region_loc': 1,
        'field_loc': 2,
        'stimulus_loc': 3,
        'condition_loc': 4,
        'mask_alias': 'noise_dn_densenoise_dnoise_dnoisegc30_chirp_lchirp_sinespot'
    }

    UserInfo().upload_user(userinfo, verbose=0)


def populate_experiment_field(verbose=False, dataset=None):
    if dataset is None:
        dataset = get_dataset()

    if verbose:
        print(f'Populate Experiment for dataset {dataset}')

    for experimenter, date, cell in __dataset_experiments[dataset]:
        restrictions = {
            'experimenter': experimenter,
            'date': datetime.strptime(date, '%Y%m%d'),
            'exp_num': int(cell[1:])}

        Experiment().rescan_filesystem(restrictions=restrictions, verboselvl=1 if verbose > 10 else -1)

    RawDataParams().add_default(compute_from_stack=True, include_artifacts=False)

    if verbose:
        print('Populate Field')
    Field().rescan_filesystem(verboselvl=1 if verbose > 10 else -1)


def populate_stimulus(verbose=False):
    import h5py

    if verbose:
        print('Populate Stimulus')

    with h5py.File(__stim_noise_file, "r") as f:
        noise_stimulus_1500 = f['k'][:].T.astype(int)

    with h5py.File(__stim_noise_long_file, "r") as f:
        noise_stimulus_2500 = f['k'][:].T.astype(int)

    Stimulus().add_nostim(skip_duplicates=True)
    Stimulus().add_chirp(spatialextent=1000, stim_name='gChirp', alias="chirp_gchirp_globalchirp", skip_duplicates=True)
    Stimulus().add_chirp(spatialextent=300, stim_name='lChirp', alias="lchirp_localchirp", skip_duplicates=True)
    Stimulus().add_noise(stim_name='noise_1500', alias="noise_dn_densenoise_dnoise", stim_family='noise',
                         pix_n_x=20, pix_n_y=15, pix_scale_x_um=30, pix_scale_y_um=30, stim_trace=noise_stimulus_1500,
                         skip_duplicates=True, ntrigger_rep=1500)
    Stimulus().add_noise(stim_name='noise_2500', alias="dnoisegc30", stim_family='noise',
                         pix_n_x=20, pix_n_y=15, pix_scale_x_um=30, pix_scale_y_um=30, stim_trace=noise_stimulus_2500,
                         skip_duplicates=True, ntrigger_rep=2500)
    Stimulus().add_stimulus(stim_name='sinespot', alias="sinespot_sine_spot", isrepeated=True, ntrigger_rep=6,
                            trial_info=[1, 2, 3, 4, 5, 6], skip_duplicates=True)


def populate_presentation(verbose=False, processes=1):
    if verbose:
        print('Populate Presentation')
    Presentation().populate(display_progress=verbose, processes=processes)


def fix_broken_recordings(plot=False):
    # Fix corrupted files where recordings stopped working after some time.
    for problem_file, bad_idx in __problem_files:
        problem_trace_keys = (Traces & (Presentation & dict(h5_header=problem_file)).proj()).fetch('KEY')

        if len(problem_trace_keys) == 0:
            continue

        if plot:
            from djimaging.utils.scanm_utils import load_traces_from_h5, load_triggers_from_h5

            ts, tts = load_traces_from_h5(problem_file)
            trs = load_triggers_from_h5(problem_file)

            plt.figure(figsize=(10, 5))
            plt.plot(tts[:, 0], ts[:, 0] - np.mean(ts[:, 0]))
            plt.axvline(tts[bad_idx, 0])
            plt.axvline(trs[0][-1], c='r')
            plt.xlim(tts[bad_idx, 0] - 10, tts[bad_idx, 0] + 10)
            plt.show()

        for key in problem_trace_keys:
            if len((PreprocessTraces & key).proj()) > 0:
                trace_size = (Traces & key).fetch1('trace').size
                if trace_size != bad_idx:
                    raise ValueError(f'PreprocessTraces {problem_file} already populated before correction\n'
                                     f'{key}\n'
                                     f'{trace_size} != {bad_idx}')
            else:
                new_key = key.copy()
                new_key['trace'] = (Traces & key).fetch1('trace')[:bad_idx]
                new_key['trace_times'] = (Traces & key).fetch1('trace_times')[:bad_idx]
                Traces().update1(new_key)


def populate_traces(verbose=False, processes=1):
    if verbose:
        print('Populate Traces')
    Traces().populate(display_progress=verbose, processes=processes)

    if verbose:
        print('fix_broken_recordings')
    fix_broken_recordings()

    if verbose:
        print('Populate PreprocessTraces')
    PreprocessParams().add_default(
        skip_duplicates=True, preprocess_id=1, window_length=60, poly_order=3, non_negative=False,
        subtract_baseline=True, standardize=2, f_cutoff=None, fs_resample=0)

    PreprocessTraces().populate(
        "preprocess_id=1", "stim_name!='sinespot'", display_progress=verbose, processes=processes)

    # For Sinespot
    PreprocessParams().add_default(
        skip_duplicates=True, preprocess_id=2, window_length=10, poly_order=3, non_negative=False,
        subtract_baseline=True, standardize=2, f_cutoff=None, fs_resample=0)

    PreprocessTraces().populate(
        "preprocess_id=2", "stim_name='sinespot'", display_progress=verbose, processes=processes)

    if verbose:
        print('Populate Snippets')
    Snippets().populate(display_progress=verbose, processes=processes)

    if verbose:
        print('Populate Averages')
    Averages().populate(display_progress=verbose, processes=processes)

    if verbose:
        print('Populate Downsampled Averages')
    DownsampledAverages().populate(display_progress=verbose, processes=processes)


def populate_metrics(verbose=False, processes=1):
    if verbose:
        print('ChirpQI')
    ChirpQI().populate(display_progress=verbose, processes=processes)
    if verbose:
        print('SineSpotQI')
    SineSpotQI().populate(display_progress=verbose, processes=processes)


def populate_quality(verbose=False, processes=1):
    QualityParams().add(skip_duplicates=True)
    if verbose:
        print('QualityIndex')
    QualityIndex().populate(display_progress=verbose, processes=processes)


def populate_cell_positions(verbose=True):
    if verbose:
        print('Populate Cell Positions')

    RetinalFieldLocationTableParams().add_table(
        table_path=__cell_position_table, col_field='',
        col_ventral_dorsal_pos='ventral-dorsal', col_temporal_nasal_pos='temporal-nasal', skip_duplicates=True)

    RetinalFieldLocationFromTable().populate()

    if verbose:
        print('Populate RetinalFieldLocationCat')

    RetinalFieldLocationCat().populate()

    if verbose:
        print('Populate RetinalFieldLocationWing')

    RetinalFieldLocationWing().populate()

    if verbose:
        print('Populate Cell Tags')


def populate_cell_tags():
    if len(CellTags().proj()) == 0:
        cat_tab = RetinalFieldLocationWing().proj(group='wing_side') \
            if get_dataset() == 'calcium' else RetinalFieldLocationCat().proj(group='nt_side')

        sides = np.unique((cat_tab & (RoiKind() & "roi_kind='stack'")).fetch('group'))

        for side in sides:
            restr = (cat_tab & dict(group=side)) & (RoiKind & "roi_kind='stack'")
            cell_keys = (CellTags().key_source & restr).fetch('KEY')
            nasal_temporal_pos = -(RetinalFieldLocationFromTable & restr).fetch('temporal_nasal_pos')
            cell_keys = [x for _, x in sorted(zip(nasal_temporal_pos, cell_keys))]

            for i, cell_key in enumerate(cell_keys):
                temporal_nasal_pos = (
                        RetinalFieldLocationFromTable() & cell_key & (RoiKind() & "roi_kind='stack'")).fetch1(
                    'temporal_nasal_pos')
                tag = f"{side}{i + 1}"
                print(cell_key)
                print(tag, temporal_nasal_pos, side)
                CellTags().insert1(dict(**cell_key, cell_tag=tag))


def populate_morphology(verbose=True, processes=1):
    if verbose:
        print('Populate SWC')
    SWC().populate(display_progress=verbose, processes=processes)
    if verbose:
        print('Populate MorphPaths')
    MorphPaths().populate(display_progress=verbose, processes=processes)
    if verbose:
        print('Populate LineStack')
    LineStack().populate(display_progress=verbose, processes=processes)


def populate_additional_morph_metrics(verbose=True, processes=1):
    if verbose:
        print('Populate LineStack')
    RelativeRoiPos().populate(display_progress=verbose, processes=processes)
    if verbose:
        print('Populate LineStack')
    FieldPathPos().populate(display_progress=verbose, processes=processes)
    if verbose:
        print('Populate ConvexHull')
    ConvexHull().populate(display_progress=True)


def populate_fit_to_morphology(
        verbose=True, processes=1, delete_preselected_unmatched=True, delete_warning_unmatched=True):
    if verbose:
        print('Fit ROIs')

    fig_folder = __roi_match_fig_folder + '_' + get_dataset()
    params = __roi_stack_pos_params.copy()

    if verbose:
        print('RoiStackPosParams')
    RoiStackPosParams().add_default(fig_folder=fig_folder, **params, skip_duplicates=True)
    if verbose:
        print('FieldStackPos')

    keys = ((FieldStackPos().key_source & (RoiKind & dict(roi_kind='roi'))) - FieldStackPos().proj()).fetch('KEY')
    np.random.shuffle(keys)
    for key in keys:
        print(key)
        pos_key, info_key, roi_keys = FieldStackPos()._fetch_and_compute(key)

        retry = 0
        while retry < 3:
            try:
                FieldStackPos().insert1(pos_key, allow_direct_insert=True, skip_duplicates=True)
                FieldStackPos().FitInfo().insert1(info_key, allow_direct_insert=True, skip_duplicates=True)
                for roi_key in roi_keys:
                    FieldStackPos().RoiStackPos().insert1(roi_key, allow_direct_insert=True, skip_duplicates=True)
                retry = 3
            except dj.errors.LostConnectionError:
                retry += 1
                print('Lost connection. Retrying...')
                dj.conn()
                time.sleep(3)

    if verbose:
        print('FieldPathPos')
    FieldPathPos().populate(display_progress=verbose, processes=processes)

    for key in (FieldStackPos() & "rec_c_warning_flag = 1").proj().fetch(as_dict=True):
        FieldStackPos().plot1(key=key)
        plt.show()
        if delete_warning_unmatched:
            (FieldStackPos() & key).delete()
        else:
            print('WARNING: the following field should probably be removed:\n', key)

    for key in __dataset_delete_unmatched_keys[get_dataset()]:
        if len(FieldStackPos() & key) > 0:
            FieldStackPos().plot1(key=key)
            plt.show()
            if delete_preselected_unmatched:
                print('Deleting field that could not be mapped to morphology:\n', key)
                (FieldStackPos() & key).delete()
            else:
                print('WARNING: the following field should probably be removed:\n', key)

    if verbose:
        print('FieldCalibratedStackPos')
    FieldCalibratedStackPos().populate(display_progress=verbose, processes=processes)
    if verbose:
        print('FieldPosMetrics')
    FieldPosMetrics().populate(display_progress=verbose, processes=processes)


def check_params(tab, params):
    ids = {k: v for k, v in params.items()
           if k.endswith('params_id') or k.endswith('params_hash')}
    table_params = (tab & ids).fetch1()
    for k in set(table_params.keys()).intersection(params.keys()):
        if table_params[k] != params[k]:
            raise ValueError(
                f"""Params already present and not the same in table {tab!r}!
                {k}:
                {table_params[k]}
                {params[k]}""")


def populate_rf_glms_traces(verbose=True, processes=1):
    if verbose:
        print('GLMDNoiseTraceParams')
    for params in __dataset_glm_dnoise_params_list[get_dataset()]:
        GLMDNoiseTraceParams().add_default(**params, skip_duplicates=True)
        check_params(GLMDNoiseTraceParams, params)
    if verbose:
        print('GLMDNoiseTrace')
    GLMDNoiseTrace().populate(display_progress=verbose, processes=processes, suppress_errors=False)


def populate_rf_glms(*restrictions, verbose=True, processes=1, suppress_fit_errors=True):
    if verbose:
        print('Fit GLMs')

    eff_processes = int(processes / 10) + 1

    for params in __dataset_glm_params_list[get_dataset()]:
        RFGLMParams().add_default(**params, other_params_dict=dict(), skip_duplicates=True)
        check_params(RFGLMParams, params)

    RFGLM().populate(*restrictions, display_progress=verbose, suppress_errors=suppress_fit_errors,
                     processes=eff_processes, order='random',
                     make_kwargs=dict(suppress_outputs=eff_processes > 1))


def populate_rf_glm_properties(verbose=False, processes=1):
    if verbose:
        print('SplitRFGLMParams')
    SplitRFGLMParams().add_default(**__split_rf_params, skip_duplicates=True)
    check_params(SplitRFGLMParams, __split_rf_params)

    if verbose:
        print('SplitRFGLM')
    SplitRFGLM().populate(display_progress=verbose, processes=processes)

    if verbose:
        print('Fit Temporal RF properties')
    TempRFGLMProperties().populate(display_progress=verbose, processes=processes)


def populate_experiments(verbose=True, processes=1):
    populate_user(verbose=verbose)
    populate_experiment_field(verbose=verbose)


def populate_core(verbose=True, processes=1):
    Roi().populate(display_progress=verbose, processes=processes)
    RoiKind().populate(display_progress=verbose, processes=processes)

    populate_stimulus(verbose=verbose)

    populate_presentation(verbose=verbose, processes=processes // 4)
    populate_traces(verbose=verbose, processes=processes)
    populate_metrics(verbose=verbose, processes=processes)
    if verbose:
        print('Done')


def add_field_rois(verbose=False):
    add_merge_rois(suffix='FieldROI', max_dist=None, min_size_um2=None, verbose=verbose)
    FieldRoiPosMetrics().populate(display_progress=verbose, processes=20)


def add_soma_rois(verbose=False):
    add_merge_rois(suffix='SomaROI', max_dist=__soma_roi_max_dist, min_size_um2=__soma_roi_min_area, verbose=verbose)


def add_merge_rois(suffix, max_dist, min_size_um2, verbose=False):
    """Merge ROIs to field ROIs or soma ROIs and add to Field and Presentation"""

    field_keys = Field().proj().fetch(as_dict=True)

    for field_key in field_keys:
        pres_keys = (Presentation & field_key).proj().fetch(as_dict=True)

        field_i = (Field() & field_key).fetch1()

        if len((Field.RoiMask() & field_key).proj()) == 0:
            continue

        if ('field' in field_i['field'].lower()) or ('soma' in field_i['field'].lower()):
            continue

        field_i_roi_mask = (Field.RoiMask() & field_key).fetch1().copy()
        field_i_avgs = (Field.StackAverages() & field_key).fetch()
        new_field_name = field_i['field'] + suffix

        field_mask = field_i_roi_mask['roi_mask'].copy()

        if max_dist is not None:
            include_rois, roi_size_um2s = (
                    Roi & field_key & "artifact_flag=0" &
                    (FieldPosMetrics.RoiPosMetrics() & f"d_dist_to_soma<={max_dist}") &
                    (FieldCalibratedStackPos.RoiCalibratedStackPos() & "success_cal_flag=1")
            ).fetch('roi_id', 'roi_size_um2')

            if len(include_rois) == 0:
                continue
            elif np.sum(roi_size_um2s) < min_size_um2:
                print(field_key)
                print('too small ROIs', np.sum(roi_size_um2s), include_rois)
                continue
            else:
                new_field_mask = gen_new_mask(field_mask, include_rois=include_rois)
                if verbose:
                    (FieldStackPos & field_key).plot1()
                    plt.show()
        else:
            include_rois = (
                    Roi & field_key & "artifact_flag=0" &
                    (FieldCalibratedStackPos.RoiCalibratedStackPos() & "success_cal_flag=1")
            ).fetch('roi_id')

            if len(include_rois) == 0:
                warnings.warn('No morph matched ROIs found for field ' + str(field_key))
                continue

            new_field_mask = gen_new_mask(field_mask, include_rois=include_rois)

        field_i['field'] = new_field_name
        field_i_roi_mask['field'] = new_field_name
        field_i_roi_mask['roi_mask'] = new_field_mask

        if len((Field().proj() & field_i)) == 0:
            if verbose:
                print(field_key)

            Field().insert1(field_i, allow_direct_insert=True)
            Field.RoiMask().insert1(field_i_roi_mask, allow_direct_insert=True)
            for field_i_avg in field_i_avgs:
                field_i_avg['field'] = new_field_name
                Field.StackAverages().insert1(field_i_avg, allow_direct_insert=True)

        if len((Presentation().proj() & field_i)) == 0:
            for pres_key in pres_keys:
                if verbose:
                    print('\n', pres_key)

                pres_i = (Presentation() & pres_key).fetch1()
                pres_i_roi_mask = (Presentation.RoiMask() & pres_key).fetch1()
                pres_i_info = (Presentation.ScanInfo() & pres_key).fetch1()
                pres_i_avgs = (Presentation.StackAverages() & pres_key).fetch()

                pres_mask = pres_i_roi_mask['roi_mask'].copy()

                new_pres_mask = gen_new_mask(pres_mask, include_rois=include_rois)

                pres_i['field'] = new_field_name
                pres_i_info['field'] = new_field_name
                pres_i_roi_mask['field'] = new_field_name
                pres_i_roi_mask['roi_mask'] = new_pres_mask

                Presentation().insert1(pres_i, allow_direct_insert=True)
                Presentation.RoiMask().insert1(pres_i_roi_mask, allow_direct_insert=True)
                Presentation.ScanInfo().insert1(pres_i_info, allow_direct_insert=True)
                for pres_i_avg in pres_i_avgs:
                    pres_i_avg['field'] = new_field_name
                    Presentation.StackAverages().insert1(pres_i_avg, allow_direct_insert=True)


def gen_new_mask(field_mask, include_rois=None):
    new_mask = field_mask.copy()

    all_roi_ids = np.unique(-field_mask[field_mask < 0])

    if include_rois is None:
        include_rois = all_roi_ids

    for roi_id in all_roi_ids:
        if roi_id not in include_rois:
            new_mask[field_mask == -roi_id] = 1
        else:
            new_mask[field_mask == -roi_id] = -1

    return new_mask
