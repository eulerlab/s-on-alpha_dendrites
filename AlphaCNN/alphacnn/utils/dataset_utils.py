import os
import warnings

from alphacnn.utils.data_utils import make_hash, make_dir, save_var, save_config, load_var, load_config


def get_stim_dir(stim_conf):
    stim_hash = make_hash(stim_conf)
    stim_dir = f"stimulus_{stim_hash}"
    return stim_dir


def get_response_subdir(output_dir, model_conf, encoder_conf):
    response_hash = make_hash(dict(model=model_conf, encoder=encoder_conf))
    response_subdir = os.path.join(output_dir, f"response_{response_hash}")
    return response_subdir


def save_stimuli_meta(output_dir, stim_conf):
    assert os.path.isdir(output_dir), output_dir
    stim_dir = get_stim_dir(stim_conf)

    output_dir_stim = os.path.join(output_dir, stim_dir)

    if os.path.isdir(output_dir_stim):
        warnings.warn(f'{output_dir_stim} already exists.')

    output_dir_stim_stimuli = os.path.join(output_dir_stim, 'stimuli')
    output_dir_stim_targets = os.path.join(output_dir_stim, 'target_pos')

    make_dir(output_dir_stim)
    make_dir(output_dir_stim_stimuli)
    make_dir(output_dir_stim_targets)
    save_config(stim_conf, os.path.join(output_dir_stim, 'stim_conf.pkl'))

    return output_dir_stim


def save_stimulus(output_dir, stimulus, target_pos, suffix, stim_conf, stim_to_video_paths=None):
    assert os.path.isdir(output_dir), output_dir

    stim_dir = get_stim_dir(stim_conf)
    output_dir_stim = os.path.join(output_dir, stim_dir)
    output_dir_stim_stimuli = os.path.join(output_dir, stim_dir, 'stimuli')
    output_dir_stim_targets = os.path.join(output_dir, stim_dir, 'target_pos')

    save_var(stimulus, os.path.join(output_dir_stim_stimuli, f'stimulus_{suffix}.pkl'))
    save_var(target_pos, os.path.join(output_dir_stim_targets, f'target_pos_{suffix}.pkl'))

    if stim_to_video_paths is not None:
        save_var(stim_to_video_paths, os.path.join(output_dir, stim_dir, f'stim_to_video_paths.pkl'))

    return output_dir_stim


def save_response(output_dir, responses, model_conf, encoder_conf):
    response_subdir = get_response_subdir(output_dir, model_conf, encoder_conf)
    print("Save response to:", response_subdir)

    if not os.path.isdir(response_subdir):
        print(f'Create folder: {response_subdir}')
    make_dir(response_subdir)

    save_var(responses, os.path.join(response_subdir, 'responses.pkl'))
    save_config(model_conf, os.path.join(response_subdir, 'model_conf.pkl'))
    save_config(encoder_conf, os.path.join(response_subdir, 'encoder_conf.pkl'))


def get_stim_id(file):
    return os.path.split(file)[1].split('.pkl')[0][-4:]


def load_stimuli(input_dir, load_raw_stimuli=True, load_target_pos=True):
    assert os.path.isdir(input_dir), input_dir

    stimulus_files = sorted([os.path.join(input_dir, 'stimuli', f)
                             for f in os.listdir(os.path.join(input_dir, 'stimuli'))
                             if f.startswith('stim')])
    target_pos_files = sorted([os.path.join(input_dir, 'target_pos', f)
                               for f in os.listdir(os.path.join(input_dir, 'target_pos'))
                               if f.startswith('target')])
    stimulus_ids = set([get_stim_id(f) for f in stimulus_files])
    target_pos_ids = set([get_stim_id(f) for f in target_pos_files])

    if len(stimulus_ids - target_pos_ids) > 0:
        warnings.warn(f'Found stimulus {len(stimulus_ids - target_pos_ids)} files without target pos file')

    if len(target_pos_ids - stimulus_ids) > 0:
        warnings.warn(f'Found {len(target_pos_ids - stimulus_ids)} target pos files without stimulus file')

    shared_ids = stimulus_ids.intersection(target_pos_ids)

    stimulus_files = [f for f in stimulus_files
                      if any([shared_id in os.path.split(f)[1] for shared_id in shared_ids])]
    target_pos_files = [f for f in target_pos_files
                        if any([shared_id in os.path.split(f)[1] for shared_id in shared_ids])]

    if load_raw_stimuli:
        stimuli = [load_var(f) for f in stimulus_files]
    else:
        stimuli = None

    if load_target_pos:
        target_pos = [load_var(f) for f in target_pos_files]
    else:
        target_pos = None

    stim_conf = load_config(os.path.join(input_dir, 'stim_conf.pkl'))

    return stimuli, target_pos, stimulus_files, target_pos_files, stim_conf


def load_response(input_dir):
    assert os.path.isdir(input_dir), input_dir
    print("Load responses in:", input_dir)

    responses = load_var(os.path.join(input_dir, 'responses.pkl'))
    model_conf = load_config(os.path.join(input_dir, 'model_conf.pkl'))
    encoder_conf = load_config(os.path.join(input_dir, 'encoder_conf.pkl'))

    return responses, model_conf, encoder_conf


def load_dataset(stim_dir, response_subdir, load_raw_stimuli=True, load_target_pos=True):
    assert os.path.isdir(stim_dir), stim_dir

    stimuli, target_pos, stimulus_files, target_pos_files, stim_conf = load_stimuli(
        stim_dir, load_raw_stimuli=load_raw_stimuli, load_target_pos=load_target_pos)

    responses, model_conf, encoder_conf = load_response(os.path.join(stim_dir, response_subdir))

    stimulus_files_valid = [r['stimulus_file'] for r in responses]
    stimulus_files_valid_idxs = [i for i, f in enumerate(stimulus_files)
                                 if f in stimulus_files_valid]

    if load_raw_stimuli:
        stimuli = [stimuli[i] for i in stimulus_files_valid_idxs]
    if load_target_pos:
        target_pos = [target_pos[i] for i in stimulus_files_valid_idxs]

    return stimuli, target_pos, responses, stim_conf, model_conf, encoder_conf


def load_dataset_decoders(input_dir):
    assert os.path.isdir(input_dir), input_dir

    decoder_files = [f for f in os.listdir(input_dir) if f.startswith('d_') and f.endswith('.pkl')]
    print(decoder_files)

    decoder_dicts = dict()

    for decoder_file in decoder_files:
        decoder_dict = load_var(os.path.join(input_dir, decoder_file))
        decoder_dicts[decoder_dict.pop('kind')] = decoder_dict

    decoder_conf = load_var(os.path.join(input_dir, 'decoder_conf.pkl'))

    return decoder_dicts, decoder_conf


def load_dataset_filters(input_dir):
    assert os.path.isdir(input_dir), input_dir

    filter_files = [f for f in os.listdir(input_dir) if f.startswith('f') and f.endswith('.pkl')]
    print(filter_files)

    filter_dicts = dict()

    for filter_file in filter_files:
        filter_dict = load_var(os.path.join(input_dir, filter_file))
        filter_dicts[filter_dict.pop('kind')] = filter_dict

    return filter_dicts


def get_path_for_artificial_stim_conf_file(stim_conf_file):
    from alphacnn.paths import CONF_A_STIM_PATH, DATASET_PATH

    stim_conf_path = os.path.join(CONF_A_STIM_PATH, stim_conf_file)
    assert os.path.isfile(stim_conf_path), stim_conf_path

    stim_dir = get_stim_dir(load_config(config_file=stim_conf_path))
    print("Load stimuli in:", stim_dir)
    stim_path = os.path.join(DATASET_PATH, 'artificial_stim', stim_dir)

    return stim_path


def get_path_for_real_stim_conf_file(stim_conf_file):
    from alphacnn.paths import CONF_STIM_PATH, DATASET_PATH

    stim_conf_path = os.path.join(CONF_STIM_PATH, stim_conf_file)
    assert os.path.isfile(stim_conf_path), stim_conf_path
    stim_dir = get_stim_dir(load_config(config_file=stim_conf_path))
    print("Load stimuli in:", stim_dir)
    stim_path = os.path.join(DATASET_PATH, 'real_stim', stim_dir)

    return stim_path
