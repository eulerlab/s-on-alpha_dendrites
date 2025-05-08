import os

import numpy as np
import pandas as pd

from alphacnn import paths


def load_raw_data_set(src_dir, data_set_file):
    """Load data set from pickle file and check consistency."""
    data_set = pd.read_pickle(os.path.join(paths.DATASET_PATH, src_dir, data_set_file)).reset_index()

    if data_set.stimulus_config_id.nunique() > 1:
        data_set.stimulus_id = data_set.stimulus_id.astype(str) + data_set.stimulus_config_id.astype(str)

    n_videos = data_set.stimulus_id.nunique()
    n_samples_per_video = data_set.bc_noise_sample.nunique()

    assert data_set.video_width.nunique() == 1 and data_set.video_height.nunique() == 1, \
        "All videos in data set must have the same dimensions."

    video_width = data_set.video_width.unique()[0]
    video_height = data_set.video_height.unique()[0]

    assert data_set.pixel_size_um.nunique() == 1, \
        "All videos in data set must have the same pixel size."

    pixel_size_um = data_set.pixel_size_um.unique()[0]

    assert n_videos * n_samples_per_video == data_set.shape[0], \
        "Number of videos times number of samples per video must equal number of samples in data set."

    return data_set, n_videos, n_samples_per_video, video_width, video_height, pixel_size_um


def flatten_raw_data_set(data_set, cricket_only=False):
    """Reformat dataset to be used for training."""
    # Reformat
    data_set['target_pos_x'] = data_set['target_pos'].apply(lambda row: row['xy'][:, 0])
    data_set['target_pos_y'] = data_set['target_pos'].apply(lambda row: row['xy'][:, 1])
    data_set['target_pos_d'] = data_set['target_pos'].apply(lambda row: row['d'])

    # Explode
    video_cols = ['response', 'response_wo', 'target_pos_x', 'target_pos_y', 'target_pos_d']
    data_set['n_frames'] = data_set.apply(lambda row: np.min([row[col].shape[0] for col in video_cols]), axis=1)
    for col in video_cols:
        data_set[col] = data_set.apply(lambda row: row[col][:row['n_frames']], axis=1)
    data_set = data_set.explode(video_cols)

    # New index
    data_set.reset_index(inplace=True)

    # Add trial id
    data_set['trial_id'] = data_set['stimulus_id'].apply(
        lambda x: x.replace('_left.mp4', '').replace('_right.mp4', '')).astype(str)

    # Drop
    data_set.drop(['index', 'level_0', 'target_pos', 'n_frames'], axis=1, inplace=True)
    data_set.drop(['wo_cricket_1', 'wo_cricket_2'], axis=1, inplace=True)
    data_set.drop(['bc_srf_config_id_1', 'bc_rect_config_id_1', 'bc_srf_config_id_2', 'bc_rect_config_id_2'], axis=1,
                  inplace=True)
    data_set.drop(['bc_noise_id', 'pr_noise_id'], axis=1, inplace=True)
    data_set.drop(['stimulus_config_id'], axis=1, inplace=True)

    data_set = data_set.infer_objects()

    if cricket_only:
        data_set = data_set[data_set.target_pos_x.notnull() & data_set.target_pos_d.notnull()].copy()
        data_set = data_set.reset_index()

    return data_set


def get_datasets_tab(data_set_file, split_id, restrictions=None):
    """Query to get full data-set tab and splits."""
    from alphacnn.database.dataset_schema import DataNorm, DataSplit, DataSet

    if restrictions is None:
        restrictions = dict()
    dataset_tab = (DataNorm * DataNorm.NormPoint * DataSplit.SplitPoint *
                   DataSet.DataPoint().proj('x_len') * DataSet().proj('n_samples_per_video')
                   & dict(data_set_file=data_set_file, split_id=split_id)
                   & restrictions)
    dataset_train_tab = dataset_tab & "split_kind='train'"
    dataset_dev_tab = dataset_tab & "split_kind='dev'"
    dataset_test_tab = dataset_tab & "split_kind='test'"

    return dataset_tab, dataset_train_tab, dataset_dev_tab, dataset_test_tab


def y_to_p(y):
    p = np.all(np.isfinite(y), axis=1)
    return p


def get_split_indexes(n_tot: int, f_train: float, seed=None):
    assert n_tot >= 1
    assert 0 < f_train <= 1.
    f_test = 1. - f_train

    if f_test > 0 and n_tot == 1:
        raise ValueError(n_tot)

    idxs = np.arange(n_tot)
    np.random.seed(seed)
    np.random.shuffle(idxs)

    n_test = int(np.round(n_tot * f_train))
    idxs_train, idxs_test = idxs[:n_test], idxs[n_test:]

    if idxs_train.size == 0:
        idxs_train = idxs_test[:1]
        idxs_test = idxs_test[1:]

    if idxs_test.size == 0:
        idxs_test = idxs_train[:1]
        idxs_train = idxs_train[1:]

    return idxs_train, idxs_test


def get_split_indexes_equal_lengths(stimulus_ids, stimulus_lengths, f_train=0.7, f_dev=0.1, seed=42):
    """Split stimulus ids into train and test set, such that the total length of the train set is f_train of the total
    length of all stimuli."""

    f_test = 1. - f_train - f_dev

    stimulus_length_tot = np.sum(stimulus_lengths)

    stimulus_length_train_min = int(stimulus_length_tot * f_train)
    stimulus_length_train = 0
    stimulus_ids_train = []

    stimulus_length_dev_min = int(stimulus_length_tot * f_dev)
    stimulus_length_dev = 0
    stimulus_ids_dev = []

    stimulus_length_test = 0
    stimulus_ids_test = []

    # Get stimulus pairs
    stimulus_ids_pairs = []
    stimulus_length_pairs = []

    for stimulus_id in stimulus_ids:
        if 'left' in stimulus_id:
            stimulus_id_other = stimulus_id.replace('left', 'right')
        elif 'right' in stimulus_id:
            stimulus_id_other = stimulus_id.replace('right', 'left')
        else:
            stimulus_id_other = None

        if 'left' in stimulus_id and np.any(stimulus_ids == stimulus_id_other):
            continue

        if (stimulus_id_other is not None) and np.any(stimulus_ids == stimulus_id_other):
            stimulus_ids_pairs.append((stimulus_id, stimulus_id_other))
            stimulus_length_pairs.append(
                stimulus_lengths[stimulus_ids == stimulus_id][0] +
                stimulus_lengths[stimulus_ids == stimulus_id_other][0])
        else:
            stimulus_ids_pairs.append((stimulus_id,))
            stimulus_length_pairs.append(stimulus_lengths[stimulus_ids == stimulus_id][0])

    stimulus_ids_pairs = np.array(stimulus_ids_pairs, dtype=object)
    stimulus_length_pairs = np.array(stimulus_length_pairs)

    # Shuffle stimulus ids
    np.random.seed(seed)
    random_idxs = np.arange(stimulus_ids_pairs.size)
    np.random.shuffle(random_idxs)

    shuffled_ids = stimulus_ids_pairs[random_idxs]
    shuffled_lengths = stimulus_length_pairs[random_idxs]

    i = 0
    if f_test > 0:
        # Make sure to add one sample to test
        stimulus_ids_test += list(shuffled_ids[i])
        stimulus_length_test += shuffled_lengths[i]
        i += 1

    if f_dev > 0:
        # Make sure to add one sample to dev
        stimulus_ids_dev += list(shuffled_ids[i])
        stimulus_length_dev += shuffled_lengths[i]
        i += 1

    # Add to train set until full
    while stimulus_length_train < stimulus_length_train_min:
        stimulus_ids_train += list(shuffled_ids[i])
        stimulus_length_train += shuffled_lengths[i]
        i += 1

    # Add to dev set until full
    while stimulus_length_dev < stimulus_length_dev_min:
        stimulus_ids_dev += list(shuffled_ids[i])
        stimulus_length_dev += shuffled_lengths[i]
        i += 1

    # Add remaining to test set
    while i < shuffled_ids.size:
        stimulus_ids_test += list(shuffled_ids[i])
        stimulus_length_test += shuffled_lengths[i]
        i += 1

    assert len(((set(stimulus_ids) - set(stimulus_ids_test)) - set(stimulus_ids_dev)) - set(stimulus_ids_train)) == 0
    assert len(set(stimulus_ids_train).intersection(set(stimulus_ids_test))) == 0, \
        set(stimulus_ids_train).intersection(set(stimulus_ids_test))
    assert len(set(stimulus_ids_train).intersection(set(stimulus_ids_dev))) == 0, \
        set(stimulus_ids_train).intersection(set(stimulus_ids_dev))

    return stimulus_ids_train, stimulus_ids_dev, stimulus_ids_test


def get_dataset_dict(data_set_file, split_id, shift_target=0, restrictions=None, flat_x=True, return_keys=False,
                     augment_train=True, augment_dev=False, augment_test=False,
                     get_wo=False, merge_wo=True, add_p=True, cricket_only=False):
    """Create a dictionary with all data from a data set."""

    if restrictions is None:
        restrictions = dict()

    dataset_tab, dataset_train_tab, dataset_dev_tab, dataset_test_tab = get_datasets_tab(
        data_set_file, split_id, restrictions)

    n_samples_per_video = dataset_tab.fetch('n_samples_per_video')
    assert np.unique(n_samples_per_video).size == 1
    n_samples_per_video = n_samples_per_video[0]

    assert len(dataset_train_tab.proj() & dataset_test_tab.proj()) == 0
    y_train, d_train, x_train, x_train_wo, x_train_lens = tab_to_data_stacks(
        dataset_train_tab, shift_target=shift_target, flat_x=False, get_wo=get_wo, cricket_only=cricket_only)
    y_dev, d_dev, x_dev, x_dev_wo, x_dev_lens = tab_to_data_stacks(
        dataset_dev_tab, shift_target=shift_target, flat_x=False, get_wo=get_wo, cricket_only=cricket_only)
    y_test, d_test, x_test, x_test_wo, x_test_lens = tab_to_data_stacks(
        dataset_test_tab, shift_target=shift_target, flat_x=False, get_wo=get_wo, cricket_only=cricket_only)

    if augment_train:
        x_train, y_train, x_train_lens, d_train = augment_data(x_train, y_train, x_train_lens, d_train)
        if x_train_wo is not None:
            x_train_wo = augment_data(x_train_wo)[0]

    if augment_dev:
        x_dev, y_dev, x_dev_lens, d_dev = augment_data(x_dev, y_dev, x_dev_lens, d_dev)
        if x_dev_wo is not None:
            x_dev_wo = augment_data(x_dev_wo)[0]

    if augment_test:
        x_test, y_test, x_test_lens, d_test = augment_data(x_test, y_test, x_test_lens, d_test)
        if x_test_wo is not None:
            x_test_wo = augment_data(x_test_wo)[0]

    if flat_x:
        x_train = x_train.reshape(x_train.shape[0], -1)
        x_dev = x_dev.reshape(x_dev.shape[0], -1)
        x_test = x_test.reshape(x_test.shape[0], -1)

        if x_train_wo is not None:
            x_train_wo = x_train_wo.reshape(x_train_wo.shape[0], -1)
            x_dev_wo = x_dev_wo.reshape(x_dev_wo.shape[0], -1)
            x_test_wo = x_test_wo.reshape(x_test_wo.shape[0], -1)

    dat = dict()
    dat['n_samples_per_video'] = n_samples_per_video

    if x_train_wo is not None:
        if merge_wo:
            x_train = np.concatenate([x_train, x_train_wo], axis=0)
            x_train_lens = np.tile(x_train_lens, 2)
            y_train = np.concatenate([y_train, np.full_like(y_train, np.nan)], axis=0)
            d_train = np.concatenate([d_train, np.full_like(d_train, np.nan)], axis=0)

            x_dev = np.concatenate([x_dev, x_dev_wo], axis=0)
            x_dev_lens = np.tile(x_dev_lens, 2)
            y_dev = np.concatenate([y_dev, np.full_like(y_dev, np.nan)], axis=0)
            d_dev = np.concatenate([d_dev, np.full_like(d_dev, np.nan)], axis=0)

            x_test = np.concatenate([x_test, x_test_wo], axis=0)
            x_test_lens = np.tile(x_test_lens, 2)
            y_test = np.concatenate([y_test, np.full_like(y_test, np.nan)], axis=0)
            d_test = np.concatenate([d_test, np.full_like(d_test, np.nan)], axis=0)

        else:
            dat['x_train_wo'] = x_train_wo
            dat['x_dev_wo'] = x_dev_wo
            dat['x_test_wo'] = x_test_wo

    dat['y_train'] = y_train
    dat['d_train'] = d_train
    dat['x_train'] = x_train
    dat['x_train_lens'] = x_train_lens
    dat['y_dev'] = y_dev
    dat['d_dev'] = d_dev
    dat['x_dev'] = x_dev
    dat['x_dev_lens'] = x_dev_lens
    dat['y_test'] = y_test
    dat['d_test'] = d_test
    dat['x_test'] = x_test
    dat['x_test_lens'] = x_test_lens

    if return_keys:
        dat['keys_train'] = np.asarray(dataset_train_tab.proj().fetch(as_dict=True))
        dat['keys_dev'] = np.asarray(dataset_dev_tab.proj().fetch(as_dict=True))
        dat['keys_test'] = np.asarray(dataset_test_tab.proj().fetch(as_dict=True))

    if add_p:
        for kind in ['train', 'dev', 'test']:
            dat[f'p_{kind}'] = np.all(np.isfinite(dat[f'y_{kind}']), axis=1)

    if cricket_only:
        for kind in ['train', 'dev', 'test']:
            # Correct x_lens by considering only frames with presence
            x_lens = dat[f'x_{kind}_lens']
            p = dat[f'p_{kind}'] if add_p else np.all(np.isfinite(dat[f'y_{kind}']), axis=1)

            assert len(x_lens) % len(dat[f'keys_{kind}']) == 0,\
                f"len(x_lens)={len(x_lens)} is not a multiple of len(keys)={len(dat[f'keys_{kind}'])}"

            x_lens_corrected = np.zeros(len(x_lens), dtype=int)
            indexes = np.cumsum(np.append(0, x_lens))
            for ki, (i1, i2) in enumerate(zip(indexes[:-1], indexes[1:])):
                x_lens_corrected[ki] = np.sum(p[i1:i2])
            dat[f'x_{kind}_lens_corrected'] = x_lens_corrected

    return dat


def tab_to_data_stacks(dataset_tab, shift_target=0, flat_x=True, get_wo=False, cricket_only=False):
    """Create numpy arrays from a data-set table."""

    if get_wo:
        y, d, x, x_wo, x_len = dataset_tab.fetch('y_norm', 'd_norm', 'x_norm', 'x_norm_wo', 'x_len')
        if x_wo[0] is None:
            x_wo = None
    else:
        y, d, x, x_len = dataset_tab.fetch('y_norm', 'd_norm', 'x_norm', 'x_len')
        x_wo = None

    # Correct for shifted target (if any)
    y = np.vstack([y_i[shift_target:shift_target + x_i.shape[0]] for y_i, x_i in zip(y, x)])
    d = np.concatenate([d_i[shift_target:shift_target + x_i.shape[0]] for d_i, x_i in zip(d, x)])

    x = np.vstack(x)
    if x_wo is not None:
        x_wo = np.vstack(x_wo)
    else:
        x_wo = None

    if flat_x:
        x = x.reshape(x.shape[0], -1)
        if x_wo is not None:
            x_wo = x_wo.reshape(x_wo.shape[0], -1)

    assert x.shape[0] == y.shape[0]
    assert x.shape[0] == d.shape[0]

    if x_wo is not None:
        assert x.shape == x_wo.shape

    if cricket_only:
        w_cricket = np.all(np.isfinite(y), axis=1) & np.isfinite(d)
        y = y[w_cricket]
        d = d[w_cricket]
        x = x[w_cricket]
        if x_wo is not None:
            x_wo = x_wo[w_cricket]
        video_idxs = np.cumsum(np.append(0, x_len))
        x_len = [np.sum(w_cricket[i1:i2]) for i1, i2 in zip(video_idxs[:-1], video_idxs[1:])]

    return y, d, x, x_wo, x_len


def augment_data(x, y=None, x_lens=None, d=None):
    if y is not None:
        p = np.all(np.isfinite(y), axis=1)
        assert np.all(y[p] >= -1) and np.all(y[p] <= 1)

    x_aug = np.concatenate([
        x,
        np.flip(x, axis=1),
        np.flip(x, axis=2),
        np.flip(np.flip(x, axis=1), axis=2),
    ], axis=0)

    if y is None:
        y_aug = None
    else:
        y_aug = np.concatenate([
            y,
            y * np.array([1., -1.]),
            y * np.array([-1., 1.]),
            y * np.array([-1, -1]),
        ], axis=0)

    d_aug = np.tile(d, 4) if d is not None else None
    x_lens_aug = np.tile(x_lens, 4) if x_lens is not None else None

    return x_aug, y_aug, x_lens_aug, d_aug


def combine_w_and_wo(x, x_wo, return_shuffle_idx=False, pairwise=False, seed=42):
    n_frames = x.shape[0]
    x_w_and_wo = np.full(x.shape + (2,), np.nan)
    g_w_and_wo = np.full(n_frames, np.nan)

    if not pairwise:
        np.random.seed(seed)
        x_wo = x_wo.copy()
        if not return_shuffle_idx:
            np.random.shuffle(x_wo)  # Shuffles along first dimension
        else:
            shuffle_idx = np.arange(x_wo.shape[0])
            np.random.shuffle(shuffle_idx)
            x_wo = x_wo[shuffle_idx]

    rnd_gs = np.random.choice((0, 1), n_frames)

    for i, g in enumerate(rnd_gs):
        x_w_and_wo[i, :, :, g] = x[i]
        x_w_and_wo[i, :, :, abs(1 - g)] = x_wo[i]
        g_w_and_wo[i] = g

    if not return_shuffle_idx:
        return x_w_and_wo, g_w_and_wo
    else:
        return x_w_and_wo, g_w_and_wo, shuffle_idx


def combine_w_and_wo_flat(x, x_wo, seed=42):
    assert x.shape == x_wo.shape, f"x.shape={x.shape} and x_wo.shape={x_wo.shape} must be equal"

    n_frames = x.shape[0]
    x_w_and_wo = np.vstack([x, x_wo])
    p_w_and_wo = np.concatenate([np.ones(n_frames, dtype=bool), np.zeros(n_frames, dtype=bool)])

    np.random.seed(seed)
    shuffle_idx = np.arange(2*n_frames)
    np.random.shuffle(shuffle_idx)
    x_w_and_wo = x_w_and_wo[shuffle_idx]
    p_w_and_wo = p_w_and_wo[shuffle_idx]

    return x_w_and_wo, p_w_and_wo, shuffle_idx


def flatten_datapoints(df, inplace=True):
    df = df.copy() if not inplace else df

    df.drop(['y_full', 'x', 'x_wo'], inplace=True, axis=1)
    df.reset_index(inplace=True)

    df['trial_id'] = df['stimulus_id'].apply(
        lambda x: x.replace('_left.mp4', '').replace('_right.mp4', '')).astype(str)
    df['frame_idx'] = df['x_len'].apply(lambda x: np.arange(x))

    df = df.explode(['frame_idx', 'd_full', 'p_full'])

    df = df[df.p_full & df.d_full.notnull()]
    df.rename(columns={'d_full': 'd', 'p_full': 'p'}, inplace=True)
    df.reset_index(inplace=True)

    df = df.infer_objects()
    return df


def split_flat_datapoints(df, n_train=7, n_dev=1, n_test=2, col_dist='d', q_dist=10, col_groups='trial_id', seed=42):
    from sklearn.model_selection import StratifiedGroupKFold

    idxs = np.arange(df.shape[0])
    dist_ids = pd.qcut(df[col_dist], q=q_dist).cat.codes
    groups = pd.Categorical(df[col_groups]).codes

    test_ratio = ((n_train + n_dev) / n_test)
    assert int(test_ratio) == test_ratio
    assert test_ratio > 0

    idxs_test = next(StratifiedGroupKFold(
        int(test_ratio), random_state=seed, shuffle=True).split(X=idxs, y=dist_ids, groups=groups))[1]

    idxs_train_dev = np.array(list(set(idxs) - set(idxs_test)))

    dev_ratio = n_train / n_dev
    assert int(dev_ratio) == dev_ratio
    assert dev_ratio > 0

    idxs_dev = idxs_train_dev[next(StratifiedGroupKFold(int(dev_ratio), random_state=seed, shuffle=True).split(
        X=idxs_train_dev, y=dist_ids[idxs_train_dev], groups=groups[idxs_train_dev]))[1]]
    idxs_train = np.array(list(set(idxs_train_dev) - set(idxs_dev)))

    assert len(set(idxs_train_dev).intersection(set(idxs_test))) == 0
    assert len(set(idxs_train).intersection(set(idxs_test))) == 0
    assert len(set(idxs_train).intersection(set(idxs_dev))) == 0
    assert len(set(idxs_dev).intersection(set(idxs_test))) == 0
    assert set(idxs_train).union(set(idxs_dev)).union(set(idxs_test)) == set(idxs)

    return idxs_train, idxs_dev, idxs_test
