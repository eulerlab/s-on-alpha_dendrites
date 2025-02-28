import os
import pickle
import warnings
from abc import abstractmethod

import datajoint as dj
import numpy as np
from djimaging.tables.misc.highresolution import load_high_res_stack

from djimaging.autorois.roi_canvas import InteractiveRoiCanvas

from djimaging.utils import scanm_utils
from djimaging.utils.datafile_utils import as_pre_filepath
from djimaging.utils.dj_utils import get_primary_key, check_unique_one
from djimaging.utils.mask_utils import to_igor_format, to_python_format, to_roi_mask_file, sort_roi_mask_files, \
    load_preferred_roi_mask_igor, load_preferred_roi_mask_pickle, compare_roi_masks
from djimaging.utils.plot_utils import plot_field


def load_stack_data(files, data_name, alt_name, from_raw_data,
                    roi_mask_dir=None, old_prefix=None, new_prefix=None):
    ch0_stacks, ch1_stacks, output_files = [], [], []

    for data_file in files:
        ch_stacks, wparams = scanm_utils.load_stacks(
            data_file, ch_names=(data_name, alt_name), from_raw_data=from_raw_data)
        ch0_stacks.append(ch_stacks[data_name])
        ch1_stacks.append(ch_stacks[alt_name])
        output_files.append(to_roi_mask_file(
            data_file, roi_mask_dir=roi_mask_dir, old_prefix=old_prefix, new_prefix=new_prefix))

    return ch0_stacks, ch1_stacks, output_files


class RoiMaskTemplate(dj.Manual):
    database = ""
    _max_shift = 5

    @property
    def definition(self):
        definition = """
        # ROI mask
        -> self.field_table
        -> self.raw_params_table
        ---
        -> self.presentation_table
        roi_mask     : blob                   # ROI mask for recording field
        """
        return definition

    class RoiMaskPresentation(dj.Part):
        @property
        def definition(self):
            definition = """
            # ROI Mask
            -> master
            -> self.presentation_table
            ---
            roi_mask      : blob       # ROI mask for presentation field
            as_field_mask : enum("same", "different", "shifted")  # relationship to field mask
            shift_dx=0    : int  # Shift in x
            shift_dy=0    : int  # Shift in y
            """
            return definition

        @property
        @abstractmethod
        def presentation_table(self):
            pass

    @property
    @abstractmethod
    def presentation_table(self):
        pass

    @property
    @abstractmethod
    def field_table(self):
        pass

    @property
    @abstractmethod
    def experiment_table(self):
        pass

    @property
    @abstractmethod
    def userinfo_table(self):
        pass

    @property
    @abstractmethod
    def raw_params_table(self):
        pass

    @property
    def key_source(self):
        try:
            return self.field_table.proj() * self.raw_params_table().proj()
        except (AttributeError, TypeError):
            pass

    def list_missing_field(self):
        missing_keys = (self.field_table.proj() & (self.presentation_table.proj() - self.proj())).fetch(as_dict=True)
        return missing_keys

    def draw_roi_mask(self, field_key=None, pres_key=None, canvas_width=20, autorois_models='default_rgc',
                      show_diagnostics=True, load_high_res=True,
                      roi_mask_dir=None, old_prefix=None, new_prefix=None, use_stim_onset=True, **kwargs):
        if canvas_width <= 0 or canvas_width >= 100:
            raise ValueError(f'canvas_width={canvas_width} must be in (0, 100)%')

        if pres_key is not None:
            field_key = (self.field_table & pres_key).proj().fetch1()
        elif field_key is None:
            field_key = np.random.choice(self.list_missing_field())

        assert field_key is not None, 'No field_key provided and no missing field found.'

        pres_keys = np.array(list((self.presentation_table & field_key).proj()),
                             dtype='object')

        from_raw_data = (self.raw_params_table & field_key).fetch1("from_raw_data")

        n_artifact, pixel_size_um, scan_type = (self.presentation_table() & pres_keys).fetch(
            'npixartifact', 'pixel_size_um', 'scan_type')
        n_artifact = check_unique_one(n_artifact, name='n_artifact')
        pixel_size_um = check_unique_one(pixel_size_um, name='pixel_size_um')
        scan_type = check_unique_one(scan_type, name='scan_type')

        if scan_type == 'xy':
            pixel_size_d1_d2 = (pixel_size_um, pixel_size_um)
        elif scan_type == 'xz':
            z_step_um = (self.presentation_table() & pres_keys).fetch('z_step_um')
            z_step_um = check_unique_one(z_step_um, name='z_step_um')
            pixel_size_d1_d2 = (pixel_size_um, z_step_um)
        else:
            raise NotImplementedError(scan_type)

        # Get pres data
        mask_alias, highres_alias = (self.userinfo_table() & field_key).fetch1(
            "mask_alias", "highres_alias")
        files, stim_names, conditions, triggertimes = (self.presentation_table & pres_keys).fetch(
            'pres_data_file', 'stim_name', 'condition', 'triggertimes')
        assert len(pres_keys) == len(files)
        scan_frequencies = (self.presentation_table.ScanInfo() & pres_keys).fetch('scan_frequency')
        assert len(pres_keys) == len(scan_frequencies)

        # Sort data by relevance
        sort_idxs = sort_roi_mask_files(files, mask_alias=mask_alias, highres_alias=highres_alias, as_index=True)
        pres_keys = pres_keys[sort_idxs]
        files = files[sort_idxs]
        stim_names = stim_names[sort_idxs]
        conditions = conditions[sort_idxs]
        triggertimes = triggertimes[sort_idxs]
        scan_frequencies = scan_frequencies[sort_idxs]

        # Load stack data
        data_name, alt_name = (self.userinfo_table & field_key).fetch1('data_stack_name', 'alt_stack_name')
        ch0_stacks, ch1_stacks, output_files = load_stack_data(
            files, data_name=data_name, alt_name=alt_name, from_raw_data=from_raw_data,
            roi_mask_dir=roi_mask_dir, old_prefix=old_prefix, new_prefix=new_prefix)

        if use_stim_onset:
            stim_onset_idxs = [int(np.floor(tt[0] * fs)) if len(tt) > 0 else 0
                               for tt, fs in zip(triggertimes, scan_frequencies)]
            ch0_stacks = [stack[:, :, stim_onset_idx:] for stack, stim_onset_idx in zip(ch0_stacks, stim_onset_idxs)]
            ch1_stacks = [stack[:, :, stim_onset_idx:] for stack, stim_onset_idx in zip(ch1_stacks, stim_onset_idxs)]

        # Load high resolution data is possible
        high_res_bg_dict = self.load_high_res_bg_dict(field_key) if load_high_res else dict()

        # Load initial ROI masks
        igor_roi_masks = (self.raw_params_table & field_key).fetch1('igor_roi_masks')
        initial_roi_mask, _ = self.load_initial_roi_mask(
            field_key=field_key, igor_roi_masks=igor_roi_masks,
            roi_mask_dir=roi_mask_dir, old_prefix=old_prefix, new_prefix=new_prefix)

        if initial_roi_mask is not None:
            initial_roi_mask = to_python_format(initial_roi_mask)

        # Load default AutoROIs models
        if isinstance(autorois_models, str):
            autorois_models = load_default_autorois_models(autorois_models)

        # Load shifts if available
        shifts = self.init_shifts(field_key, pres_keys)

        max_shift = self._max_shift if kwargs.get('max_shift', None) is None else kwargs.pop('max_shift')

        roi_canvas = InteractiveRoiCanvas(
            stim_names=[f"{stim_name}({condition})" for stim_name, condition in zip(stim_names, conditions)],
            ch0_stacks=ch0_stacks, ch1_stacks=ch1_stacks, n_artifact=n_artifact, bg_dict=high_res_bg_dict,
            main_stim_idx=0, initial_roi_mask=initial_roi_mask, shifts=shifts,
            canvas_width=canvas_width, autorois_models=autorois_models, output_files=output_files,
            pixel_size_um=pixel_size_d1_d2, show_diagnostics=show_diagnostics, max_shift=max_shift,
            **kwargs,
        )
        print(f"Returned InteractiveRoiCanvas object. To start GUI, call <enter_object_name>.start_gui().")
        return roi_canvas

    def init_shifts(self, field_key, pres_keys):
        if len(self & field_key) == 0:
            return None
        shifts = []
        for pres_key in pres_keys:
            if len(self.RoiMaskPresentation & pres_key) > 0:
                as_field_mask, shift_dx, shift_dy = \
                    (self.RoiMaskPresentation & pres_key).fetch1("as_field_mask", "shift_dx", "shift_dy")

                if as_field_mask == "same":
                    assert shift_dx == 0 and shift_dy == 0, 'Shifts are non-zero but should be'
                    shifts.append((0, 0))
                elif as_field_mask == "shifted":
                    shifts.append((shift_dx, shift_dy))
                elif as_field_mask == "different":
                    warnings.warn(f"""
                    Inconsistent ROI mask for field_key=\n{field_key}\nand pres_key=\n{pres_key}\n.
                    This is not supported; Presentation will be initialized with Field ROI mask instead.
                    If you really want to define different (i.e. not simply shifted) ROI masks for a field
                    open the GUI for a single Presentation key and save the ROI mask.
                    """)
                    shifts.append((0, 0))
                else:
                    warnings.warn(f"as_field_mask=\n{as_field_mask}\n is not a valid value. Default to zero shift.")
                    shifts.append((0, 0))
            else:
                warnings.warn(f"Pres_key=\n{pres_key}\n was not in RoiMask, but at least one other field_key was. " +
                              f"Default to zero shift.")
                shifts.append((0, 0))
        return shifts

    def load_initial_roi_mask(self, field_key, igor_roi_masks=str, roi_mask_dir=None, old_prefix=None, new_prefix=None):
        """
        Load initial ROI mask for field.
        First try to load from database.
        Second try pickle file, unless ROI masks should be exclusively loaded from Igor.
        Last try to load from Igor file, unless ROI masks should never be loaded from Igor.
        """

        roi_mask = self.load_field_roi_mask_database(field_key=field_key)

        if roi_mask is not None:
            return roi_mask, 'database'

        if igor_roi_masks != 'yes':
            roi_mask, src_file = self.load_field_roi_mask_pickle(
                field_key=field_key, roi_mask_dir=roi_mask_dir, old_prefix=old_prefix, new_prefix=new_prefix)
            if roi_mask is not None:
                return roi_mask, src_file

        if igor_roi_masks != 'no':
            roi_mask, src_file = self.load_field_roi_mask_igor(field_key=field_key)
            if roi_mask is not None:
                return roi_mask, src_file

        return None, 'none'

    def load_field_roi_mask_database(self, field_key):
        """Load ROI mask that was generated in DataJoint GUI"""
        database_roi_masks = (self & field_key).fetch("roi_mask")

        if len(database_roi_masks) == 1:
            database_roi_mask = database_roi_masks[0].copy()
        elif len(database_roi_masks) == 0:
            database_roi_mask = None
        else:
            raise ValueError(f'Found multiple ROI masks for key=\n{field_key}')

        return database_roi_mask

    def load_field_roi_mask_pickle(self, field_key, roi_mask_dir=None, old_prefix=None, new_prefix=None) -> (
            np.ndarray, str):
        mask_alias, highres_alias = (self.userinfo_table() & field_key).fetch1("mask_alias", "highres_alias")
        files = (self.presentation_table() & field_key).fetch("pres_data_file")

        roi_mask, src_file = load_preferred_roi_mask_pickle(
            files, mask_alias=mask_alias, highres_alias=highres_alias,
            roi_mask_dir=roi_mask_dir, old_prefix=old_prefix, new_prefix=new_prefix)
        if roi_mask is not None:
            print(f'Loaded ROI mask from file={src_file} for files=\n{files}\nfor mask_alias={mask_alias}')

        return roi_mask, src_file

    def load_field_roi_mask_igor(self, field_key) -> (np.ndarray, str):
        mask_alias, highres_alias, raw_data_dir, pre_data_dir = (self.userinfo_table() & field_key).fetch1(
            "mask_alias", "highres_alias", "raw_data_dir", "pre_data_dir")
        files = (self.presentation_table() & field_key).fetch("pres_data_file")

        files = [as_pre_filepath(f, raw_data_dir=raw_data_dir, pre_data_dir=pre_data_dir) for f in files]

        roi_mask, src_file = load_preferred_roi_mask_igor(files, mask_alias=mask_alias, highres_alias=highres_alias)
        if roi_mask is not None:
            print(f'Loaded ROI mask from file={src_file} for files=\n{files}\nfor mask_alias={mask_alias}')

        return roi_mask, src_file

    def rescan_filesystem(self, restrictions: dict = None, verboselvl: int = 0, suppress_errors: bool = False,
                          only_new_fields: bool = True, roi_mask_dir=None, old_prefix=None, new_prefix=None):
        """Scan filesystem for new ROI masks and add them to the database."""
        if restrictions is None:
            restrictions = dict()

        if only_new_fields:
            restrictions = (self.key_source - self) & restrictions

        err_list = []

        for key in (self.key_source & restrictions):
            try:
                self.add_field_roi_masks(
                    key, roi_mask_dir=roi_mask_dir, old_prefix=old_prefix, new_prefix=new_prefix,
                    verboselvl=verboselvl)
            except Exception as e:
                if suppress_errors:
                    warnings.warn(f'Error for key={key}:\n{e}')
                    err_list.append((key, e))
                else:
                    raise e

        return err_list

    def add_field_roi_masks(self, field_key, roi_mask_dir=None, old_prefix=None, new_prefix=None, verboselvl=0):
        if verboselvl > 2:
            print('\nfield_key:', field_key)

        pres_keys = (self.presentation_table.proj() & field_key)
        roi_masks = [self.load_presentation_roi_mask(key, roi_mask_dir, old_prefix=old_prefix, new_prefix=new_prefix)
                     for key in pres_keys]

        data_pairs = zip(pres_keys, roi_masks)

        # Filter out keys without ROI mask
        data_pairs = [(pres_key, roi_mask) for pres_key, roi_mask in data_pairs
                      if roi_mask is not None]

        if len(data_pairs) == 0:
            if verboselvl > 1:
                print('No ROI masks found for field:', field_key)
            if verboselvl > 2:
                print('pres_keys:', [k for k in pres_keys])
            return

        # Filter out keys that are already present
        data_pairs = [(pres_key, roi_mask) for pres_key, roi_mask in data_pairs
                      if pres_key not in self.RoiMaskPresentation().proj()]

        if len(data_pairs) == 0:
            if verboselvl > 1:
                print('Nothing new to add for field:', field_key)
            return

        if verboselvl > 0:
            print(f'Adding {len(data_pairs)} ROI masks for field:', field_key)

        # Find preferred file that should be used as main key.
        mask_alias, highres_alias = (self.userinfo_table & field_key).fetch1("mask_alias", "highres_alias")
        files = [(self.presentation_table & pres_key).fetch1('pres_data_file') for pres_key, roi_mask in data_pairs]
        sort_idxs = sort_roi_mask_files(files, mask_alias=mask_alias, highres_alias=highres_alias, as_index=True)

        main_pres_key, main_roi_mask = data_pairs[sort_idxs[0]]

        self.insert1({**field_key, **main_pres_key, "roi_mask": main_roi_mask}, skip_duplicates=True)
        for pres_key, roi_mask in data_pairs:
            as_field_mask, (shift_dx, shift_dy) = compare_roi_masks(roi_mask, main_roi_mask, max_shift=self._max_shift)
            self.RoiMaskPresentation().insert1(
                {**pres_key, "roi_mask": roi_mask, "as_field_mask": as_field_mask,
                 "shift_dx": shift_dx, "shift_dy": shift_dy},
                skip_duplicates=True)

    def load_presentation_roi_mask(self, key, roi_mask_dir=None, old_prefix=None, new_prefix=None):
        igor_roi_masks, from_raw_data = (self.raw_params_table & key).fetch1('igor_roi_masks', 'from_raw_data')
        input_file = (self.presentation_table & key).fetch1("pres_data_file")

        if igor_roi_masks == 'yes':
            assert not from_raw_data, 'Inconsistent parameters'
            filesystem_roi_mask = scanm_utils.load_roi_mask_from_h5(filepath=input_file, ignore_not_found=True)
        else:
            roimask_file = to_roi_mask_file(
                input_file, roi_mask_dir=roi_mask_dir, old_prefix=old_prefix, new_prefix=new_prefix)

            if os.path.isfile(roimask_file):
                with open(roimask_file, 'rb') as f:
                    filesystem_roi_mask = pickle.load(f).copy().astype(np.int32)
                filesystem_roi_mask = to_igor_format(filesystem_roi_mask)
            else:
                filesystem_roi_mask = None

        if len((self.RoiMaskPresentation & key).proj()) > 0:
            database_roi_mask = (self.RoiMaskPresentation & key).fetch1('roi_mask')
            if not np.all(filesystem_roi_mask == database_roi_mask):
                raise ValueError(f'ROI mask for key=\n{key}\nhas been changed on filesystem but not in database.')
            else:
                filesystem_roi_mask = database_roi_mask.copy()

        return filesystem_roi_mask

    def plot1(self, key=None):
        key = get_primary_key(table=self.proj() * self.presentation_table.proj(), key=key)
        npixartifact = (self.field_table & key).fetch1('npixartifact')
        data_name, alt_name = (self.userinfo_table & key).fetch1('data_stack_name', 'alt_stack_name')
        main_ch_average = (self.presentation_table.StackAverages & key & f'ch_name="{data_name}"').fetch1('ch_average')
        alt_ch_average = (self.presentation_table.StackAverages & key & f'ch_name="{alt_name}"').fetch1('ch_average')
        roi_mask = (self.RoiMaskPresentation & key).fetch1('roi_mask')
        plot_field(main_ch_average, alt_ch_average, roi_mask=roi_mask, title=key, npixartifact=npixartifact)

    def load_high_res_bg_dict(self, key):
        field = (self.field_table() & key).fetch1("field")
        field_loc = (self.userinfo_table() & key).fetch1("field_loc")
        highres_alias = (self.userinfo_table() & key).fetch1("highres_alias")
        header_path = (self.experiment_table() & key).fetch1('header_path')
        pre_data_path = os.path.join(header_path, (self.userinfo_table() & key).fetch1("pre_data_dir"))
        raw_data_path = os.path.join(header_path, (self.userinfo_table() & key).fetch1("raw_data_dir"))

        try:
            filepath, ch_stacks, wparams = load_high_res_stack(
                pre_data_path=pre_data_path, raw_data_path=raw_data_path,
                highres_alias=highres_alias, field=field, field_loc=field_loc)
        except Exception as e:
            warnings.warn(f'Failed to load high resolution data because of error:\n{e}')
            return dict()

        if ch_stacks is None:
            return dict()

        bg_dict = dict()
        for name, stack in ch_stacks.items():
            bg_dict[f'HR[{name}]'] = np.nanmedian(stack, 2)

        return bg_dict


def load_default_autorois_models(kind='default_rgc'):
    autorois_models = dict()
    _add_autorois_corr(autorois_models, kind=kind)
    return autorois_models if len(autorois_models) > 0 else None


def _add_autorois_corr(autorois_models: dict, kind: str):
    from djimaging.autorois.corr_roi_mask_utils import CorrRoiMask

    if kind == 'default_rgc':
        kws = dict(cut_x=(0, 0), cut_z=(0, 0), min_area_um2=10, max_area_um2=400, n_pix_max=None, line_threshold_q=70,
                   use_ch0_stack=True, grow_use_corr_map=False, grow_threshold=0.1,
                   grow_only_local_thresh_pixels=True)
    elif kind == 'default_bc' or kind == 'default_ac':
        kws = dict(cut_x=(4, 2), cut_z=(8, 2), min_area_um2=0.5, max_area_um2=12.6, n_pix_max=None, line_threshold_q=70,
                   use_ch0_stack=True, grow_use_corr_map=False, grow_threshold=None, line_threshold_min=0.1,
                   grow_only_local_thresh_pixels=True)
    else:
        raise NotImplementedError(kind)

    corr_model = CorrRoiMask(**kws)
    autorois_models['CorrRoiMask'] = corr_model


def _add_cellpose(autorois_models: dict, kind: str = 'default_rgc'):
    from djimaging.autorois.cellpose_wrapper import CellposeWrapper

    if kind == 'default_rgc':
        init_params = dict(
            model_type='cyto',
            gpu=False,
        )

        eval_params = dict(
            min_size=4,
            diameter=15,
            channels=[0, 0],
        )
    else:
        raise NotImplementedError(kind)

    model_cellpose = CellposeWrapper(init_kwargs=init_params, eval_kwargs=eval_params)
    autorois_models['Cellpose'] = model_cellpose
