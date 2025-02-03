import datajoint as dj

from djimaging.tables import core, core_autorois, misc, response, location
from djimaging.user.alpha.tables.traces import WbgSpotsTemplate

schema = dj.Schema()


@schema
class UserInfo(core.UserInfoTemplate):
    pass


@schema
class RawDataParams(core_autorois.RawDataParamsTemplate):
    userinfo_table = UserInfo


@schema
class Experiment(core.ExperimentTemplate):
    userinfo_table = UserInfo

    class ExpInfo(core.ExperimentTemplate.ExpInfo):
        pass

    class Animal(core.ExperimentTemplate.Animal):
        pass

    class Indicator(core.ExperimentTemplate.Indicator):
        pass

    class PharmInfo(core.ExperimentTemplate.PharmInfo):
        pass


@schema
class Field(core_autorois.FieldTemplate):
    userinfo_table = UserInfo
    raw_params_table = RawDataParams
    experiment_table = Experiment

    class StackAverages(core_autorois.FieldTemplate.StackAverages):
        pass


@schema
class Stimulus(core.StimulusTemplate):
    pass


@schema
class Presentation(core_autorois.PresentationTemplate):
    _allow_clipping = True

    userinfo_table = UserInfo
    experiment_table = Experiment
    field_table = Field
    stimulus_table = Stimulus
    raw_params_table = RawDataParams

    class ScanInfo(core_autorois.PresentationTemplate.ScanInfo):
        pass

    class StackAverages(core_autorois.PresentationTemplate.StackAverages):
        pass


@schema
class RoiMask(core_autorois.RoiMaskTemplate):
    _max_shift = 10  # Maximum shift of ROI mask in pixels

    field_table = Field
    presentation_table = Presentation
    experiment_table = Experiment
    userinfo_table = UserInfo
    raw_params_table = RawDataParams

    class RoiMaskPresentation(core_autorois.RoiMaskTemplate.RoiMaskPresentation):
        presentation_table = Presentation


@schema
class Roi(core_autorois.RoiTemplate):
    roi_mask_table = RoiMask
    userinfo_table = UserInfo
    field_table = Field


@schema
class Traces(core_autorois.TracesTemplate):
    userinfo_table = UserInfo
    raw_params_table = RawDataParams
    presentation_table = Presentation
    roi_table = Roi
    roi_mask_table = RoiMask


# Misc
@schema
class HighRes(misc.HighResTemplate):
    field_table = Field
    experiment_table = Experiment
    userinfo_table = UserInfo

    class StackAverages(misc.HighResTemplate.StackAverages):
        pass


@schema
class PreprocessParams(core.PreprocessParamsTemplate):
    stimulus_table = Stimulus


@schema
class PreprocessTraces(core.PreprocessTracesTemplate):
    _baseline_max_dt = 2.  # seconds before stimulus used for baseline calculation

    presentation_table = Presentation
    preprocessparams_table = PreprocessParams
    traces_table = Traces


# use this if you want to upsample averages
@schema
class Snippets(core.SnippetsTemplate):
    _pad_trace = True

    stimulus_table = Stimulus
    presentation_table = Presentation
    traces_table = Traces
    preprocesstraces_table = PreprocessTraces


@schema
class Averages(core.ResampledAveragesTemplate):
    _f_resample = 500  # Frequency in Hz to resample averages
    _norm_kind = 'amp_one'  # How to normalize averages?

    snippets_table = Snippets


@schema
class OpticDisk(location.OpticDiskTemplate):
    userinfo_table = UserInfo
    experiment_table = Experiment
    raw_params_table = RawDataParams


@schema
class RelativeFieldLocation(location.RelativeFieldLocationTemplate):
    field_table = Field
    presentation_table = Presentation
    opticdisk_table = OpticDisk


@schema
class RetinalFieldLocation(location.RetinalFieldLocationTemplate):
    relativefieldlocation_table = RelativeFieldLocation
    expinfo_table = Experiment.ExpInfo


@schema
class ChirpQI(response.ChirpQITemplate):
    stimulus_table = Stimulus
    snippets_table = Snippets


@schema
class HighRes(misc.HighResTemplate):
    field_table = Field
    experiment_table = Experiment
    userinfo_table = UserInfo

    class StackAverages(misc.HighResTemplate.StackAverages):
        pass


@schema
class GroupSnippets(core.GroupSnippetsTemplate):
    stimulus_table = Stimulus
    presentation_table = Presentation
    traces_table = Traces
    preprocesstraces_table = PreprocessTraces


@schema
class WbgSpots(WbgSpotsTemplate):
    presentation_table = Presentation
    group_snippets_table = GroupSnippets


@schema
class SizeRoiMask(core_autorois.RoiMaskTemplate):  # Only for size estimation
    _max_shift = 10  # Maximum shift of ROI mask in pixels

    field_table = Field
    presentation_table = Presentation
    experiment_table = Experiment
    userinfo_table = UserInfo
    raw_params_table = RawDataParams

    class RoiMaskPresentation(core_autorois.RoiMaskTemplate.RoiMaskPresentation):
        presentation_table = Presentation


@schema
class SizeRoi(core_autorois.RoiTemplate):
    roi_mask_table = SizeRoiMask
    userinfo_table = UserInfo
    field_table = Field
