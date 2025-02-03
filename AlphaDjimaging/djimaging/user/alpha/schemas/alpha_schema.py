import numpy as np
import datajoint as dj

from djimaging.tables import core, core_autorois, clustering, rf_glm, location, receptivefield, response
from djimaging.tables.location import location_from_table
from djimaging.user.alpha.tables import morphology, quality, traces, rf_contours, misc

schema = dj.Schema()


# Core tables
@schema
class UserInfo(core.UserInfoTemplate):
    pass


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
class CellTags(morphology.CellTagsTemplate):
    experiment_table = Experiment


@schema
class Field(core.FieldTemplate):
    _load_field_roi_masks = True
    userinfo_table = UserInfo
    experiment_table = Experiment

    class RoiMask(core.FieldTemplate.RoiMask):
        pass

    class StackAverages(core.FieldTemplate.StackAverages):
        pass


@schema
class RoiKind(misc.RoiKindTemplate):
    field_table = Field


@schema
class Stimulus(core.StimulusTemplate):
    pass


@schema
class RawDataParams(core.RawDataParamsTemplate):
    pass


@schema
class Presentation(core.PresentationTemplate):
    userinfo_table = UserInfo
    experiment_table = Experiment
    field_table = Field
    stimulus_table = Stimulus
    params_table = RawDataParams

    class ScanInfo(core.PresentationTemplate.ScanInfo):
        pass

    class StackAverages(core.PresentationTemplate.StackAverages):
        pass

    class RoiMask(core.PresentationTemplate.RoiMask):
        pass


@schema
class MatchTemplate(core_autorois.RoiMaskTemplate):
    experiment_table = Experiment
    field_table = Field
    presentation_table = Presentation
    userinfo_table = UserInfo
    raw_params_table = RawDataParams

    class RoiMaskPresentation(core_autorois.RoiMaskTemplate.RoiMaskPresentation):
        presentation_table = Presentation


@schema
class Roi(core.RoiTemplate):
    userinfo_table = UserInfo
    field_or_pres_table = Field


@schema
class Traces(core.TracesTemplate):
    _ignore_incompatible_roi_masks = True  # For gChirp light artifact was drawn, so mask is not compatible
    userinfo_table = UserInfo
    params_table = RawDataParams
    presentation_table = Presentation
    roi_table = Roi


@schema
class PreprocessParams(core.PreprocessParamsTemplate):
    pass


@schema
class PreprocessTraces(core.PreprocessTracesTemplate):
    _baseline_max_dt = np.inf

    presentation_table = Presentation
    preprocessparams_table = PreprocessParams
    traces_table = Traces


@schema
class Snippets(core.SnippetsTemplate):
    _dt_base_rng = {
        # 'sinespot': (-0.2, 0.1),
    }
    _pad_trace = {
        # 'sinespot': True,
    }
    _delay = {
        'sinespot': -0.25,
    }

    stimulus_table = Stimulus
    presentation_table = Presentation
    traces_table = Traces
    preprocesstraces_table = PreprocessTraces


@schema
class Averages(core.AveragesTemplate):
    _norm_kind = 'amp_one'
    snippets_table = Snippets


@schema
class ChirpQI(response.ChirpQITemplate):
    stimulus_table = Stimulus
    snippets_table = Snippets


# GLM tables
@schema
class GLMDNoiseTraceParams(receptivefield.DNoiseTraceParamsTemplate):
    pass


@schema
class GLMDNoiseTrace(receptivefield.DNoiseTraceTemplate):
    presentation_table = Presentation
    stimulus_table = Stimulus
    traces_table = PreprocessTraces
    params_table = GLMDNoiseTraceParams


@schema
class RFGLMParams(rf_glm.RFGLMParamsTemplate):
    pass


@schema
class RFGLM(rf_glm.RFGLMTemplate):
    params_table = RFGLMParams
    noise_traces_table = GLMDNoiseTrace


@schema
class SplitRFGLMParams(receptivefield.SplitRFParamsTemplate):
    pass


@schema
class SplitRFGLM(receptivefield.SplitRFTemplate):
    _max_dt_future = 0.05
    stimulus_table = Stimulus
    rf_table = RFGLM
    split_rf_params_table = SplitRFGLMParams


# Fit sRF for GLM
@schema
class FitGauss2DRFGLM(receptivefield.FitGauss2DRFTemplate):
    split_rf_table = SplitRFGLM
    stimulus_table = Stimulus


@schema
class GlmGaussOffset(morphology.RfOffsetTemplate):
    _stimulus_offset_dx_um = -15.
    _stimulus_offset_dy_um = -15.

    stimulus_tab = Stimulus
    rf_split_tab = SplitRFGLM
    rf_fit_tab = FitGauss2DRFGLM


@schema
class FitPosGauss2DRFGLM(receptivefield.FitGauss2DRFTemplate):
    _polarity = 1

    split_rf_table = SplitRFGLM
    stimulus_table = Stimulus


@schema
class GlmPosGaussOffset(morphology.RfOffsetTemplate):
    _stimulus_offset_dx_um = -15.
    _stimulus_offset_dy_um = -15.

    stimulus_tab = Stimulus
    rf_split_tab = SplitRFGLM
    rf_fit_tab = FitPosGauss2DRFGLM


@schema
class FitDoG2DRFGLM(receptivefield.FitDoG2DRFTemplate):
    split_rf_table = SplitRFGLM
    stimulus_table = Stimulus


@schema
class GlmDogOffset(morphology.RfOffsetTemplate):
    _stimulus_offset_dx_um = -15.
    _stimulus_offset_dy_um = -15.

    stimulus_tab = Stimulus
    rf_split_tab = SplitRFGLM
    rf_fit_tab = FitDoG2DRFGLM


@schema
class FitPosDoG2DRFGLM(receptivefield.FitDoG2DRFTemplate):
    _polarity = 1
    split_rf_table = SplitRFGLM
    stimulus_table = Stimulus


@schema
class GlmPosDogOffset(morphology.RfOffsetTemplate):
    _stimulus_offset_dx_um = -15.
    _stimulus_offset_dy_um = -15.

    stimulus_tab = Stimulus
    rf_split_tab = SplitRFGLM
    rf_fit_tab = FitPosDoG2DRFGLM


@schema
class TempRFGLMProperties(receptivefield.TempRFPropertiesTemplate):
    rf_table = RFGLM
    split_rf_table = SplitRFGLM


# Sinespot tables
@schema
class SineSpotQI(response.SineSpotQITemplate):
    stimulus_table = Stimulus
    snippets_table = Snippets


@schema
class SineSpotSurroundIndex(response.SineSpotSurroundIndexTemplate):
    _stim_name = 'sinespot'
    _dt_spot = 1.  # Duration of the spot in seconds including the pause between spots
    _dt_baseline = 0.25  # Duration of the baseline in seconds
    snippets_table = Snippets


# Location tables
@schema
class RetinalFieldLocationTableParams(location_from_table.RetinalFieldLocationTableParamsTemplate):
    pass


@schema
class RetinalFieldLocationFromTable(location_from_table.RetinalFieldLocationFromTableTemplate):
    field_table = Field
    params_table = RetinalFieldLocationTableParams
    expinfo_table = Experiment.ExpInfo


@schema
class RetinalFieldLocationCat(location.RetinalFieldLocationCatTemplate):
    _center_dist = 0.
    _ventral_dorsal_key = 'ventral_dorsal_pos'
    _temporal_nasal_key = 'temporal_nasal_pos'
    retinalfieldlocation_table = RetinalFieldLocationFromTable


@schema
class RetinalFieldLocationWing(location.RetinalFieldLocationWingTemplate):
    _center_dist = 0.
    _ventral_dorsal_key = 'ventral_dorsal_pos'
    _temporal_nasal_key = 'temporal_nasal_pos'
    retinalfieldlocation_table = RetinalFieldLocationFromTable


# Quality tables
@schema
class QualityParams(quality.QualityParamsTemplate):
    pass


@schema
class QualityIndex(quality.QualityIndexTemplate):
    params_table = QualityParams
    roi_table = Roi
    sinespot_qi_table = SineSpotQI
    chirp_qi_table = ChirpQI


# Clustering tables
@schema
class DownsampledAverages(traces.DownsampledAveragesTemplate):
    averages_table = Averages
    _fdownsample = 4


@schema
class ChirpSurroundIndex(response.ChirpSurroundIndexTemplate):
    _l_name = 'lChirp'
    _g_name = 'gChirp'
    _normalized_avg = False
    _fixed_polarity = 1  # 1 for ON, -1 for OFF, None for automatic detection

    _t_on_step = 2
    _t_off_step = 5
    _dt = 3
    _t_plot_max = 8
    snippets_table = Snippets


# Events per second
@schema
class EventsPerSecondParams(receptivefield.DNoiseTraceParamsTemplate):
    pass


@schema
class EventsPerSecond(traces.EventsPerSecondTemplate):
    params_table = EventsPerSecondParams

    presentation_table = Presentation
    stimulus_table = Stimulus
    traces_table = PreprocessTraces


# Contours
@schema
class GLMContoursParams(rf_contours.RFContoursParamsTemplate):
    pass


@schema
class GLMContours(rf_contours.RFContoursTemplate):
    split_rf_table = SplitRFGLM
    stimulus_table = Stimulus
    rf_contours_params_table = GLMContoursParams


@schema
class GLMContourMetrics(rf_contours.RFContourMetricsTemplate):
    rf_contour_table = GLMContours


@schema
class GLMContourOffset(morphology.RfOffsetTemplate):
    _stimulus_offset_dx_um = -15.
    _stimulus_offset_dy_um = -15.

    stimulus_tab = Stimulus
    rf_split_tab = SplitRFGLM
    rf_fit_tab = GLMContours


# Morphology tables
@schema
class SWC(morphology.SWCTemplate):
    _swc_folder = 'Raw'

    field_table = Field
    experiment_table = Experiment


@schema
class MorphPaths(morphology.MorphPathsTemplate):
    field_table = Field
    swc_table = SWC


@schema
class ConvexHull(morphology.ConvexHullTemplate):
    morph_table = MorphPaths


@schema
class LineStack(morphology.LineStackTemplate):
    morph_table = MorphPaths
    field_table = Field


@schema
class RoiStackPosParams(morphology.RoiStackPosParamsTemplate):
    pass


@schema
class RelativeRoiPos(morphology.RoiOffsetTemplate):
    presentation_table = Presentation
    roi_table = Roi
    roi_mask_table = Presentation.RoiMask


@schema
class FieldStackPos(morphology.FieldStackPosTemplate):
    userinfo_table = UserInfo
    experiment_table = Experiment
    field_table = Field
    linestack_table = LineStack
    params_table = RoiStackPosParams
    morph_table = MorphPaths
    template_table = MatchTemplate

    class RoiStackPos(morphology.FieldStackPosTemplate.RoiStackPos):
        roi_table = Roi

    class FitInfo(morphology.FieldStackPosTemplate.FitInfo):
        pass


@schema
class FieldPathPos(morphology.FieldPathPosTemplate):
    field_table = Field
    linestack_table = LineStack
    fieldstackpos_table = FieldStackPos


@schema
class FieldCalibratedStackPos(morphology.FieldCalibratedStackPosTemplate):
    fieldstackpos_table = FieldStackPos
    linestack_table = LineStack
    field_table = Field
    morph_table = MorphPaths
    params_table = RoiStackPosParams

    class RoiCalibratedStackPos(morphology.FieldCalibratedStackPosTemplate.RoiCalibratedStackPos):
        roi_table = Roi


@schema
class FieldPosMetrics(morphology.FieldPosMetricsTemplate):
    fieldcalibratedstackpos_table = FieldCalibratedStackPos
    morph_table = MorphPaths

    class RoiPosMetrics(morphology.FieldPosMetricsTemplate.RoiPosMetrics):
        roi_table = Roi


@schema
class FieldRoiPosMetrics(morphology.FieldRoiPosMetricsTemplate):
    field_table = Field
    roi_kind_table = RoiKind
    fieldposmetrics_table = FieldPosMetrics


# Clustering
@schema
class FeaturesParams(clustering.FeaturesParamsTemplate):
    pass


@schema
class Features(clustering.FeaturesTemplate):
    _restr_filter = dict(q_tot=1)
    roi_filter_table = QualityIndex

    params_table = FeaturesParams
    averages_table = DownsampledAverages
    roi_table = Roi
    roikind_table = RoiKind
    roicalibratedstackpos_table = FieldCalibratedStackPos.RoiCalibratedStackPos

    class RoiFeatures(clustering.FeaturesTemplate.RoiFeatures):
        pass


@schema
class ClusteringParameters(clustering.ClusteringParametersTemplate):
    pass


@schema
class Clustering(clustering.ClusteringTemplate):
    features_table = Features
    params_table = ClusteringParameters

    class RoiCluster(clustering.ClusteringTemplate.RoiCluster):
        pass
