import warnings

import datajoint as dj
import numpy as np


def reformat_numerical_trial_info(trial_info):
    """Change old list format to new list[dict] format"""
    return [dict(name=trial_info_i, ntrigger=1) for i, trial_info_i in enumerate(trial_info)]


def check_trial_info(trial_info, ntrigger_rep):
    """Check if trial info is valid and change to new format if necessary"""
    if not isinstance(trial_info, (list, np.ndarray)):
        raise TypeError('trial_info must either be a list or an array')

    if not isinstance(trial_info[0], dict):
        trial_info = reformat_numerical_trial_info(trial_info)

    for trial_info_i in trial_info:
        for k, v in trial_info_i.items():
            assert k in ['name', 'ntrigger', 'ntrigger_split'], f'Unknown key in trial_info k={k}'

            if k in ['ntrigger', 'ntrigger_split']:
                assert int(v) == v, f'Value for k={k} but be an integer but is v={v}'

    ntrigger_ti = sum([trial_info_i["ntrigger"] for trial_info_i in trial_info])
    if not (ntrigger_ti == ntrigger_rep):
        msg = f'Number of triggers in trial_info={ntrigger_ti} must match ntrigger_rep={ntrigger_rep}.'
        if ntrigger_rep > 1:
            raise ValueError(msg)
        else:
            # Raise only warnings for previous work-around solution
            warnings.warn(msg)

    return trial_info


class StimulusTemplate(dj.Manual):
    database = ""

    @property
    def definition(self):
        definition = """
        # Light stimuli
        stim_name           :varchar(16)       # Unique string identifier
        ---
        alias               :varchar(191)      # Strings (_ seperator) to identify this stimulus, not case sensitive!
        stim_family=""      :varchar(32)       # To group stimuli (e.g. gChirp and lChirp) for downstream processing 
        framerate=0         :float              # framerate in Hz
        isrepeated=0        :tinyint unsigned   # Is the stimulus repeated? Used for snippets
        ntrigger_rep=0      :int unsigned       # Number of triggers (per repetition)  
        stim_path=""        :varchar(191)       # Path to hdf5 file containing numerical array and info about stim
        commit_id=""        :varchar(191)       # Commit id corresponding to stimulus entry in GitHub repo
        stim_hash=""        :varchar(191)       # QDSPy hash
        trial_info=NULL     :longblob           # trial information, e.g. directions of moving bar
        stim_trace=NULL     :longblob           # array of stimulus if available
        stim_dict=NULL      :longblob           # stimulus information dictionary, contains e.g. spatial extent
        """
        return definition

    def check_alias(self, alias: str, stim_name: str):
        existing_aliases = (self - [dict(stim_name=stim_name)]).fetch('alias')  # Skip duplicate comparison
        for existing_alias in existing_aliases:
            for existing_alias_i in existing_alias.split('_'):
                assert existing_alias_i not in alias.split('_'), \
                    f'Found existing alias `{existing_alias_i}`. Set `unique_alias` to False to insert duplicate.'

    def add_stimulus(self, stim_name: str, alias: str, stim_family: str = "", framerate: float = 0,
                     isrepeated: bool = 0, ntrigger_rep: int = 0, stim_path: str = "", commit_id: str = "",
                     trial_info: object = None, stim_trace: np.ndarray = None, stim_dict: dict = None,
                     skip_duplicates: bool = False, unique_alias: bool = True) -> None:
        """
        Add stimulus to database
        :param stim_name: See table definition.
        :param alias: See table definition.
        :param stim_family: See table definition.
        :param framerate: See table definition.
        :param isrepeated: See table definition.
        :param ntrigger_rep: See table definition.
        :param stim_path: See table definition.
        :param commit_id: See table definition.
        :param trial_info: See table definition.
        :param stim_trace: See table definition.
        :param stim_dict: See table definition.
        :param skip_duplicates: Silently skip duplicates.
        :param unique_alias: Check if any of the aliases is already in use.
        """
        assert np.round(ntrigger_rep) == ntrigger_rep, 'Value needs to be an integer'
        assert np.round(isrepeated) == isrepeated, 'Value needs to be an integer'

        if unique_alias:
            self.check_alias(alias, stim_name=stim_name)

        if stim_dict is not None:
            missing_info = [k for k, v in stim_dict.items() if v is None]
            if len(missing_info) > 0:
                warnings.warn(f'Values for {missing_info} in `stim_dict` for stimulus `{stim_name}` are None. '
                              + 'This may cause problems downstream.')

        if trial_info is not None:
            # noinspection PyTypeChecker
            check_trial_info(trial_info=trial_info, ntrigger_rep=ntrigger_rep)

        key = {
            "stim_name": stim_name,
            "alias": alias.lower(),
            "stim_family": stim_family,
            "ntrigger_rep": int(np.round(ntrigger_rep)),
            "isrepeated": int(np.round(isrepeated)),
            "framerate": framerate,
            "stim_path": stim_path,
            "commit_id": commit_id,
            "trial_info": trial_info,
            "stim_trace": stim_trace,
            "stim_dict": stim_dict,
        }

        self.insert1(key, skip_duplicates=skip_duplicates)

    def update_trial_info_format(self, restriction=None):
        """Update all trial info formats to list[dict] format. Breaks compatability with old code!"""
        if restriction is None:
            restriction = dict()

        for key in (self & restriction).proj().fetch(as_dict=True):
            trial_info, ntrigger_rep = (self & key).fetch1('trial_info', 'ntrigger_rep')
            if trial_info is not None:
                trial_info = check_trial_info(trial_info=trial_info, ntrigger_rep=ntrigger_rep)
                self.update1(dict(**key, trial_info=trial_info))

    def add_nostim(self, alias="nostim_none", skip_duplicates=False):
        """Add none stimulus"""
        self.add_stimulus(
            stim_name='nostim',
            alias=alias,
            framerate=0,
            skip_duplicates=skip_duplicates,
            unique_alias=True
        )

    def add_noise(self, stim_name: str = "noise", stim_family: str = 'noise',
                  framerate: float = 5., ntrigger_rep: int = 1500, isrepeated: bool = False,
                  alias: str = None, pix_n_x: int = None, pix_n_y: int = None,
                  pix_scale_x_um: float = None, pix_scale_y_um: float = None, stim_trace=None,
                  skip_duplicates: bool = False) -> None:

        if alias is None:
            alias = f"dn_noise_dn{pix_scale_x_um}m_noise{pix_scale_x_um}m"

        stim_dict = {
            "pix_n_x": pix_n_x,
            "pix_n_y": pix_n_y,
            "pix_scale_x_um": pix_scale_x_um,
            "pix_scale_y_um": pix_scale_y_um,
        }

        self.add_stimulus(
            stim_name=stim_name,
            stim_family=stim_family,
            alias=alias,
            ntrigger_rep=ntrigger_rep,
            isrepeated=isrepeated,
            framerate=framerate,
            skip_duplicates=skip_duplicates,
            unique_alias=True,
            stim_trace=stim_trace,
            stim_dict=stim_dict,
        )

    def add_chirp(self, stim_name: str = "chirp", stim_family: str = 'chirp',
                  spatialextent: float = None, framerate: float = 1 / 60.,
                  ntrigger_rep: int = 2, isrepeated: bool = True, stim_trace: np.ndarray = None,
                  alias: str = None, skip_duplicates: bool = False):

        if alias is None:
            alias = "chirp_gchirp_globalchirp_lchirp_localchirp"

        stim_dict = {
            "spatialextent": spatialextent,
        }

        self.add_stimulus(
            stim_name=stim_name,
            stim_family=stim_family,
            alias=alias,
            ntrigger_rep=ntrigger_rep,
            isrepeated=isrepeated,
            framerate=framerate,
            skip_duplicates=skip_duplicates,
            unique_alias=True,
            stim_trace=stim_trace,
            stim_dict=stim_dict,
        )

    def add_movingbar(self, stim_name: str = "movingbar", stim_family: str = 'movingbar',
                      bardx: float = None, bardy: float = None, velumsec: float = None, tmovedurs: float = None,
                      ntrigger_rep: int = 1, isrepeated: bool = 1, trial_info=None, framerate: float = 1 / 60.,
                      alias: str = None, skip_duplicates: bool = False):

        if trial_info is None:
            trial_info = np.array([0, 180, 45, 225, 90, 270, 135, 315])

        if alias is None:
            alias = "mb_mbar_bar_movingbar"

        stim_dict = {
            "bardx": bardx,
            "bardy": bardy,
            "velumsec": velumsec,
            "tmovedurs": tmovedurs,
        }

        self.add_stimulus(
            stim_name=stim_name,
            stim_family=stim_family,
            alias=alias,
            ntrigger_rep=ntrigger_rep,
            isrepeated=isrepeated,
            framerate=framerate,
            trial_info=trial_info,
            stim_dict=stim_dict,
            skip_duplicates=skip_duplicates,
            unique_alias=True,
        )
