import os
from pathlib import Path

PROJECT_ROOT = os.path.abspath(Path(__file__).resolve().parents[1])
VIDEO_ROOT = os.path.join(os.path.abspath(Path(__file__).resolve().parents[3]), 'data', 'Model')

def get_project_path():
    if os.path.exists(PROJECT_ROOT):
        return PROJECT_ROOT
    else:
        raise FileNotFoundError(f'Project path {PROJECT_ROOT} does not exist')


def get_video_in_path():
    if os.path.exists(VIDEO_ROOT):
        return VIDEO_ROOT
    else:
        raise FileNotFoundError(f'Project path {PROJECT_ROOT} does not exist')


VIDEO_OUT_PATH = os.path.join(get_project_path(), "videos")
FIG_OUT_PATH = os.path.join(get_project_path(), "figures")
DATA_OUT_PATH = os.path.join(get_project_path(), "data")
DATASET_PATH = os.path.join(get_project_path(), "dataset")

CONF_STIM_PATH = os.path.join(get_project_path(), 'alphacnn/configs/stim/')
CONF_ENCODER_PATH = os.path.join(get_project_path(), 'alphacnn/configs/encoder/')
CONF_DECODER_PATH = os.path.join(get_project_path(), 'alphacnn/configs/decoder/')

# Specific to user
CONFIG_FILE = 'enter_some_path/conf.json'
SCHEMA_PREFIX = 'enter_some_prefix_'