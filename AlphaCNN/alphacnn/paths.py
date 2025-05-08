import os
from pathlib import Path

HOME_DIR = os.path.expanduser("~")
USERNAME = os.path.split(HOME_DIR)[-1]
CONFIG_FILE = os.path.join(HOME_DIR, "datajoint", f"dj_{USERNAME}_conf.json")  # Change if necessary
SCHEMA_PREFIX = f'ageuler_{USERNAME}_alpha_'  # Change if necessary

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
DATASET_PATH = os.path.join(get_project_path(), "dataset")

CONF_STIM_PATH = os.path.join(get_project_path(), 'alphacnn/configs/stim/')
CONF_ENCODER_PATH = os.path.join(get_project_path(), 'alphacnn/configs/encoder/')
CONF_DECODER_PATH = os.path.join(get_project_path(), 'alphacnn/configs/decoder/')