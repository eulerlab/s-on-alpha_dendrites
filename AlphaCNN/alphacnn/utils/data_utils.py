import hashlib
from collections import OrderedDict
from typing import Mapping, Iterable

import h5py
import pickle
import os


def load_h5_data(file_name):
    """Helper function to load h5 file."""
    with h5py.File(file_name, 'r') as f:
        return {key: f[key][:] for key in list(f.keys())}


def load_h5_file_catch_error(file, raise_error=False):
    if file is None or not os.path.isfile(file):
        if raise_error:
            raise FileNotFoundError(file)
        return None
    else:
        return load_h5_data(file)


def remove_file(filename, notexists='raise', verbose=True):
    if not os.path.isfile(filename):
        if notexists == 'raise':
            raise FileNotFoundError(filename)
        elif notexists == 'ignore':
            return None
        else:
            raise NotImplementedError(notexists)

    os.remove(filename)
    if verbose:
        print(f'Deleted file {filename}')


def load_var(filename, none_var='load', notexists='raise'):
    """Load pickled data"""
    if not os.path.isfile(filename):
        if notexists == 'raise':
            raise FileNotFoundError(filename)
        elif notexists == 'ignore':
            return None
        else:
            raise NotImplementedError(notexists)

    with open(filename, 'rb') as f:
        var = pickle.load(f)
    if var is None:
        if none_var == 'error':
            raise ValueError('var is None')
        elif none_var != 'load':
            raise NotImplementedError(none_var)
    return var


def save_var(var, filename, none_var='save'):
    """Save data to pickle"""
    if var is None:
        if none_var == 'ignore':
            return
        elif none_var == 'error':
            raise ValueError('var is None')
        elif none_var != 'save':
            raise NotImplementedError(none_var)

    with open(filename, 'wb') as f:
        pickle.dump(var, f)


def make_dir(dirname):
    """Creates folder if it does not exist yet"""
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def load_config(config_file):
    import yaml
    with open(config_file, 'r') as file:
        conf = yaml.safe_load(file)
    return conf


def save_config(config, config_file):
    import yaml
    with open(config_file, 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)


def make_hash(obj: object) -> str:
    """
    Given a Python object, returns a 32 character hash string to uniquely identify
    the content of the object. The object can be arbitrary nested (i.e. dictionary
    of dictionary of list etc), and hashing is applied recursively to uniquely
    identify the content.

    For dictionaries (at any level), the key order is ignored when hashing
    so that {"a":5, "b": 3, "c": 4} and {"b": 3, "a": 5, "c": 4} will both
    give rise to the same hash. Exception to this rule is when an OrderedDict
    is passed, in which case difference in key order is respected. To keep
    compatible with previous versions of Python and the assumed general
    intentions, key order will be ignored even in Python 3.7+ where the
    default dictionary is officially an ordered dictionary.

    :param obj: A (potentially nested) Python object
    :return: hash: str - a 32 charcter long hash string to uniquely identify the object.
    """
    hashed = hashlib.md5()

    if isinstance(obj, str):
        hashed.update(obj.encode())
    elif isinstance(obj, OrderedDict):
        for k, v in obj.items():
            hashed.update(str(k).encode())
            hashed.update(make_hash(v).encode())
    elif isinstance(obj, Mapping):
        for k in sorted(obj, key=str):
            hashed.update(str(k).encode())
            hashed.update(make_hash(obj[k]).encode())
    elif isinstance(obj, Iterable):
        for v in obj:
            hashed.update(make_hash(v).encode())
    else:
        hashed.update(str(obj).encode())

    return hashed.hexdigest()
