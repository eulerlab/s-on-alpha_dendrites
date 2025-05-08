import hashlib
import json

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


def make_hash(obj) -> str:
    """
    Creates a 32-character hash that uniquely identifies the content of a Python object.

    This function handles arbitrarily nested objects (dictionaries, lists, etc.) by
    recursively processing the structure and generating a consistent hash based on content.

    Parameters:
    -----------
    obj : Any
        The Python object to hash. Can be a primitive type or a complex nested structure
        like dictionaries, lists, tuples, sets, etc.

    Returns:
    --------
    str
        A 32-character hexadecimal hash string that uniquely identifies the object content

    Notes:
    ------
    - The function handles common Python data types including:
      - Primitives (int, float, str, bool, None)
      - Collections (dict, list, tuple, set)
      - Nested combinations of the above
    - Objects that aren't directly serializable will be converted to their string representation
    - Dictionary keys are sorted to ensure consistent hashing regardless of key order
    """

    def _prepare_for_hashing(value):
        """
        Recursively prepares an object for hashing by converting to serializable form.
        Handles nested structures and ensures consistent representation.
        """
        if isinstance(value, dict):
            # Sort dictionary items by key for consistent ordering
            return {
                k: _prepare_for_hashing(v)
                for k, v in sorted(value.items())
            }
        elif isinstance(value, (list, tuple)):
            return [_prepare_for_hashing(item) for item in value]
        elif isinstance(value, set):
            # Convert set to sorted list for consistent ordering
            return [_prepare_for_hashing(item) for item in sorted(value)]
        elif isinstance(value, (int, float, str, bool)) or value is None:
            # Primitive types can be used directly
            return value
        else:
            # For other types, use their string representation
            return str(value)

    # Prepare the object by handling nested structures and ensuring consistent representation
    prepared_obj = _prepare_for_hashing(obj)

    # Convert to a string with sorted keys for consistent serialization
    json_str = json.dumps(prepared_obj, sort_keys=True)

    # Generate MD5 hash (32 characters) of the JSON string
    hash_obj = hashlib.md5(json_str.encode('utf-8'))

    return hash_obj.hexdigest()
