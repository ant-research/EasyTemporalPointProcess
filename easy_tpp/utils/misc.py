import copy
import os
import pickle

import numpy as np
import yaml
import json
from easy_tpp.utils.const import RunnerPhase


def py_assert(condition, exception_type, msg):
    """An assert function that ensures the condition holds, otherwise throws a message.

    Args:
        condition (bool): a formula to ensure validity.
        exception_type (_StandardError): Error type, such as ValueError.
        msg (str): a message to throw out.

    Raises:
        exception_type: throw an error when the condition does not hold.
    """
    if not condition:
        raise exception_type(msg)


def make_config_string(config, max_num_key=4):
    """Generate a name for config files.

    Args:
        config (dict): configuration dict.
        max_num_key (int, optional): max number of keys to concat in the output. Defaults to 4.

    Returns:
        dict: a concatenated string from config dict.
    """
    str_config = ''
    num_key = 0
    for k, v in config.items():
        if num_key < max_num_key:  # for the moment we only record model name
            if k == 'name':
                str_config += str(v) + '_'
                num_key += 1
    return str_config[:-1]


def save_yaml_config(save_dir, config):
    """A function that saves a dict of config to yaml format file.

    Args:
        save_dir (str): the path to save config file.
        config (dict): the target config object.
    """
    prt_dir = os.path.dirname(save_dir)

    from collections import OrderedDict
    # add yaml representer for different type
    yaml.add_representer(
        OrderedDict,
        lambda dumper, data: dumper.represent_mapping('tag:yaml.org,2002:map', data.items())
    )

    if prt_dir != '' and not os.path.exists(prt_dir):
        os.makedirs(prt_dir)

    with open(save_dir, 'w') as f:
        yaml.dump(config, stream=f, default_flow_style=False, sort_keys=False)

    return


def load_yaml_config(config_dir):
    """ Load yaml config file from disk.

    Args:
        config_dir: str or Path
            The path of the config file.

    Returns:
        Config: dict.
    """
    with open(config_dir) as config_file:
        # load configs
        config = yaml.load(config_file, Loader=yaml.FullLoader)

    return config


def get_stage(stage):
    stage = stage.lower()
    if stage in ['train', 'training']:
        return RunnerPhase.TRAIN
    elif stage in ['valid', 'dev', 'eval']:
        return RunnerPhase.VALIDATE
    else:
        return RunnerPhase.PREDICT


def create_folder(*args):
    """Create path if the folder doesn't exist.

    Returns:
        str: the created folder's path.
    """
    path = os.path.join(*args)
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def load_pickle(file_dir):
    """Load from pickle file.

    Args:
        file_dir (BinaryIO): dir of the pickle file.

    Returns:
        any type: the loaded data.
    """
    with open(file_dir, 'rb') as file:
        try:
            data = pickle.load(file, encoding='latin-1')
        except Exception:
            data = pickle.load(file)

    return data


def save_pickle(file_dir, object_to_save):
    """Save the object to a pickle file.

    Args:
        file_dir (str): dir of the pickle file.
        object_to_save (any): the target data to be saved.
    """

    with open(file_dir, "wb") as f_out:
        pickle.dump(object_to_save, f_out)

    return


def save_json(data, file_dir):
    """
    Save data to a JSON file.

    Args:
        data: The data to be saved. It should be JSON serializable (e.g., a dictionary or list).
        file_dir (str): The path to the file where the data will be saved.

    Raises:
        IOError: If the file cannot be opened or written to.
    """
    with open(file_dir, 'w') as outfile:
        json.dump(data, outfile, indent=4)
    print(f"Data successfully saved to {file_dir}")


def load_json(file_dir):
    """
    Reads data from a JSON file.

    Args:
        file_dir (str): The path to the JSON file to be read.

    Returns:
        The data read from the JSON file.

    Raises:
        IOError: If the file cannot be opened or read.
        json.JSONDecodeError: If the file is not a valid JSON.
    """
    with open(file_dir, 'r') as infile:
        data = json.load(infile)
    return data


def has_key(target_dict, target_keys):
    """Check if the keys exist in the target dict.

    Args:
        target_dict (dict): a dict.
        target_keys (str, list): list of keys.

    Returns:
        bool: True if all the key exist in the dict; False otherwise.
    """
    if not isinstance(target_keys, list):
        target_keys = [target_keys]
    for k in target_keys:
        if k not in target_dict:
            return False
    return True


def array_pad_cols(arr, max_num_cols, pad_index):
    """Pad the array by columns.

    Args:
        arr (np.array): target array to be padded.
        max_num_cols (int): target num cols for padded array.
        pad_index (int): pad index to fill out the padded elements

    Returns:
        np.array: the padded array.
    """
    res = np.ones((arr.shape[0], max_num_cols)) * pad_index

    res[:, :arr.shape[1]] = arr

    return res


def concat_element(arrs, pad_index):
    """ Concat element from each batch output  """

    n_lens = len(arrs)
    n_elements = len(arrs[0])

    # found out the max seq len (num cols) in output arrays
    max_len = max([x[0].shape[1] for x in arrs])

    concated_outputs = []
    for j in range(n_elements):
        a_output = []
        for i in range(n_lens):
            arrs_ = array_pad_cols(arrs[i][j], max_num_cols=max_len, pad_index=pad_index)
            a_output.append(arrs_)

        concated_outputs.append(np.concatenate(a_output, axis=0))

    # n_elements * [ [n_lens, dim_of_element] ]
    return concated_outputs


def to_dict(obj, classkey=None):
    if isinstance(obj, dict):
        data = {}
        for (k, v) in obj.items():
            data[k] = to_dict(v, classkey)
        return data
    elif hasattr(obj, "_ast"):
        return to_dict(obj._ast())
    elif hasattr(obj, "__iter__"):
        return [to_dict(v, classkey) for v in obj]
    elif hasattr(obj, "__dict__"):
        data = dict([(key, to_dict(value, classkey))
                     for key, value in obj.__dict__.iteritems()
                     if not callable(value) and not key.startswith('_') and key not in ['name']])
        if classkey is not None and hasattr(obj, "__class__"):
            data[classkey] = obj.__class__.__name__
        return data
    else:
        return obj


def dict_deep_update(target, source, is_add_new_key=True):
    """ Update 'target' dict by 'source' dict deeply, and return a new dict copied from target and source deeply.

    Args:
        target: dict
        source: dict
        is_add_new_key: bool, default True.
            Identify if add a key that in source but not in target into target.

    Returns:
        New target: dict. It contains the both target and source values, but keeps the values from source when the key
        is duplicated.
    """
    # deep copy for avoiding to modify the original dict
    result = copy.deepcopy(target) if target is not None else {}

    if source is None:
        return result

    for key, value in source.items():
        if key not in result:
            if is_add_new_key:
                result[key] = value
            continue
        # both target and source have the same key
        base_type_list = [int, float, str, tuple, bool]
        if type(result[key]) in base_type_list or type(source[key]) in base_type_list:
            result[key] = value
        else:
            result[key] = dict_deep_update(result[key], source[key], is_add_new_key=is_add_new_key)
    return result
