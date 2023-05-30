import os
import random

import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib

if tf.__version__ >= '2.0':
    tf = tf.compat.v1
    tf.disable_v2_behavior()


def set_seed(seed=1029):
    """Setup random seed.

    Args:
        seed (int, optional): random seed. Defaults to 1029.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_random_seed(seed)


def is_gpu_available():
    """Check if GPU is available.

    Returns:
        bool: True if available False if not.
    """
    local_device_protos = device_lib.list_local_devices()
    for device in local_device_protos:
        if device.device_type == 'GPU':
            return True
    return False


def set_device(gpu=-1):
    """Setup the device.

    Args:
        gpu (int, optional): _description_. Defaults to -1.
    """
    if gpu >= 0 and is_gpu_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    return


def set_optimizer(optimizer, lr):
    """Setup the optimizer.

    Args:
        optimizer (str): name of the optimizer.
        lr (float): learning rate.

    Raises:
        NotImplementedError: if the optimizer's name is wrong or the optimizer is not supported,
        we raise error.

    Returns:
        tf.train.optimzer: tf optimizer.
    """
    optimizer = optimizer.capitalize() + 'Optimizer'
    try:
        optimizer = getattr(tf.train, optimizer)(learning_rate=lr)
    except Exception:
        raise NotImplementedError("optimizer={} is not supported.".format(optimizer))

    return optimizer


def get_shape_list(x):
    """Deal with dynamic shape in tensorflow cleanly.

    Args:
        x (tensor): input tensor.

    Returns:
        list: shape list of the tensor.
    """
    static = x.shape.as_list()
    dynamic = tf.shape(x)
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]


def tensordot(tensor_a, tensor_b):
    """ Tensor dot function. The last dimension of tensor_a and the first dimension of tensor_b must be the same.

    Args:
        tensor_a (tensor): input tensor.
        tensor_b (tensor): input tensor.

    Returns:
        tensor: the result of tensor_a tensor dot tensor_b.
    """
    last_idx_a = len(tensor_a.get_shape().as_list()) - 1
    return tf.tensordot(tensor_a, tensor_b, [[last_idx_a], [0]])


def swap_axes(tensor, axis1, axis2):
    """Interchange two axes of an tensor.
    :param tensor:
    :param axis1: First axis.
    :param axis2: Second axis.
    :return:
    """
    tensor_perm = list(range(len(tensor.shape.as_list())))
    tensor_perm[axis1] = axis2
    tensor_perm[axis2] = axis1

    return tf.transpose(tensor, perm=tensor_perm)


def create_tensor(shape, value):
    """Creates a tensor with all elements set to be the value.

    Args:
        shape (list): the shape of the target tensor to be created.
        value (float): value to fill the tensor.

    Returns:
        tensor: created tensor with target value filled.
    """
    tensor_shape = tf.stack(shape)
    return tf.fill(tensor_shape, value)


def count_model_params():
    """Count the number of params of the model.

    Args:
        model (tf.keras.Model): a torch model.

    Returns:
        int: total num of the parameters.
    """
    return np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
