import os
import random

import numpy as np
import torch


def set_seed(seed=1029):
    """Setup random seed.

    Args:
        seed (int, optional): random seed. Defaults to 1029.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def set_device(gpu=-1):
    """Setup the device.

    Args:
        gpu (int, optional): num of GPU to use. Defaults to -1 (not use GPU, i.e., use CPU).
    """
    if gpu >= 0 and torch.cuda.is_available():
        device = torch.device("cuda:" + str(gpu))
    else:
        device = torch.device("cpu")
    return device


def set_optimizer(optimizer, params, lr):
    """Setup the optimizer.

    Args:
        optimizer (str): name of the optimizer.
        params (dict): dict of params for the optimizer.
        lr (float): learning rate.

    Raises:
        NotImplementedError: if the optimizer's name is wrong or the optimizer is not supported,
        we raise error.

    Returns:
        torch.otim: torch optimizer.
    """
    if isinstance(optimizer, str):
        if optimizer.lower() == "adam":
            optimizer = "Adam"
    try:
        optimizer = getattr(torch.optim, optimizer)(params, lr=lr)
    except Exception:
        raise NotImplementedError("optimizer={} is not supported.".format(optimizer))
    return optimizer


def count_model_params(model):
    """Count the number of params of the model.

    Args:
        model (torch.nn.Moduel): a torch model.

    Returns:
        int: total num of the parameters.
    """
    return sum(p.numel() for p in model.parameters())
