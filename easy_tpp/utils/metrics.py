from collections import defaultdict

import numpy as np

from easy_tpp.utils.log_utils import default_logger as logger


class MetricsHelper:
    MAXIMIZE = 'maximize'
    MINIMIZE = 'minimize'
    _registry_center = defaultdict(tuple)

    @staticmethod
    def get_metric_function(name):
        if name in MetricsHelper._registry_center:
            return MetricsHelper._registry_center[name][0]
        else:
            logger.warn(f'Metric is not found: {name}')
            return None

    @staticmethod
    def get_metric_direction(name):
        if name in MetricsHelper._registry_center:
            return MetricsHelper._registry_center[name][1]
        else:
            return None

    @staticmethod
    def get_all_registered_metric():
        return MetricsHelper._registry_center.values

    @staticmethod
    def register(name, direction, overwrite=True):
        registry_center = MetricsHelper._registry_center

        def _add_metric_to_registry(func):
            if name in registry_center:
                if overwrite:
                    registry_center[name] = (func, direction)
                else:
                    logger.warn(f'The metric {name} is already registered, and cannot be overwritten!')
            else:
                registry_center[name] = (func, direction)
            return func

        return _add_metric_to_registry

    @staticmethod
    def metrics_dict_to_str(metrics_dict):
        """ Convert metrics to a string to show in console  """
        eval_info = ''
        for k, v in metrics_dict.items():
            eval_info += '{0} is {1}, '.format(k, v)

        return eval_info[:-2]

    @staticmethod
    def get_metrics_callback_from_names(metric_names):
        """ Metrics function callbacks    """
        metric_functions = []
        metric_names_ = []
        for name in metric_names:
            metric = MetricsHelper.get_metric_function(name)
            if metric is not None:
                metric_functions.append(metric)
                metric_names_.append(name)

        def metrics(preds, labels, **kwargs):
            """ call metrics functions """
            res = dict()
            for metric_name, metric_func in zip(metric_names_, metric_functions):
                res[metric_name.lower()] = metric_func(preds, labels, **kwargs)
            return res

        return metrics


class MetricsTracker:
    """Track and record the metrics.
    """

    def __init__(self):
        self.current_best = {
            'loglike': np.finfo(float).min,
            'distance': np.finfo(float).max
        }
        self.episode_best = 'NeverUpdated'

    def update_best(self, key, value, epoch):
        """Update the recorder for the best metrics.

        Args:
            key (str): metrics key.
            value (float): metrics value.
            epoch (int): num of epoch.

        Raises:
            NotImplementedError: for keys other than 'loglike'.

        Returns:
            bool: whether the recorder has been updated.
        """
        updated = False
        if key == 'loglike':
            if value > self.current_best[key]:
                updated = True
                self.current_best[key] = value
                self.episode_best = epoch
        else:
            raise NotImplementedError

        return updated
