import numpy as np

from easy_tpp.utils.const import PredOutputIndex
from easy_tpp.utils.metrics import MetricsHelper


@MetricsHelper.register(name='rmse', direction=MetricsHelper.MINIMIZE, overwrite=False)
def rmse_metric_function(predictions, labels, **kwargs):
    """Compute rmse metrics of the time predictions.

    Args:
        predictions (np.array): model predictions.
        labels (np.array): ground truth.

    Returns:
        float: average rmse of the time predictions.
    """
    seq_mask = kwargs.get('seq_mask')
    if seq_mask is None or len(seq_mask) == 0:
        # If mask is empty or None, use all predictions
        pred = predictions[PredOutputIndex.TimePredIndex]
        label = labels[PredOutputIndex.TimePredIndex]
    else:
        pred = predictions[PredOutputIndex.TimePredIndex][seq_mask]
        label = labels[PredOutputIndex.TimePredIndex][seq_mask]

    pred = np.reshape(pred, [-1])
    label = np.reshape(label, [-1])
    return np.sqrt(np.mean((pred - label) ** 2))


@MetricsHelper.register(name='acc', direction=MetricsHelper.MAXIMIZE, overwrite=False)
def acc_metric_function(predictions, labels, **kwargs):
    """Compute accuracy ratio metrics of the type predictions.

    Args:
        predictions (np.array): model predictions.
        labels (np.array): ground truth.

    Returns:
        float: accuracy ratio of the type predictions.
    """
    seq_mask = kwargs.get('seq_mask')
    if seq_mask is None or len(seq_mask) == 0:
        # If mask is empty or None, use all predictions
        pred = predictions[PredOutputIndex.TypePredIndex]
        label = labels[PredOutputIndex.TypePredIndex]
    else:
        pred = predictions[PredOutputIndex.TypePredIndex][seq_mask]
        label = labels[PredOutputIndex.TypePredIndex][seq_mask]
    pred = np.reshape(pred, [-1])
    label = np.reshape(label, [-1])
    return np.mean(pred == label)
