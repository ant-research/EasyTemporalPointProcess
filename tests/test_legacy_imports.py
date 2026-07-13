import importlib
import sys

import pytest


def test_new_imports():
    from easy_tpp.model import EventSampler, IntensityFree, NHP, TorchNHP
    from easy_tpp.model.intensity_free import LogNormalMixtureDistribution

    assert NHP is TorchNHP
    assert all((EventSampler, IntensityFree, LogNormalMixtureDistribution))


def test_legacy_deep_import_warns():
    legacy_package = 'easy_tpp.model.' + 'torch_' + 'model'

    with pytest.warns(
            DeprecationWarning,
            match=rf'{legacy_package} is deprecated'):
        if legacy_package in sys.modules:
            importlib.reload(sys.modules[legacy_package])
        else:
            importlib.import_module(legacy_package)


def test_legacy_deep_module_paths():
    from easy_tpp.model.basemodel import TorchBaseModel as NewTorchBaseModel
    from easy_tpp.model.intensity_free import (
        LogNormalMixtureDistribution as NewLogNormalMixtureDistribution,
    )
    from easy_tpp.model.thinning import EventSampler as NewEventSampler
    legacy_package = 'easy_tpp.model.' + 'torch_' + 'model'
    legacy_basemodel = importlib.import_module(f'{legacy_package}.torch_basemodel')
    legacy_intensity_free = importlib.import_module(
        f'{legacy_package}.torch_intensity_free'
    )
    legacy_thinning = importlib.import_module(f'{legacy_package}.torch_thinning')

    assert (legacy_intensity_free.LogNormalMixtureDistribution
            is NewLogNormalMixtureDistribution)
    assert legacy_basemodel.TorchBaseModel is NewTorchBaseModel
    assert legacy_thinning.EventSampler is NewEventSampler
