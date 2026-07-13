import pytest
import torch

from easy_tpp.config_factory import DataSpecConfig, ModelConfig
from easy_tpp.model import TorchIntensityFree, TorchNHP, TorchRMTPP
from easy_tpp.preprocess.dataset import TPPDataset, EventTokenizer, get_data_loader


def make_model_config(model_id, model_specs=None):
    config = ModelConfig.parse_from_yaml_config({
        'model_id': model_id,
        'hidden_size': 8,
        'num_event_types': 3,
        'num_event_types_pad': 4,
        'event_pad_index': 3,
        'gpu': -1,
        'model_specs': model_specs or {},
        'thinning': {
            'num_sample': 2,
            'num_exp': 4,
            'over_sample_rate': 2,
            'num_samples_boundary': 3,
            'dtime_max': 2,
            'patience_counter': 2,
            'num_step_gen': 2,
        },
    })
    config.set('mean_log_inter_time', 0.0)
    config.set('std_log_inter_time', 1.0)
    return config


def make_batch_dict():
    time_delta_seqs = torch.tensor([
        [0.0, 0.4, 0.6, 0.5, 0.7, 0.8],
        [0.0, 0.3, 0.2, 0.6, 0.4, 0.5],
    ])
    return {
        'time_seqs': torch.cumsum(time_delta_seqs, dim=-1),
        'time_delta_seqs': time_delta_seqs,
        'type_seqs': torch.tensor([
            [0, 1, 2, 0, 1, 2],
            [2, 1, 0, 2, 1, 0],
        ]),
        'seq_non_pad_mask': torch.ones(2, 6, dtype=torch.bool),
        'attention_mask': torch.zeros(2, 6, 6, dtype=torch.bool),
    }


def test_kwargs_dict_tuple_equivalence():
    torch.manual_seed(0)
    model = TorchNHP(make_model_config('NHP'))
    batch = make_batch_dict()

    kwargs_loss, kwargs_events = model.loglike_loss(**batch)
    dict_loss, dict_events = model.loglike_loss(batch)
    with pytest.warns(DeprecationWarning):
        tuple_loss, tuple_events = model.loglike_loss(tuple(batch.values()))

    assert torch.equal(kwargs_loss, dict_loss)
    assert torch.equal(kwargs_loss, tuple_loss)
    assert kwargs_events == dict_events == tuple_events


def test_alias_batch_non_pad_mask():
    torch.manual_seed(0)
    model = TorchIntensityFree(make_model_config(
        'IntensityFree', {'num_mix_components': 3}
    ))
    batch = make_batch_dict()
    alias_batch = dict(batch)
    alias_batch['batch_non_pad_mask'] = alias_batch.pop('seq_non_pad_mask')

    expected = model.loglike_loss(**batch)
    actual = model.loglike_loss(**alias_batch)

    assert torch.equal(expected[0], actual[0])
    assert expected[1] == actual[1]


def test_real_dataloader_batch():
    input_data = {
        'time_seqs': [[0.0, 0.2, 0.7], [0.0, 0.4]],
        'time_delta_seqs': [[0.0, 0.2, 0.5], [0.0, 0.4]],
        'type_seqs': [[0, 1, 2], [2, 0]],
    }
    data_config = DataSpecConfig.parse_from_yaml_config({
        'num_event_types': 3,
        'batch_size': 2,
        'pad_token_id': 3,
    })
    loader = get_data_loader(
        TPPDataset(input_data), 'torch', EventTokenizer(data_config), batch_size=2
    )
    batch = next(iter(loader))
    model = TorchRMTPP(make_model_config('RMTPP'))

    kwargs_result = model.loglike_loss(**batch)
    dict_result = model.loglike_loss(batch)

    assert torch.equal(kwargs_result[0], dict_result[0])
    assert kwargs_result[1] == dict_result[1]


def test_predict_paths_accept_kwargs():
    model = TorchRMTPP(make_model_config('RMTPP'))
    batch = make_batch_dict()
    legacy_batch = tuple(batch.values())

    with torch.no_grad():
        torch.manual_seed(0)
        kwargs_one_step = model.predict_one_step_at_every_event(**batch)
        torch.manual_seed(0)
        with pytest.warns(DeprecationWarning):
            legacy_one_step = model.predict_one_step_at_every_event(legacy_batch)

        torch.manual_seed(0)
        kwargs_multi_step = model.predict_multi_step_since_last_event(**batch)
        torch.manual_seed(0)
        with pytest.warns(DeprecationWarning):
            legacy_multi_step = model.predict_multi_step_since_last_event(legacy_batch)

    assert [value.shape for value in kwargs_one_step] == [
        value.shape for value in legacy_one_step
    ]
    assert [value.shape for value in kwargs_multi_step] == [
        value.shape for value in legacy_multi_step
    ]
