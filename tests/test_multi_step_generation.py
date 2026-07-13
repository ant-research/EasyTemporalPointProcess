import torch

from easy_tpp.config_factory import ModelConfig
from easy_tpp.model import IntensityFree, RMTPP


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


def make_padded_batch():
    time_delta_seqs = torch.tensor([
        [0.0, 0.4, 0.6, 0.5, 0.7, 0.8, 0.9, 1.0],
        [0.0, 0.3, 0.2, 0.6, 0.4, 0.0, 0.0, 0.0],
    ])
    time_seqs = torch.cumsum(time_delta_seqs, dim=-1)
    type_seqs = torch.tensor([
        [0, 1, 2, 0, 1, 2, 0, 1],
        [2, 1, 0, 2, 1, 3, 3, 3],
    ])
    batch_non_pad_mask = torch.tensor([
        [True, True, True, True, True, True, True, True],
        [True, True, True, True, True, False, False, False],
    ])
    attention_mask = torch.zeros(2, 8, 8, dtype=torch.bool)
    return time_seqs, time_delta_seqs, type_seqs, batch_non_pad_mask, attention_mask


def test_padded_batch_labels_are_real_events():
    model = RMTPP(make_model_config('RMTPP'))
    batch = make_padded_batch()

    _, _, label_dtimes, label_types = model.predict_multi_step_since_last_event(batch)

    assert torch.equal(label_dtimes[1], batch[1][1, 2:5])
    assert torch.equal(label_types[1], batch[2][1, 2:5])


def test_intensity_free_multi_step_generation():
    torch.manual_seed(0)
    model = IntensityFree(make_model_config(
        'IntensityFree',
        model_specs={'num_mix_components': 3},
    ))
    batch = make_padded_batch()
    sample_dtimes = torch.rand(2, 8, 5) + 0.01

    intensities = model.compute_intensities_at_sample_times(
        batch[0], batch[1], batch[2], sample_dtimes
    )

    assert intensities.shape == (2, 8, 5, 4)
    assert torch.isfinite(intensities).all()
    assert (intensities > 0).all()

    outputs = model.predict_multi_step_since_last_event(batch)
    for output in outputs:
        assert output.shape == (2, 3)
        assert torch.isfinite(output).all()
