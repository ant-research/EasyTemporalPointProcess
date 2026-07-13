import torch

from easy_tpp.config_factory import ModelConfig
from easy_tpp.model import ANHN


def make_model_config():
    return ModelConfig.parse_from_yaml_config({
        'model_id': 'ANHN',
        'hidden_size': 8,
        'time_emb_size': 8,
        'num_layers': 1,
        'num_heads': 1,
        'num_event_types': 3,
        'num_event_types_pad': 4,
        'event_pad_index': 3,
        'gpu': -1,
        'loss_integral_num_sample_per_step': 5,
    })


def make_batch():
    time_delta_seqs = torch.tensor([
        [0.0, 0.4, 0.6, 0.5, 0.7],
        [0.0, 0.3, 0.2, 0.6, 0.4],
    ])
    time_seqs = torch.cumsum(time_delta_seqs, dim=-1)
    type_seqs = torch.tensor([
        [0, 1, 2, 0, 1],
        [2, 1, 0, 2, 1],
    ], dtype=torch.long)
    seq_non_pad_mask = torch.ones_like(type_seqs, dtype=torch.bool)
    attention_mask = torch.triu(torch.ones(2, 5, 5, dtype=torch.bool), diagonal=1)
    return {
        'time_seqs': time_seqs,
        'time_delta_seqs': time_delta_seqs,
        'type_seqs': type_seqs,
        'seq_non_pad_mask': seq_non_pad_mask,
        'attention_mask': attention_mask,
    }


def test_construction():
    ANHN(make_model_config())


def test_loglike_loss_runs():
    model = ANHN(make_model_config())

    loss, num_events = model.loglike_loss(**make_batch())

    assert torch.isfinite(loss)
    assert num_events > 0
    assert loss.requires_grad
    loss.backward()


def test_sample_states_shape():
    model = ANHN(make_model_config())
    batch = make_batch()
    dtime_seqs = batch['time_delta_seqs'][:, 1:]
    type_seqs = batch['type_seqs'][:, :-1]
    attention_mask = batch['attention_mask'][:, 1:, :-1]

    _, (intensity_base, intensity_alpha, intensity_delta), (base_dtime, _) = model.forward(
        dtime_seqs, type_seqs, attention_mask
    )
    sample_dtimes = model.make_dtime_loss_samples(dtime_seqs)
    states = model.compute_states_at_sample_times(
        intensity_base, intensity_alpha, intensity_delta, base_dtime, sample_dtimes
    )

    assert states.ndim == 4
    assert states.shape == (2, 4, 5, 8)
