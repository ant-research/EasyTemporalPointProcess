import torch

from easy_tpp.config_factory.model_config import ModelConfig
from easy_tpp.model import BaseModel, ITHP


def make_config(integration_method='mc'):
    return ModelConfig(
        model_id='ITHP',
        hidden_size=4,
        num_layers=1,
        num_event_types=3,
        num_event_types_pad=4,
        event_pad_index=3,
        gpu=-1,
        training=True,
        loss_integral_num_sample_per_step=5,
        model_specs={
            'd_inner': 12,
            'd_k': 4,
            'd_v': 8,
            'integration_method': integration_method,
            'grid_step': 0.2,
            'query_chunk_size': 5,
            'use_type_loss': True,
            'type_loss_weight': 0.5,
        },
    )


def make_batch(padded=False):
    times = torch.tensor([[0.0, 0.4, 1.0]])
    dtimes = torch.tensor([[0.0, 0.4, 0.6]])
    types = torch.tensor([[0, 1, 2]])
    mask = torch.tensor([[True, True, True]])
    if padded:
        times = torch.cat((times, times[:, -1:].expand(-1, 2)), dim=1)
        dtimes = torch.cat((dtimes, torch.zeros(1, 2)), dim=1)
        types = torch.cat((types, torch.full((1, 2), 3)), dim=1)
        mask = torch.cat((mask, torch.zeros(1, 2, dtype=torch.bool)), dim=1)
    return {
        'time_seqs': times,
        'time_delta_seqs': dtimes,
        'type_seqs': types,
        'seq_non_pad_mask': mask,
        'attention_mask': torch.zeros((1, times.size(1), times.size(1)), dtype=torch.bool),
    }


def test_registration_named_batch_and_gradients():
    model = BaseModel.generate_model_from_config(make_config())
    assert isinstance(model, ITHP)
    torch.manual_seed(7)
    loss, count = model.loglike_loss(**make_batch())
    assert count == 2
    assert torch.isfinite(loss)
    loss.backward()
    assert model.encoder_layer.self_attention.w_vs.weight.grad is not None
    assert torch.isfinite(model.encoder_layer.self_attention.w_vs.weight.grad).all()


def test_mc_is_random_and_trapezoid_is_fixed():
    dtimes = torch.ones(1, 2)
    mc_model = ITHP(make_config('mc'))
    first = mc_model.make_dtime_loss_samples(dtimes)
    second = mc_model.make_dtime_loss_samples(dtimes)
    assert not torch.equal(first, second)
    assert ((first >= 0) & (first <= 1)).all()

    trapezoid_model = ITHP(make_config('trapezoid'))
    expected = torch.linspace(0, 1, 5).expand(1, 2, -1)
    assert torch.equal(trapezoid_model.make_dtime_loss_samples(dtimes), expected)


def test_padding_does_not_change_valid_likelihood():
    model = ITHP(make_config())
    model.eval()
    torch.manual_seed(17)
    plain_loss, plain_count = model.loglike_loss(**make_batch())
    torch.manual_seed(17)
    padded_loss, padded_count = model.loglike_loss(**make_batch(padded=True))
    assert plain_count == padded_count == 2
    torch.testing.assert_close(plain_loss, padded_loss, rtol=0, atol=1e-5)


def test_future_event_does_not_change_earlier_intensity():
    model = ITHP(make_config())
    model.eval()
    batch = make_batch()
    baseline = model(batch['time_seqs'], batch['type_seqs'], batch['seq_non_pad_mask'])
    changed_times = batch['time_seqs'].clone()
    changed_types = batch['type_seqs'].clone()
    changed_times[:, -1] = 9.0
    changed_types[:, -1] = 0
    changed = model(changed_times, changed_types, batch['seq_non_pad_mask'])
    torch.testing.assert_close(baseline[:, 0], changed[:, 0])


def test_auxiliary_type_loss_applies_only_during_training():
    model = ITHP(make_config('fixed_grid'))
    model.train()
    train_loss, _ = model.loglike_loss(**make_batch())
    model.eval()
    valid_loss, _ = model.loglike_loss(**make_batch())
    assert train_loss > valid_loss


def test_integration_modes_and_intensity_shape():
    batch = make_batch()
    for method in ('mc', 'trapezoid', 'fixed_grid'):
        model = ITHP(make_config(method))
        model.eval()
        loss, count = model.loglike_loss(**batch)
        assert count == 2
        assert torch.isfinite(loss)
        intensities = model.compute_intensities_at_sample_times(
            time_seqs=batch['time_seqs'][:, :-1],
            time_delta_seqs=batch['time_delta_seqs'][:, :-1],
            type_seqs=batch['type_seqs'][:, :-1],
            sample_dtimes=torch.full((1, 2, 3), 0.1),
        )
        assert intensities.shape == (1, 2, 3, 3)
        assert (intensities > 0).all()


def test_generated_config_reconstructs_checkpoint_shape():
    config = make_config('trapezoid')
    original = ITHP(config)
    restored_config = ModelConfig.parse_from_yaml_config(config.get_yaml_config())
    restored = ITHP(restored_config)
    restored.load_state_dict(original.state_dict(), strict=True)
    assert restored.integration_method == 'trapezoid'
    assert restored.use_mc_samples is False
