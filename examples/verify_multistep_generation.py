"""Verification script for the issue #13 fixes:
1. padding bug in predict_multi_step_since_last_event
2. IntensityFree.compute_intensities_at_sample_times (closed-form hazard)
"""
import torch

from easy_tpp.config_factory import ModelConfig
from easy_tpp.model import TorchIntensityFree, TorchRMTPP

PAD = 3


def make_model_config(model_id, model_specs=None, num_step_gen=2):
    config = ModelConfig.parse_from_yaml_config({
        'model_id': model_id,
        'hidden_size': 8,
        'num_event_types': 3,
        'num_event_types_pad': 4,
        'event_pad_index': PAD,
        'gpu': -1,
        'model_specs': model_specs or {},
        'thinning': {
            'num_sample': 5,
            'num_exp': 100,
            'over_sample_rate': 1.5,
            'num_samples_boundary': 5,
            'dtime_max': 5,
            'patience_counter': 5,
            'num_step_gen': num_step_gen,
        },
    })
    config.set('mean_log_inter_time', 0.0)
    config.set('std_log_inter_time', 1.0)
    return config


def make_padded_batch():
    """Two right-padded sequences: true lengths 8 and 5 (padded to 8).

    Pads mimic the real tokenizer: type = pad_token_id, times = pad value.
    """
    time_delta_seqs = torch.tensor([
        [0.0, 0.4, 0.6, 0.5, 0.7, 0.8, 0.9, 1.0],
        [0.0, 0.3, 0.2, 0.6, 0.4, float(PAD), float(PAD), float(PAD)],
    ])
    time_seqs = torch.tensor([
        [0.0, 0.4, 1.0, 1.5, 2.2, 3.0, 3.9, 4.9],
        [0.0, 0.3, 0.5, 1.1, 1.5, float(PAD), float(PAD), float(PAD)],
    ])
    type_seqs = torch.tensor([
        [0, 1, 2, 0, 1, 2, 0, 1],
        [2, 1, 0, 2, 1, PAD, PAD, PAD],
    ])
    batch_non_pad_mask = torch.tensor([
        [True] * 8,
        [True] * 5 + [False] * 3,
    ])
    attention_mask = torch.zeros(2, 8, 8, dtype=torch.bool)
    return time_seqs, time_delta_seqs, type_seqs, batch_non_pad_mask, attention_mask


def predict_multi_step_old(model, batch, forward=False):
    """Verbatim copy of the PRE-FIX predict_multi_step_since_last_event."""
    time_seq_label, time_delta_seq_label, event_seq_label, _, _ = batch
    num_step = model.gen_config.num_step_gen
    if not forward:
        time_seq = time_seq_label[:, :-num_step]
        time_delta_seq = time_delta_seq_label[:, :-num_step]
        event_seq = event_seq_label[:, :-num_step]
    else:
        time_seq, time_delta_seq, event_seq = time_seq_label, time_delta_seq_label, event_seq_label
    for _ in range(num_step):
        dtime_boundary = time_delta_seq + model.event_sampler.dtime_max
        accepted_dtimes, weights = model.event_sampler.draw_next_time_one_step(
            time_seq, time_delta_seq, event_seq, dtime_boundary,
            model.compute_intensities_at_sample_times, compute_last_step_only=True)
        dtimes_pred = torch.sum(accepted_dtimes * weights, dim=-1)
        intensities_at_times = model.compute_intensities_at_sample_times(
            time_seq, time_delta_seq, event_seq, dtimes_pred[:, :, None],
            max_steps=event_seq.size()[1])
        intensities_at_times = intensities_at_times.squeeze(dim=-2)
        types_pred = torch.argmax(intensities_at_times, dim=-1)
        types_pred_, dtimes_pred_ = types_pred[:, -1:], dtimes_pred[:, -1:]
        time_pred_ = time_seq[:, -1:] + dtimes_pred_
        time_seq = torch.cat([time_seq, time_pred_], dim=-1)
        time_delta_seq = torch.cat([time_delta_seq, dtimes_pred_], dim=-1)
        event_seq = torch.cat([event_seq, types_pred_], dim=-1)
    return (time_delta_seq[:, -num_step - 1:], event_seq[:, -num_step - 1:],
            time_delta_seq_label[:, -num_step - 1:], event_seq_label[:, -num_step - 1:])


def check_padding_bug():
    print('=' * 70)
    print('1. Padding bug in multi-step generation (short row, true length 5)')
    print('=' * 70)
    torch.manual_seed(0)
    model = TorchRMTPP(make_model_config('RMTPP'))
    batch = make_padded_batch()
    with torch.no_grad():
        old = predict_multi_step_old(model, batch)
        new = model.predict_multi_step_since_last_event(batch)

    print(f'true last 3 real deltas of short row : {batch[1][1, 2:5].tolist()}')
    print(f'true last 3 real types  of short row : {batch[2][1, 2:5].tolist()}')
    print(f'OLD label deltas (pad values!)       : {old[2][1].tolist()}')
    print(f'OLD label types  (pad values!)       : {old[3][1].tolist()}')
    print(f'NEW label deltas                     : {new[2][1].tolist()}')
    print(f'NEW label types                      : {new[3][1].tolist()}')

    assert torch.equal(new[2][1], batch[1][1, 2:5]), 'new label deltas wrong'
    assert torch.equal(new[3][1], batch[2][1, 2:5]), 'new label types wrong'
    assert torch.all(old[3][1] == PAD), 'expected old labels to be pads'
    print('PASS: new code returns real events; old code returned pads.\n')


def check_intensity_free_hazard():
    print('=' * 70)
    print('2. IntensityFree closed-form hazard lambda(t) = f(t) / S(t)')
    print('=' * 70)
    torch.manual_seed(0)
    model = TorchIntensityFree(make_model_config(
        'IntensityFree', model_specs={'num_mix_components': 1}))
    batch = make_padded_batch()
    time_seqs, time_delta_seqs, type_seqs = batch[0], batch[1], batch[2]
    sample_dtimes = torch.linspace(0.1, 3.0, 20)[None, None, :].expand(2, 8, 20)

    with torch.no_grad():
        lambdas = model.compute_intensities_at_sample_times(
            time_seqs, time_delta_seqs, type_seqs, sample_dtimes)

        # closed form with a single mixture component: tau ~ LogNormal(loc, scale)
        context = model.forward(time_delta_seqs, type_seqs)
        raw = model.linear(context)
        loc, log_scale = raw[..., 0], raw[..., 1].clamp(-5.0, 3.0)
        ln = torch.distributions.LogNormal(loc[..., None], log_scale.exp()[..., None])
        f = ln.log_prob(sample_dtimes).exp()
        S = 1.0 - ln.cdf(sample_dtimes)
        hazard = f / S
        mark_probs = torch.softmax(model.mark_linear(context), dim=-1)
        expected = hazard[..., None] * mark_probs[:, :, None, :]

    err = (lambdas - expected).abs().max().item()
    print(f'shape: {tuple(lambdas.shape)} (batch, seq, samples, marks)')
    print(f'all positive: {bool((lambdas > 0).all())}, all finite: {bool(torch.isfinite(lambdas).all())}')
    print(f'max |method - closed form| = {err:.2e}')
    assert err < 1e-4, 'hazard mismatch'
    print('PASS: implemented intensity matches the closed-form lognormal hazard.\n')


def check_end_to_end_generation():
    print('=' * 70)
    print('3. End-to-end multi-step generation (thinning) incl. IntensityFree')
    print('=' * 70)
    batch = make_padded_batch()
    for name, model in [
        ('RMTPP', TorchRMTPP(make_model_config('RMTPP'))),
        ('IntensityFree', TorchIntensityFree(make_model_config(
            'IntensityFree', model_specs={'num_mix_components': 3}))),
    ]:
        torch.manual_seed(0)
        with torch.no_grad():
            pred_dt, pred_ty, lab_dt, lab_ty = model.predict_multi_step_since_last_event(batch)
        ok = all(torch.isfinite(t.float()).all() for t in (pred_dt, pred_ty, lab_dt, lab_ty))
        print(f'{name:>14}: generated dtimes {pred_dt.shape} -> {pred_dt[1].tolist()} finite={ok}')
        assert ok and pred_dt.shape == (2, 3)
    print('PASS: both models generate finite multi-step predictions.\n')


if __name__ == '__main__':
    check_padding_bug()
    check_intensity_free_hazard()
    check_end_to_end_generation()
    print('All checks passed.')
