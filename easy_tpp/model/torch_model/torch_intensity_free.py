import torch
import torch.distributions as D
from torch import nn
from torch.distributions import Categorical, TransformedDistribution
from torch.distributions import MixtureSameFamily as TorchMixtureSameFamily
from torch.distributions import Normal as TorchNormal

from easy_tpp.model.torch_model.torch_basemodel import TorchBaseModel


def clamp_preserve_gradients(x, min_val, max_val):
    """Clamp the tensor while preserving gradients in the clamped region.

    Args:
        x (tensor): tensor to be clamped.
        min_val (float): minimum value.
        max_val (float): maximum value.
    """
    return x + (x.clamp(min_val, max_val) - x).detach()


class Normal(TorchNormal):
    """Normal distribution, redefined `log_cdf` and `log_survival_function` due to
    no numerically stable implementation of them is available for normal distribution.
    """

    def log_cdf(self, x):
        cdf = clamp_preserve_gradients(self.cdf(x), 1e-7, 1 - 1e-7)
        return cdf.log()

    def log_survival_function(self, x):
        cdf = clamp_preserve_gradients(self.cdf(x), 1e-7, 1 - 1e-7)
        return torch.log(1.0 - cdf)


class MixtureSameFamily(TorchMixtureSameFamily):
    """Mixture (same-family) distribution, redefined `log_cdf` and `log_survival_function`.
    """

    def log_cdf(self, x):
        x = self._pad(x)
        log_cdf_x = self.component_distribution.log_cdf(x)
        mix_logits = self.mixture_distribution.logits
        return torch.logsumexp(log_cdf_x + mix_logits, dim=-1)

    def log_survival_function(self, x):
        x = self._pad(x)
        log_sf_x = self.component_distribution.log_survival_function(x)
        mix_logits = self.mixture_distribution.logits
        return torch.logsumexp(log_sf_x + mix_logits, dim=-1)


class LogNormalMixtureDistribution(TransformedDistribution):
    """
    Mixture of log-normal distributions.

    Args:
        locs (tensor): [batch_size, seq_len, num_mix_components].
        log_scales (tensor): [batch_size, seq_len, num_mix_components].
        log_weights (tensor): [batch_size, seq_len, num_mix_components].
        mean_log_inter_time (float): Average log-inter-event-time.
        std_log_inter_time (float): Std of log-inter-event-times.
    """

    def __init__(self, locs, log_scales, log_weights, mean_log_inter_time, std_log_inter_time, validate_args=None):
        mixture_dist = D.Categorical(logits=log_weights)
        component_dist = Normal(loc=locs, scale=log_scales.exp())
        GMM = MixtureSameFamily(mixture_dist, component_dist)
        if mean_log_inter_time == 0.0 and std_log_inter_time == 1.0:
            transforms = []
        else:
            transforms = [D.AffineTransform(loc=mean_log_inter_time, scale=std_log_inter_time)]
        self.mean_log_inter_time = mean_log_inter_time
        self.std_log_inter_time = std_log_inter_time
        transforms.append(D.ExpTransform())

        self.transforms = transforms
        sign = 1
        for transform in self.transforms:
            sign = sign * transform.sign
        self.sign = int(sign)
        super().__init__(GMM, transforms, validate_args=validate_args)

    def log_cdf(self, x):
        for transform in self.transforms[::-1]:
            x = transform.inv(x)
        if self._validate_args:
            self.base_dist._validate_sample(x)

        if self.sign == 1:
            return self.base_dist.log_cdf(x)
        else:
            return self.base_dist.log_survival_function(x)

    def log_survival_function(self, x):
        for transform in self.transforms[::-1]:
            x = transform.inv(x)
        if self._validate_args:
            self.base_dist._validate_sample(x)

        if self.sign == 1:
            return self.base_dist.log_survival_function(x)
        else:
            return self.base_dist.log_cdf(x)


class IntensityFree(TorchBaseModel):
    """Torch implementation of Intensity-Free Learning of Temporal Point Processes, ICLR 2020.
    https://openreview.net/pdf?id=HygOjhEYDH

    reference: https://github.com/shchur/ifl-tpp
    """

    def __init__(self, model_config):
        """Initialize the model

        Args:
            model_config (EasyTPP.ModelConfig): config of model specs.

        """
        super(IntensityFree, self).__init__(model_config)

        self.num_mix_components = model_config.model_specs['num_mix_components']
        self.mean_log_inter_time = model_config.get("mean_log_inter_time", 0.0)
        self.std_log_inter_time = model_config.get("std_log_inter_time", 1.0)

        self.num_features = 1 + self.hidden_size

        self.layer_rnn = nn.GRU(input_size=self.num_features,
                                hidden_size=self.hidden_size,
                                num_layers=1,  # used in original paper
                                batch_first=True)

        self.mark_linear = nn.Linear(self.hidden_size, self.num_event_types_pad)
        self.linear = nn.Linear(self.hidden_size, 3 * self.num_mix_components)

    def forward(self, time_delta_seqs, type_seqs):
        """Call the model.

        Args:
            time_delta_seqs (tensor): [batch_size, seq_len], inter-event time seqs.
            type_seqs (tensor): [batch_size, seq_len], event type seqs.

        Returns:
            list: hidden states, [batch_size, seq_len, hidden_dim], states right before the event happens.
        """
        # [batch_size, seq_len, hidden_size]
        # We dont normalize inter-event time here
        temporal_seqs = torch.log(time_delta_seqs + self.eps).unsqueeze(-1)

        # [batch_size, seq_len, hidden_size]
        type_emb = self.layer_type_emb(type_seqs)

        # [batch_size, seq_len, hidden_size + 1]
        rnn_input = torch.cat([temporal_seqs, type_emb], dim=-1)

        # [batch_size, seq_len, hidden_size]
        context = self.layer_rnn(rnn_input)[0]

        return context

    def loglike_loss(self, batch):
        """Compute the loglike loss.

        Args:
            batch (list): batch input.

        Returns:
            tuple: loglikelihood loss and num of events.
        """
        time_seqs, time_delta_seqs, type_seqs, batch_non_pad_mask, _ = batch

        # [batch_size, seq_len, hidden_size]
        context = self.forward(time_delta_seqs[:, :-1], type_seqs[:, :-1])

        # [batch_size, seq_len, 3 * num_mix_components]
        raw_params = self.linear(context)
        locs = raw_params[..., :self.num_mix_components]
        log_scales = raw_params[..., self.num_mix_components: (2 * self.num_mix_components)]
        log_weights = raw_params[..., (2 * self.num_mix_components):]

        log_scales = clamp_preserve_gradients(log_scales, -5.0, 3.0)
        log_weights = torch.log_softmax(log_weights, dim=-1)
        inter_time_dist = LogNormalMixtureDistribution(
            locs=locs,
            log_scales=log_scales,
            log_weights=log_weights,
            mean_log_inter_time=self.mean_log_inter_time,
            std_log_inter_time=self.std_log_inter_time
        )

        inter_times = time_delta_seqs[:, 1:].clamp(min=1e-5)
        # [batch_size, seq_len]
        event_mask = torch.logical_and(batch_non_pad_mask[:, 1:], type_seqs[:, 1:] != self.pad_token_id)
        time_ll = inter_time_dist.log_prob(inter_times) * event_mask

        # [batch_size, seq_len, num_marks]
        mark_logits = torch.log_softmax(self.mark_linear(context), dim=-1)
        mark_dist = Categorical(logits=mark_logits)
        mark_ll = mark_dist.log_prob(type_seqs[:, 1:]) * event_mask

        log_p = time_ll + mark_ll

        # [batch_size,]
        loss = -log_p.sum()

        num_events = event_mask.sum().item()

        return loss, num_events

    def predict_one_step_at_every_event(self, batch):
        """One-step prediction for every event in the sequence.

        Args:
            time_seqs (tensor): [batch_size, seq_len].
            time_delta_seqs (tensor): [batch_size, seq_len].
            type_seqs (tensor): [batch_size, seq_len].

        Returns:
            tuple: tensors of dtime and type prediction, [batch_size, seq_len].
        """
        time_seq, time_delta_seq, event_seq, batch_non_pad_mask, _ = batch

        # remove the last event, as the prediction based on the last event has no label
        # time_delta_seq should start from 1, because the first one is zero
        time_seq, time_delta_seq, event_seq = time_seq[:, :-1], time_delta_seq[:, :-1], event_seq[:, :-1]

        # [batch_size, seq_len, hidden_size]
        context = self.forward(time_delta_seq, event_seq)

        # [batch_size, seq_len, 3 * num_mix_components]
        raw_params = self.linear(context)
        locs = raw_params[..., :self.num_mix_components]
        log_scales = raw_params[..., self.num_mix_components: (2 * self.num_mix_components)]
        log_weights = raw_params[..., (2 * self.num_mix_components):]

        log_scales = clamp_preserve_gradients(log_scales, -5.0, 3.0)
        log_weights = torch.log_softmax(log_weights, dim=-1)
        inter_time_dist = LogNormalMixtureDistribution(
            locs=locs,
            log_scales=log_scales,
            log_weights=log_weights,
            mean_log_inter_time=self.mean_log_inter_time,
            std_log_inter_time=self.std_log_inter_time
        )

        # [num_samples, batch_size, seq_len]
        accepted_dtimes = inter_time_dist.sample((self.event_sampler.num_sample,))
        dtimes_pred = accepted_dtimes.mean(dim=0)

        # [batch_size, seq_len, num_marks]
        mark_logits = torch.log_softmax(self.mark_linear(context), dim=-1)  # Marks are modeled conditionally independently from times
        types_pred = torch.argmax(mark_logits, dim=-1)
        return dtimes_pred, types_pred
