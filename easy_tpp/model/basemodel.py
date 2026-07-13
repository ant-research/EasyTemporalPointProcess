""" Base model with common functionality  """

import warnings
from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from easy_tpp.model.thinning import EventSampler
from easy_tpp.utils import set_device


class BaseModel(nn.Module):
    def __init__(self, model_config):
        """Initialize the BaseModel

        Args:
            model_config (EasyTPP.ModelConfig): model spec of configs
        """
        super(BaseModel, self).__init__()
        self.loss_integral_num_sample_per_step = model_config.loss_integral_num_sample_per_step
        self.hidden_size = model_config.hidden_size
        self.num_event_types = model_config.num_event_types  # not include [PAD], [BOS], [EOS]
        self.num_event_types_pad = model_config.num_event_types_pad  # include [PAD], [BOS], [EOS]
        self.pad_token_id = model_config.pad_token_id
        self.eps = torch.finfo(torch.float32).eps

        self.layer_type_emb = nn.Embedding(self.num_event_types_pad,  # have padding
                                           self.hidden_size,
                                           padding_idx=self.pad_token_id)

        self.gen_config = model_config.thinning
        self.event_sampler = None
        self.device = set_device(model_config.gpu)
        self.use_mc_samples = model_config.use_mc_samples

        self.to(self.device)

        if self.gen_config:
            self.event_sampler = EventSampler(num_sample=self.gen_config.num_sample,
                                              num_exp=self.gen_config.num_exp,
                                              over_sample_rate=self.gen_config.over_sample_rate,
                                              patience_counter=self.gen_config.patience_counter,
                                              num_samples_boundary=self.gen_config.num_samples_boundary,
                                              dtime_max=self.gen_config.dtime_max,
                                              device=self.device)

    @staticmethod
    def generate_model_from_config(model_config):
        """Generate the model in derived class based on model config.

        Args:
            model_config (EasyTPP.ModelConfig): config of model specs.
        """
        model_id = model_config.model_id

        for subclass in BaseModel.__subclasses__():
            if subclass.__name__ == model_id:
                return subclass(model_config)

        raise RuntimeError('No model named ' + model_id)

    @staticmethod
    def resolve_batch_inputs(batch=None, kwargs=None):
        """Normalize model inputs.

        Returns (time_seqs, time_delta_seqs, type_seqs, seq_non_pad_mask,
        attention_mask). Accepts HF-style keyword arguments; a dict or
        BatchEncoding; or the legacy positional tuple/list (deprecated).
        """
        kwargs = kwargs or {}
        input_names = ('time_seqs', 'time_delta_seqs', 'type_seqs',
                       'seq_non_pad_mask', 'attention_mask')

        if batch is None or isinstance(batch, Mapping):
            inputs = kwargs if batch is None else batch
            values = [inputs.get(name) for name in input_names]
            if values[3] is None:
                values[3] = inputs.get('batch_non_pad_mask')
            return tuple(values)

        warnings.warn(
            'Passing a positional batch to a model is deprecated; pass keyword inputs '
            '(model.loglike_loss(**batch)) instead.',
            DeprecationWarning,
            stacklevel=2,
        )
        values = tuple(batch)
        return (values + (None,) * len(input_names))[:len(input_names)]

    @staticmethod
    def get_logits_at_last_step(logits, batch_non_pad_mask, sample_len=None):
        """Retrieve the hidden states of last non-pad events.

        Args:
            logits (tensor): [batch_size, seq_len, hidden_dim], a sequence of logits
            batch_non_pad_mask (tensor): [batch_size, seq_len], a sequence of masks
            sample_len (tensor): default None, use batch_non_pad_mask to find out the last non-mask position

        ref: https://medium.com/analytics-vidhya/understanding-indexing-with-pytorch-gather-33717a84ebc4

        Returns:
            tensor: retrieve the logits of EOS event
        """

        seq_len = batch_non_pad_mask.sum(dim=1)
        select_index = seq_len - 1 if sample_len is None else seq_len - 1 - sample_len
        # [batch_size, hidden_dim]
        select_index = select_index.unsqueeze(1).repeat(1, logits.size(-1))
        # [batch_size, 1, hidden_dim]
        select_index = select_index.unsqueeze(1)
        # [batch_size, hidden_dim]
        last_logits = torch.gather(logits, dim=1, index=select_index).squeeze(1)
        return last_logits

    def compute_loglikelihood(self, time_delta_seq, lambda_at_event, lambdas_loss_samples, seq_mask, type_seq):
        """Compute the loglikelihood of the event sequence based on Equation (8) of NHP paper.

        Args:
            time_delta_seq (tensor): [batch_size, seq_len], time_delta_seq from model input.
            lambda_at_event (tensor): [batch_size, seq_len, num_event_types], unmasked intensity at
            (right after) the event.
            lambdas_loss_samples (tensor): [batch_size, seq_len, num_sample, num_event_types],
            intensity at sampling times.
            seq_mask (tensor): [batch_size, seq_len], sequence mask vector to mask the padded events.
            type_seq (tensor): [batch_size, seq_len], sequence of mark ids, with padded events having a mark of self.pad_token_id

        Returns:
            tuple: event loglike, non-event loglike, intensity at event with padding events masked
        """

        # First, add an epsilon to every marked intensity for stability
        lambda_at_event = lambda_at_event + self.eps
        lambdas_loss_samples = lambdas_loss_samples + self.eps

        log_marked_event_lambdas = lambda_at_event.log()
        total_sampled_lambdas = lambdas_loss_samples.sum(dim=-1)

        # Compute event LL - [batch_size, seq_len]
        event_ll = -F.nll_loss(
            log_marked_event_lambdas.permute(0, 2, 1),  # mark dimension needs to come second, not third to match nll_loss specs
            target=type_seq,
            ignore_index=self.pad_token_id,  # Padded events have a pad_token_id as a value
            reduction='none', # Does not aggregate, and replaces what would have been the log(marked intensity) with 0.
        )

        # Compute non-event LL [batch_size, seq_len]
        # interval_integral = length_interval * average of sampled lambda(t)
        if self.use_mc_samples:
            non_event_ll = total_sampled_lambdas.mean(dim=-1) * time_delta_seq * seq_mask
        else: # Use trapezoid rule
            non_event_ll = 0.5 * (total_sampled_lambdas[..., 1:] + total_sampled_lambdas[..., :-1]).mean(dim=-1) * time_delta_seq * seq_mask

        num_events = torch.masked_select(event_ll, event_ll.ne(0.0)).size()[0]
        return event_ll, non_event_ll, num_events

    def make_dtime_loss_samples(self, time_delta_seq):
        """Generate the time point samples for every interval.

        Args:
            time_delta_seq (tensor): [batch_size, seq_len].

        Returns:
            tensor: [batch_size, seq_len, n_samples]
        """
        # [1, 1, n_samples]
        dtimes_ratio_sampled = torch.linspace(start=0.0,
                                              end=1.0,
                                              steps=self.loss_integral_num_sample_per_step,
                                              device=self.device)[None, None, :]

        # [batch_size, max_len, n_samples]
        sampled_dtimes = time_delta_seq[:, :, None] * dtimes_ratio_sampled

        return sampled_dtimes

    def compute_states_at_sample_times(self, **kwargs):
        raise NotImplementedError('This need to implemented in inherited class ! ')

    def predict_one_step_at_every_event(self, batch=None, **kwargs):
        """One-step prediction for every event in the sequence.

        Args:
            batch (Mapping or tuple, optional): Legacy batch input. Prefer passing
                time_seqs, time_delta_seqs, type_seqs, seq_non_pad_mask, and
                attention_mask as keyword inputs.
            **kwargs: HF-style keyword model inputs.

        Returns:
            tuple: tensors of dtime and type prediction, [batch_size, seq_len].
        """
        time_seq, time_delta_seq, event_seq, batch_non_pad_mask, _ = \
            self.resolve_batch_inputs(batch, kwargs)

        # remove the last event, as the prediction based on the last event has no label
        # note: the first dts is 0
        # [batch_size, seq_len]
        time_seq, time_delta_seq, event_seq = time_seq[:, :-1], time_delta_seq[:, :-1], event_seq[:, :-1]

        # [batch_size, seq_len]
        dtime_boundary = torch.max(time_delta_seq * self.event_sampler.dtime_max,
                                   time_delta_seq + self.event_sampler.dtime_max)

        # [batch_size, seq_len, num_sample]
        accepted_dtimes, weights = self.event_sampler.draw_next_time_one_step(time_seq,
                                                                              time_delta_seq,
                                                                              event_seq,
                                                                              dtime_boundary,
                                                                              self.compute_intensities_at_sample_times,
                                                                              compute_last_step_only=False)  # make it explicit

        # We should condition on each accepted time to sample event mark, but not conditioned on the expected event time.
        # 1. Use all accepted_dtimes to get intensity.
        # [batch_size, seq_len, num_sample, num_marks]
        intensities_at_times = self.compute_intensities_at_sample_times(time_seq,
                                                                        time_delta_seq,
                                                                        event_seq,
                                                                        accepted_dtimes)

        # 2. Normalize the intensity over last dim and then compute the weighted sum over the `num_sample` dimension.
        # Each of the last dimension is a categorical distribution over all marks.
        # [batch_size, seq_len, num_sample, num_marks]
        intensities_normalized = intensities_at_times / intensities_at_times.sum(dim=-1, keepdim=True)

        # 3. Compute weighted sum of distributions and then take argmax.
        # [batch_size, seq_len, num_marks]
        intensities_weighted = torch.einsum('...s,...sm->...m', weights, intensities_normalized)

        # [batch_size, seq_len]
        types_pred = torch.argmax(intensities_weighted, dim=-1)

        # [batch_size, seq_len]
        dtimes_pred = torch.sum(accepted_dtimes * weights, dim=-1)  # compute the expected next event time
        return dtimes_pred, types_pred

    def predict_multi_step_since_last_event(self, batch=None, forward=False, **kwargs):
        """Multi-step prediction since last event in the sequence.

        Args:
            batch (Mapping or tuple, optional): Legacy batch input. Prefer passing
                time_seqs, time_delta_seqs, type_seqs, seq_non_pad_mask, and
                attention_mask as keyword inputs.
            forward (bool, optional): Whether to use the entire sequence for prediction. Defaults to False.
            **kwargs: HF-style keyword model inputs.

        Returns:
            tuple: tensors of dtime and type prediction, [batch_size, seq_len].
        """
        time_seq_label, time_delta_seq_label, event_seq_label, batch_non_pad_mask, _ = \
            self.resolve_batch_inputs(batch, kwargs)
        if batch_non_pad_mask is None:
            raise ValueError('seq_non_pad_mask is required for multi-step prediction.')

        num_step = self.gen_config.num_step_gen
        true_lengths = batch_non_pad_mask.sum(dim=-1)
        if not forward and torch.any(true_lengths <= num_step):
            raise ValueError('All sequences must contain more events than num_step_gen for multi-step prediction.')

        batch_size = time_seq_label.size(0)
        output_shape = (batch_size, num_step + 1)
        pred_dtimes = time_delta_seq_label.new_empty(output_shape)
        pred_types = event_seq_label.new_empty(output_shape)
        label_dtimes = time_delta_seq_label.new_empty(output_shape)
        label_types = event_seq_label.new_empty(output_shape)

        for true_length in torch.unique(true_lengths):
            length = int(true_length.item())
            row_indices = torch.nonzero(true_lengths == true_length, as_tuple=False).squeeze(-1)

            time_seq_group = time_seq_label.index_select(0, row_indices)[:, :length]
            time_delta_seq_group = time_delta_seq_label.index_select(0, row_indices)[:, :length]
            event_seq_group = event_seq_label.index_select(0, row_indices)[:, :length]

            if not forward:
                time_seq = time_seq_group[:, :length - num_step]
                time_delta_seq = time_delta_seq_group[:, :length - num_step]
                event_seq = event_seq_group[:, :length - num_step]
            else:
                time_seq, time_delta_seq, event_seq = time_seq_group, time_delta_seq_group, event_seq_group

            for i in range(num_step):
                # [batch_size, seq_len]
                dtime_boundary = time_delta_seq + self.event_sampler.dtime_max

                # [batch_size, 1, num_sample]
                accepted_dtimes, weights = \
                    self.event_sampler.draw_next_time_one_step(time_seq,
                                                               time_delta_seq,
                                                               event_seq,
                                                               dtime_boundary,
                                                               self.compute_intensities_at_sample_times,
                                                               compute_last_step_only=True)

                # [batch_size, 1]
                dtimes_pred = torch.sum(accepted_dtimes * weights, dim=-1)

                # [batch_size, seq_len, 1, event_num]
                intensities_at_times = self.compute_intensities_at_sample_times(time_seq,
                                                                                time_delta_seq,
                                                                                event_seq,
                                                                                dtimes_pred[:, :, None],
                                                                                max_steps=event_seq.size()[1])

                # [batch_size, seq_len, event_num]
                intensities_at_times = intensities_at_times.squeeze(dim=-2)

                # [batch_size, seq_len]
                types_pred = torch.argmax(intensities_at_times, dim=-1)

                # [batch_size, 1]
                types_pred_ = types_pred[:, -1:]
                dtimes_pred_ = dtimes_pred[:, -1:]
                time_pred_ = time_seq[:, -1:] + dtimes_pred_

                # concat to the prefix sequence
                time_seq = torch.cat([time_seq, time_pred_], dim=-1)
                time_delta_seq = torch.cat([time_delta_seq, dtimes_pred_], dim=-1)
                event_seq = torch.cat([event_seq, types_pred_], dim=-1)

            pred_dtimes.index_copy_(0, row_indices, time_delta_seq[:, -num_step - 1:])
            pred_types.index_copy_(0, row_indices, event_seq[:, -num_step - 1:])
            label_dtimes.index_copy_(0, row_indices,
                                     time_delta_seq_group[:, length - num_step - 1:length])
            label_types.index_copy_(0, row_indices, event_seq_group[:, length - num_step - 1:length])

        return pred_dtimes, pred_types, label_dtimes, label_types


TorchBaseModel = BaseModel
