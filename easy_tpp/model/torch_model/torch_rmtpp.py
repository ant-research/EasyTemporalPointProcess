import torch
from torch import nn

from easy_tpp.model.torch_model.torch_basemodel import TorchBaseModel


class RMTPP(TorchBaseModel):
    """Torch implementation of Recurrent Marked Temporal Point Processes, KDD 2016.
    https://www.kdd.org/kdd2016/papers/files/rpp1081-duA.pdf
    """

    def __init__(self, model_config):
        """Initialize the model

        Args:
            model_config (EasyTPP.ModelConfig): config of model specs.
        """
        super(RMTPP, self).__init__(model_config)

        self.layer_temporal_emb = nn.Linear(1, self.hidden_size)

        self.layer_rnn = nn.RNN(input_size=self.hidden_size, hidden_size=self.hidden_size,
                                num_layers=1, batch_first=True)

        self.layer_hidden = nn.Linear(self.hidden_size, self.num_event_types)

        self.factor_intensity_base = torch.nn.Parameter(torch.empty([1, 1, self.num_event_types], device=self.device))
        self.factor_intensity_current_influence = torch.nn.Parameter(torch.empty([1, 1, self.num_event_types], device=self.device))

        nn.init.xavier_normal_(self.factor_intensity_base)
        nn.init.xavier_normal_(self.factor_intensity_current_influence)

    def state_decay(self, states_to_decay, duration_t):
        """Equation (11), which computes intensity
        """

        # [batch_size, seq_len, num_event_types]
        states_to_decay_ = self.layer_hidden(states_to_decay)

        # [batch_size, seq_len, num_event_types]
        # put a max number to avoid explode during HPO
        intensity = torch.exp(
            states_to_decay_ + self.factor_intensity_current_influence * duration_t +
            self.factor_intensity_base).clamp(max=1e5)
        return intensity

    def forward(self, time_seqs, time_delta_seqs, type_seqs, **kwargs):
        """Call the model.

        Args:
            batch (list): batch input.

        Returns:
            list: hidden states, [batch_size, seq_len, hidden_dim], states right before the event happens;
                  stacked decay states,  [batch_size, max_seq_length, 4, hidden_dim], states right after
                  the event happens.
        """

        max_steps = kwargs.get('max_steps', None)

        # last event has no time label
        max_seq_length = max_steps if max_steps is not None else type_seqs.size(1) - 1

        # [batch_size, seq_len, hidden_size]
        type_emb = self.layer_type_emb(type_seqs)

        # [batch_size, seq_len, hidden_size]
        temporal_emb = self.layer_temporal_emb(time_seqs[..., None])

        # [batch_size, seq_len, hidden_size]
        # states right after the event
        decay_states, _ = self.layer_rnn(type_emb + temporal_emb)

        # if only one event, then we dont decay
        if max_seq_length == 1:
            h_t = decay_states
        else:
            # States decay - Equation (7) in the paper
            # states before the happening of the next event
            h_t = self.state_decay(decay_states, time_delta_seqs[..., None])

        return h_t, decay_states

    def loglike_loss(self, batch):
        """Compute the loglike loss.

        Args:
            batch (list): batch input.

        Returns:
            tuple: loglikelihood loss and num of events.
        """
        time_seqs, time_delta_seqs, type_seqs, batch_non_pad_mask, _, type_mask = batch

        lambda_at_event, decay_states = self.forward(time_seqs[:, :-1], time_delta_seqs[:, 1:], type_seqs[:, :-1])

        # Num of samples in each batch and num of event time point in the sequence
        batch_size, seq_len, _ = lambda_at_event.size()

        # Compute the big lambda integral in equation (8)
        # 1 - take num_mc_sample rand points in each event interval
        # 2 - compute its lambda value for every sample point
        # 3 - take average of these sample points
        # 4 - times the interval length

        # interval_t_sample - [batch_size, num_times=max_len-1, num_mc_sample]
        # for every batch and every event point => do a sampling (num_mc_sampling)
        # the first dtime is zero, so we use time_delta_seqs[:, 1:]
        interval_t_sample = self.make_dtime_loss_samples(time_delta_seqs[:, 1:])

        # [batch_size, num_times = max_len - 1, num_sample, event_num]
        lambda_t_sample = self.compute_states_at_sample_times(decay_states, interval_t_sample)

        event_ll, non_event_ll, num_events = self.compute_loglikelihood(lambda_at_event=lambda_at_event,
                                                                        lambdas_loss_samples=lambda_t_sample,
                                                                        time_delta_seq=time_delta_seqs[:, 1:],
                                                                        seq_mask=batch_non_pad_mask[:, 1:],
                                                                        lambda_type_mask=type_mask[:, 1:])

        # (num_samples, num_times)
        loss = - (event_ll - non_event_ll).sum()
        return loss, num_events

    def compute_states_at_sample_times(self, decay_states, sample_dtimes):
        """Compute the hidden states at sampled times.

        Args:
            decay_states (tensor): [batch_size, seq_len, hidden_size].
            sample_dtimes (tensor): [batch_size, seq_len, num_samples].

        Returns:
            tensor: hidden state at each sampled time.
        """
        # update the states given last event

        # Use broadcasting to compute the decays at all time steps
        # decay_states[..., None, :]: [batch_size, seq_len, 1, hidden_size]
        # sample_dtimes[..., None]: [batch_size, seq_len, num_mc_sample, 1]
        # h_ts shape (batch_size, num_times, num_mc_sample, hidden_dim)
        h_ts = self.state_decay(decay_states[..., None, :],
                                sample_dtimes[..., None])

        return h_ts

    def compute_intensities_at_sample_times(self, time_seqs, time_delta_seqs, type_seqs, sample_times, **kwargs):
        """Compute the intensity at sampled times, not only event times.

        Args:
            time_seq (tensor): [batch_size, seq_len], times seqs.
            time_delta_seq (tensor): [batch_size, seq_len], time delta seqs.
            event_seq (tensor): [batch_size, seq_len], event type seqs.
            sample_dtimes (tensor): [batch_size, seq_len, num_sample], sampled inter-event timestamps.

        Returns:
            tensor: [batch_size, num_times, num_mc_sample, num_event_types],
                    intensity at each timestamp for each event type.
        """

        compute_last_step_only = kwargs.get('compute_last_step_only', False)

        # forward to the last but one event
        _, decay_states = self.forward(time_seqs, time_delta_seqs, type_seqs, **kwargs)

        # Num of samples in each batch and num of event time point in the sequence
        batch_size, seq_len, _ = decay_states.size()

        if compute_last_step_only:
            interval_t_sample = sample_times[:, -1:, :, None]
            # [batch_size, 1, num_mc_sample, num_event_types]
            sampled_intensities = self.state_decay(decay_states[:, -1:, None, :],
                                                   interval_t_sample)

        else:
            # interval_t_sample - [batch_size, num_times, num_mc_sample, 1]
            interval_t_sample = sample_times[..., None]
            # Use broadcasting to compute the decays at all time steps
            # at all sample points
            # sampled_intensities shape (batch_size, num_times, num_mc_sample, hidden_dim)
            # decay_states[:, :, None, :]  (batch_size, num_times, 1, hidden_dim)
            sampled_intensities = self.state_decay(decay_states[..., None, :],
                                                   interval_t_sample)

        return sampled_intensities
