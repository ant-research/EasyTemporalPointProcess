import torch
from torch import nn

from easy_tpp.model.torch_model.torch_baselayer import MultiHeadAttention
from easy_tpp.model.torch_model.torch_basemodel import TorchBaseModel


class ANHN(TorchBaseModel):
    """Torch implementation of Attentive Neural Hawkes Network, IJCNN 2021.
       http://arxiv.org/abs/2211.11758
    """

    def __init__(self, model_config):
        """Initialize the model

        Args:
            model_config (EasyTPP.ModelConfig): config of model specs.
        """
        super(ANHN, self).__init__(model_config)

        self.d_time = model_config['time_emb_size']
        self.use_norm = model_config['use_ln']

        self.n_layers = model_config['num_layers']
        self.n_head = model_config['num_heads']
        self.dropout = model_config['dropout']

        self.layer_rnn = nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size, batch_first=True)

        self.lambda_w = torch.empty([self.hidden_size, self.num_event_types])
        self.lambda_b = torch.empty([self.num_event_types, 1])
        nn.init.xavier_normal_(self.lambda_w)
        nn.init.xavier_normal_(self.lambda_b)

        self.layer_time_delta = nn.Sequential(nn.Linear(2 * self.hidden_size, self.hidden_size), nn.Softplus())

        self.layer_base_intensity = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size), nn.Sigmoid())

        self.layer_att = MultiHeadAttention(self.n_head,
                                            self.hidden_size,
                                            self.hidden_size,
                                            self.dropout)

        self.layer_intensity = nn.Sequential(nn.Linear(self.hidden_size, self.num_event_types), nn.Softplus())

        self.layer_temporal_emb = nn.Linear(1, self.hidden_size)

    def forward(self, dtime_seqs, type_seqs, attention_mask):
        """Call the model.

        Args:
            dtime_seqs (tensor): [batch_size, seq_len].
            type_seqs (tensor): [batch_size, seq_len].
            attention_mask (tensor): [batch_size, seq_len, hidden_size].

        Returns:
            list: hidden states, [batch_size, seq_len, hidden_size], states right before the event happens;
                  stacked decay states,  [batch_size, max_seq_length, 4, hidden_dim], states right after
                  the event happens.
        """

        # [batch_size, seq_len, hidden_size]
        event_emb = self.layer_type_emb(type_seqs)

        # [batch_size, seq_len, hidden_size]
        rnn_output, _ = self.layer_rnn(event_emb)

        # [batch_size, seq_len, hidden_size]
        # mu in Equation (3)
        intensity_base = self.layer_base_intensity(rnn_output)

        # [batch_size, num_head, seq_len, seq_len]
        _, att_weight = self.layer_att(rnn_output,
                                       rnn_output,
                                       rnn_output,
                                       mask=attention_mask,
                                       output_weight=True)

        # [batch_size, seq_len, seq_len, 1]
        att_weight = torch.sum(att_weight, dim=1)[..., None]

        # At each step, alpha and delta reply on all previous event embeddings because there is a cumsum in Equation
        # (3), therefore the alpha and beta have shape [batch_size, seq_len, seq_len, hidden_size] when performing
        # matrix operations.
        # [batch_size, seq_len, seq_len, hidden_dim]
        # alpha in Equation (3)
        intensity_alpha = att_weight * rnn_output[:, None, :, :]

        # compute delta
        max_len = event_emb.size()[1]

        # [batch_size, seq_len, seq_len, hidden_dim]
        left = rnn_output[:, None, :, :].repeat(1, max_len, 1, 1)
        right = rnn_output[:, :, None, :].repeat(1, 1, max_len, 1)
        # [batch_size, seq_len, seq_len, hidden_dim * 2]
        cur_prev_concat = torch.concat([left, right], dim=-1)
        # [batch_size, seq_len, seq_len, hidden_dim]
        intensity_delta = self.layer_time_delta(cur_prev_concat)

        # compute time elapse
        # [batch_size, seq_len, seq_len, 1]
        base_dtime, target_cumsum_dtime = self.compute_cumsum_dtime(dtime_seqs)

        # [batch_size, max_len, hidden_size]
        imply_lambdas = self.compute_states_at_event_times(intensity_base,
                                                           intensity_alpha,
                                                           intensity_delta,
                                                           target_cumsum_dtime)

        return imply_lambdas, (intensity_base, intensity_alpha, intensity_delta), (base_dtime, target_cumsum_dtime)

    def loglike_loss(self, batch):
        """Compute the loglike loss.

        Args:
            batch (list): batch input.

        Returns:
            tuple: loglikelihood loss and num of events.
        """
        time_seqs, time_delta_seqs, type_seqs, batch_non_pad_mask, attention_mask, type_mask = batch

        imply_lambdas, (intensity_base, intensity_alpha, intensity_delta), (base_dtime, target_cumsum_dtime) \
            = self.forward(time_delta_seqs[:, 1:],
                           type_seqs[:, :-1],
                           attention_mask[:, 1:, :-1])
        lambda_at_event = self.layer_intensity(imply_lambdas)

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

        state_t_sample = self.compute_states_at_sample_times(intensity_base, intensity_alpha, intensity_delta,
                                                             base_dtime, interval_t_sample)
        lambda_t_sample = self.layer_intensity(state_t_sample)

        event_ll, non_event_ll, num_events = self.compute_loglikelihood(lambda_at_event=lambda_at_event,
                                                                        lambdas_loss_samples=lambda_t_sample,
                                                                        time_delta_seq=time_delta_seqs[:, 1:],
                                                                        seq_mask=batch_non_pad_mask[:, 1:],
                                                                        lambda_type_mask=type_mask[:, 1:])

        # (num_samples, num_times)
        loss = - (event_ll - non_event_ll).sum()
        return loss, num_events

    def compute_cumsum_dtime(self, dtime_seqs):
        """Compute cumulative delta times.

        Args:
            dtime_seqs (tensor): [batch_size, seq_len].

        Returns:
            tensor: [batch_size, seq_len].
        """
        # try to replicate tf.cumsum()
        # [batch_size, seq_len, num_sample]
        # [0, dt_1, dt_2] => [dt_1 + dt_2, dt_2, 0]
        cum_dtimes = torch.cumsum(torch.flip(dtime_seqs, dims=[-1]), dim=1)
        cum_dtimes = torch.concat([torch.zeros_like(cum_dtimes[:, :1]), cum_dtimes[:, 1:]], dim=1)

        # [batch_size, seq_len, seq_len, 1] (lower triangular: positive, upper: negative, diagonal: zero)
        base_elapses = torch.unsqueeze(cum_dtimes[:, None, :] - cum_dtimes[:, :, None], dim=-1)

        # [batch_size, seq_len, seq_len， 1]
        target_cumsum = base_elapses + dtime_seqs[:, :, None, None]

        return base_elapses, target_cumsum

    def compute_states_at_event_times(self, intensity_base, intensity_alpha, intensity_delta, cumsum_dtimes):
        """Compute implied lambda based on Equation (3).

        Args:
            intensity_base (tensor): [batch_size, seq_len, (num_sample), hidden_size]
            intensity_alpha (tensor): [batch_size, seq_len, seq_len, (num_sample), hidden_size]
            intensity_delta (tensor): [batch_size, seq_len, seq_len, (num_sample), hidden_size]
            cumsum_dtimes: [batch_size, seq_len, (num_sample), 1]

        Returns:
            hidden states at all cumsum_dtimes: [batch_size, seq_len, num_samples, hidden_size]

        """
        # to avoid nan calculated by exp after (nan * 0 = nan)
        elapse = torch.abs(cumsum_dtimes)

        # [batch_size, seq_len, hidden_dim]
        cumsum_term = torch.sum(intensity_alpha * torch.exp(-intensity_delta * elapse), dim=-2)
        # [batch_size, seq_len, hidden_dim]
        imply_lambdas = intensity_base + cumsum_term

        return imply_lambdas

    def compute_states_at_sample_times(self, intensity_base, intensity_alpha, intensity_delta, base_dtime,
                                       sample_dtimes):
        """Compute the hidden states at sampled times.

        Args:
            intensity_base (tensor): [batch_size, seq_len, hidden_size].
            intensity_alpha (tensor): [batch_size, seq_len, seq_len, hidden_size].
            intensity_delta (tensor): [batch_size, seq_len, seq_len, hidden_size].
            base_dtime (tensor): [batch_size, seq_len, seq_len, hidden_size].
            sample_dtimes (tensor): [batch_size, seq_len, num_samples].

        Returns:
            tensor: hidden state at each sampled time, [batch_size, seq_len, num_sample, hidden_size].
        """

        # [batch_size, seq_len, 1, hidden_size]
        mu = intensity_base[:, :, None]

        # [batch_size, seq_len, 1, seq_len, hidden_size]
        alpha = intensity_alpha[:, :, None]
        delta = intensity_delta[:, :, None]
        base_elapses = base_dtime[:, :, None]

        # [batch_size, seq_len, num_samples, 1, 1]
        sample_dtimes_ = sample_dtimes[:, :, :, None, None]

        states_samples = []
        seq_len = intensity_base.size()[1]
        for _ in range(seq_len):
            states_samples_ = self.compute_states_at_event_times(mu, alpha, delta, base_elapses + sample_dtimes_)
            states_samples.append(states_samples_)

        # [batch_size, seq_len, num_sample, hidden_size]
        states_samples = torch.stack(states_samples, dim=1)
        return states_samples

    def compute_intensities_at_sample_times(self, time_seqs, time_delta_seqs, type_seqs, sample_dtimes, **kwargs):
        """Compute the intensity at sampled times.

        Args:
            time_seqs (tensor): [batch_size, seq_len], sequences of timestamps.
            time_delta_seqs (tensor): [batch_size, seq_len], sequences of delta times.
            type_seqs (tensor): [batch_size, seq_len], sequences of event types.
            sampled_dtimes (tensor): [batch_size, seq_len, num_sample], sampled time delta sequence.

        Returns:
            tensor: intensities as sampled_dtimes, [batch_size, seq_len, num_samples, event_num].
        """

        attention_mask = kwargs.get('attention_mask', None)
        compute_last_step_only = kwargs.get('compute_last_step_only', False)

        if attention_mask is None:
            batch_size, seq_len = time_seqs.size()
            attention_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).unsqueeze(0)
            attention_mask = attention_mask.expand(batch_size, -1, -1).to(torch.bool)

        # [batch_size, seq_len, num_samples]
        imply_lambdas, (intensity_base, intensity_alpha, intensity_delta), (base_dtime, target_cumsum_dtime) \
            = self.forward(time_delta_seqs, type_seqs, attention_mask)

        # [batch_size, seq_len, num_samples, hidden_size]
        encoder_output = self.compute_states_at_sample_times(intensity_base, intensity_alpha, intensity_delta,
                                                             base_dtime, sample_dtimes)

        if compute_last_step_only:
            lambdas = self.softplus(encoder_output[:, -1:, :, :])
        else:
            # [batch_size, seq_len, num_samples, num_event_types]
            lambdas = self.softplus(encoder_output)
        return lambdas
