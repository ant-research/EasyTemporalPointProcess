import torch
import torch.nn as nn

from easy_tpp.model.torch_model.torch_baselayer import EncoderLayer, MultiHeadAttention, \
    TimeShiftedPositionalEncoding
from easy_tpp.model.torch_model.torch_basemodel import TorchBaseModel


class SAHP(TorchBaseModel):
    """Torch implementation of Self-Attentive Hawkes Process, ICML 2020.
    Part of the code is collected from https://github.com/yangalan123/anhp-andtt/blob/master/sahp

    I slightly modify the original code because it is not stable.

    """

    def __init__(self, model_config):
        """Initialize the model

        Args:
            model_config (EasyTPP.ModelConfig): config of model specs.
        """
        super(SAHP, self).__init__(model_config)
        self.d_model = model_config.hidden_size
        self.d_time = model_config.time_emb_size

        self.use_norm = model_config.use_ln

        # position vector, used for temporal encoding
        self.layer_position_emb = TimeShiftedPositionalEncoding(d_model=self.d_model)

        self.n_layers = model_config.num_layers
        self.n_head = model_config.num_heads
        self.dropout = model_config.dropout_rate

        # convert hidden vectors into a scalar
        self.layer_intensity_hidden = nn.Linear(self.d_model, self.num_event_types)
        self.softplus = nn.Softplus()

        self.stack_layers = nn.ModuleList(
            [EncoderLayer(
                self.d_model,
                MultiHeadAttention(self.n_head, self.d_model, self.d_model, self.dropout,
                                   output_linear=False),

                use_residual=False,
                dropout=self.dropout
            ) for _ in range(self.n_layers)])

        if self.use_norm:
            self.norm = nn.LayerNorm(self.d_model)

        # Equation (12): mu
        self.mu = torch.empty([self.d_model, self.num_event_types])
        # Equation (13): eta
        self.eta = torch.empty([self.d_model, self.num_event_types])
        # Equation (14): gamma
        self.gamma = torch.empty([self.d_model, self.num_event_types])

        nn.init.xavier_normal_(self.mu)
        nn.init.xavier_normal_(self.eta)
        nn.init.xavier_normal_(self.gamma)

    def state_decay(self, encode_state, mu, eta, gamma, duration_t):
        """Equation (15), which computes the pre-intensity states

        Args:
            encode_state (tensor): [batch_size, seq_len, hidden_size].
            mu (tensor): [batch_size, seq_len, hidden_size].
            eta (tensor): [batch_size, seq_len, hidden_size].
            gamma (tensor): [batch_size, seq_len, hidden_size].
            duration_t (tensor): [batch_size, seq_len, num_sample].

        Returns:
            tensor: hidden states at event times.
        """

        # [batch_size, hidden_dim]
        states = torch.matmul(encode_state, mu) + (
                    torch.matmul(encode_state, eta) - torch.matmul(encode_state, mu)) * torch.exp(
            -torch.matmul(encode_state, gamma) * duration_t)
        return states

    def forward(self, time_seqs, time_delta_seqs, event_seqs, attention_mask):
        """Call the model

        Args:
            time_seqs (tensor): [batch_size, seq_len], timestamp seqs.
            time_delta_seqs (tensor): [batch_size, seq_len], inter-event time seqs.
            event_seqs (tensor): [batch_size, seq_len], event type seqs.
            attention_mask (tensor): [batch_size, seq_len, hidden_size], attention masks.

        Returns:
            tensor: hidden states at event times.
        """
        type_embedding = self.layer_type_emb(event_seqs)
        position_embedding = self.layer_position_emb(time_seqs, time_delta_seqs)

        enc_output = type_embedding + position_embedding

        for enc_layer in self.stack_layers:
            enc_output = enc_layer(
                enc_output,
                mask=attention_mask)
            if self.use_norm:
                enc_output = self.norm(enc_output)
        # [batch_size, seq_len, hidden_dim]
        return enc_output

    def loglike_loss(self, batch):
        """Compute the loglike loss.

        Args:
            batch (tuple, list): batch input.

        Returns:
            list: loglike loss, num events.
        """
        time_seqs, time_delta_seqs, type_seqs, batch_non_pad_mask, attention_mask, type_mask = batch

        enc_out = self.forward(time_seqs[:, :-1], time_delta_seqs[:, 1:], type_seqs[:, :-1], attention_mask[:, 1:, :-1])

        cell_t = self.state_decay(encode_state=enc_out,
                                  mu=self.mu[None, ...],
                                  eta=self.eta[None, ...],
                                  gamma=self.gamma[None, ...],
                                  duration_t=time_delta_seqs[:, 1:, None])

        # [batch_size, seq_len, num_event_types]
        lambda_at_event = self.softplus(cell_t)

        # 2. compute non-event-loglik (using MC sampling to compute integral)
        # 2.1 sample times
        # [batch_size, seq_len, num_sample]
        sample_dtimes = self.make_dtime_loss_samples(time_delta_seqs[:, 1:])

        # 2.2 compute intensities at sampled times
        # [batch_size, num_times = max_len - 1, num_sample, event_num]
        state_t_sample = self.compute_states_at_sample_times(encode_state=enc_out,
                                                             sample_dtimes=sample_dtimes)
        lambda_t_sample = self.softplus(state_t_sample)

        event_ll, non_event_ll, num_events = self.compute_loglikelihood(lambda_at_event=lambda_at_event,
                                                                        lambdas_loss_samples=lambda_t_sample,
                                                                        time_delta_seq=time_delta_seqs[:, 1:],
                                                                        seq_mask=batch_non_pad_mask[:, 1:],
                                                                        lambda_type_mask=type_mask[:, 1:])

        # return enc_inten to compute accuracy
        loss = - (event_ll - non_event_ll).sum()

        return loss, num_events

    def compute_states_at_sample_times(self,
                                       encode_state,
                                       sample_dtimes):
        """Compute the hidden states at sampled times.

        Args:
            encode_state (tensor): three tensors with each shape [batch_size, seq_len, hidden_size].
            sample_dtimes (tensor): [batch_size, seq_len, num_samples].

        Returns:
            tensor: [batch_size, seq_len, num_samples, hidden_size]， hidden state at each sampled time.
        """

        cell_states = self.state_decay(encode_state[:, :, None, :],
                                       self.mu[None, None, ...],
                                       self.eta[None, None, ...],
                                       self.gamma[None, None, ...],
                                       sample_dtimes[:, :, :, None])

        return cell_states

    def compute_intensities_at_sample_times(self,
                                            time_seqs,
                                            time_delta_seqs,
                                            type_seqs,
                                            sample_dtimes,
                                            **kwargs):
        """Compute hidden states at sampled times.

        Args:
            time_seqs (tensor): [batch_size, seq_len], times seqs.
            time_delta_seqs (tensor): [batch_size, seq_len], time delta seqs.
            type_seqs (tensor): [batch_size, seq_len], event type seqs.
            sample_dtimes (tensor): [batch_size, seq_len, num_samples], sampled inter-event timestamps.

        Returns:
            tensor: [batch_size, seq_len, num_samples, num_event_types], intensity at all sampled times.
        """

        attention_mask = kwargs.get('attention_mask', None)
        compute_last_step_only = kwargs.get('compute_last_step_only', False)

        if attention_mask is None:
            batch_size, seq_len = time_seqs.size()
            attention_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).unsqueeze(0)
            attention_mask = attention_mask.expand(batch_size, -1, -1).to(torch.bool)

        # [batch_size, seq_len, num_samples]
        enc_out = self.forward(time_seqs, time_delta_seqs, type_seqs, attention_mask)

        # [batch_size, seq_len, num_samples, hidden_size]
        encoder_output = self.compute_states_at_sample_times(enc_out, sample_dtimes)

        if compute_last_step_only:
            lambdas = self.softplus(encoder_output[:, -1:, :, :])
        else:
            # [batch_size, seq_len, num_samples, num_event_types]
            lambdas = self.softplus(encoder_output)
        return lambdas
