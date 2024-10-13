import torch
from torch import nn

from easy_tpp.model.torch_model.torch_basemodel import TorchBaseModel
from easy_tpp.model.torch_model.torch_baselayer import ScaledSoftplus


class ContTimeLSTMCell(nn.Module):
    """LSTM Cell in Neural Hawkes Process, NeurIPS'17.
    """

    def __init__(self, hidden_dim):
        """Initialize the continuous LSTM cell.

        Args:
            hidden_dim (int): dim of hidden state.
        """
        super(ContTimeLSTMCell, self).__init__()
        self.hidden_dim = hidden_dim
        self.init_dense_layer(hidden_dim, bias=True)

    def init_dense_layer(self, hidden_dim, bias):
        """Initialize linear layers given Equations (5a-6c) in the paper.

        Args:
            hidden_dim (int): dim of hidden state.
        """

        self.linear_layer = nn.Linear(2 * hidden_dim, 7 * hidden_dim, bias=bias)
        self.softplus = nn.Softplus()

    def forward(self, x_i, hidden_ti_minus, ct_ti_minus, c_bar_im1):
        """Update the continuous-time LSTM cell.

        Args:
            x_i (tensor): event embedding vector at t_i.
            hidden_ti_minus (tensor): hidden state at t_i-
            ct_ti_minus (tensor): cell state c(t) at t_i-
            c_bar_im1 (tensor): cell state c_bar at t_{i-1} (c_bar_{i-1})

        Returns:
            list: cell state, cell bar state, decay and output at t_i
        """

        x_i_ = torch.cat((x_i, hidden_ti_minus), dim=1)

        i_i, i_bar_i, f_i, f_bar_i, z_i, o_i, delta_i = self.linear_layer(x_i_).chunk(7, dim=-1)

        i_i, i_bar_i, f_i, f_bar_i, z_i, o_i, delta_i = (
            torch.sigmoid(i_i),  # Eq (5a)
            torch.sigmoid(i_bar_i),  # Eq (5a) - Bar version
            torch.sigmoid(f_i),  # Eq (5b)
            torch.sigmoid(f_bar_i),  # Eq (5b) - Bar version
            torch.tanh(z_i),  # Eq (5c)
            torch.sigmoid(o_i),  # Eq (5d)
            self.softplus(delta_i)  # Eq (6c)
        )

        # Eq (6a)
        c_i = f_i * ct_ti_minus + i_i * z_i

        # Eq (6b)
        c_bar_i = f_bar_i * c_bar_im1 + i_bar_i * z_i

        return c_i, c_bar_i, delta_i, o_i

    def decay(self, c_i, c_bar_i, delta_i, o_i, dtime):
        """Cell and hidden state decay according to Equation (7).

        Args:
            c_i (tensor): cell state c(t) at t_i.
            c_bar_i (tensor): cell state c_bar at t_i (c_bar_i).
            delta_i (tensor): gate decay state at t_i.
            o_i (tensor): gate output state at t_i.
            dtime (tensor): delta time to decay.

        Returns:
            list: list of cell and hidden state tensors after the decay.
        """

        c_t = c_bar_i + (c_i - c_bar_i) * torch.exp(-delta_i * dtime)
        h_t = o_i * torch.tanh(c_t)
        return c_t, h_t


class NHP(TorchBaseModel):
    """Torch implementation of The Neural Hawkes Process: A Neurally Self-Modulating Multivariate Point Process,
       NeurIPS 2017, https://arxiv.org/abs/1612.09328.
    """

    def __init__(self, model_config):
        """Initialize the NHP model.

        Args:
            model_config (EasyTPP.ModelConfig): config of model specs.
        """
        super(NHP, self).__init__(model_config)
        self.beta = model_config.model_specs.get('beta', 1.0)
        self.bias = model_config.model_specs.get('bias', True)
        self.rnn_cell = ContTimeLSTMCell(self.hidden_size)

        self.layer_intensity = nn.Sequential(  # eq. 4a,
            nn.Linear(self.hidden_size, self.num_event_types, self.bias),
            ScaledSoftplus(self.num_event_types))  # learnable mark-specific beta

    def get_init_state(self, batch_size):
        c_t, c_bar_t, delta_t, o_t = torch.zeros(
            batch_size,
            4 * self.hidden_size,
            device=self.device).chunk(4, dim=1)
        return c_t, c_bar_t, delta_t, o_t  # Okay to initialize delta to be zero because c==c_bar at the beginning

    def forward(self, batch):
        '''
        Suppose we have inputs with original sequence length N+1
        ts: [t0, t1, ..., t_N]
        dts: [0, t1 - t0, t2 - t1, ..., t_N - t_{N-1}]
        marks: [k0, k1, ..., k_N] (k0 and kN could be padded marks if t0 and tN correspond to left and right windows)

        Return:
            Left limits of [t_1, ..., t_N] of shape: (batch_size, seq_len - 1, hidden_dim)
            Right limits of [t_0, ..., t_{N-1}, t_N] of shape: (batch_size, seq_len, 4 * hidden_dim)
            We need the right limit of t_N to sample continuation.

        > rnn_cell.recurrence(event_emb_t, h_tm1, c_tm1, c_bar_tm1) -> c_t, c_bar_t, gate_delta, gate_o
        > rnn_cell.decay(c_t, c_bar_t, delta_t, o_t, dt) -> c_d_t, h_d_t
        '''
        t_BN, dt_BN, marks_BN, _, _ = batch
        B, N = dt_BN.shape
        left_hs = []
        right_states = []

        all_event_emb_BNP = self.layer_type_emb(marks_BN)
        c_t, c_bar_t, delta_t, o_t = self.get_init_state(B)  # initialize the right limits
        for i in range(N):
            # Take last right limit and evolve into left limit; we will discard this value for t0 because dt=0
            ct_d_t, h_d_t = self.rnn_cell.decay(c_t, c_bar_t, delta_t, o_t, dt_BN[..., i][..., None])

            # Take left limit and update to be right limit
            event_emb_t = all_event_emb_BNP[..., i, :]
            c_t, c_bar_t, delta_t, o_t = self.rnn_cell(
                x_i=event_emb_t,
                hidden_ti_minus=h_d_t,
                ct_ti_minus=ct_d_t,
                c_bar_im1=c_bar_t,
            )

            left_hs.append(h_d_t)
            right_states.append(torch.cat((c_t, c_bar_t, delta_t, o_t), dim=-1))

        left_hiddens = torch.stack(left_hs[1:], dim=-2)  # (batch_size, seq_len - 1, hidden_dim)
        right_hiddens = torch.stack(right_states, dim=-2)  # (batch_size, seq_len, 4 * hidden_dim)
        return left_hiddens, right_hiddens

    def get_states(self, right_hiddens, sample_dts):
        """
        right_hiddens:  (batch_size, seq_len, 4 * hidden_dim): (c_t, c_bar_t, delta_t, o_t)
        sample_dts: (batch_size, seq_len, MC_points)

        > rnn_cell.decay(c_t, c_bar_t, delta_t, o_t, dt) -> c_d_t, h_d_t
        """
        c_t, c_bar_t, delta_t, o_t = torch.chunk(right_hiddens, 4, dim=-1)
        _, h_ts = self.rnn_cell.decay(c_t[:, :, None, :],
                                      c_bar_t[:, :, None, :],
                                      delta_t[:, :, None, :],
                                      o_t[:, :, None, :],
                                      sample_dts[..., None])
        return h_ts

    def loglike_loss(self, batch):
        """Compute the log-likelihood loss.

        Args:
            batch (list): batch input.

        Returns:
            tuple: loglikelihood loss and num of events.
        """
        ts_BN, dts_BN, marks_BN, batch_non_pad_mask, _ = batch

        # 1. compute hidden states at event time
        # left limits of [t_1, ..., t_N]
        # right limits of [t_0, ..., t_{N-1}, t_N]
        left_hiddens, right_hiddens = self.forward((ts_BN, dts_BN, marks_BN, None, None))
        right_hiddens = right_hiddens[..., :-1, :]  # discard right limit at t_N for logL

        # 2. evaluate intensity values at each event *from the left limit*
        intensity_B_Nm1_M = self.layer_intensity(left_hiddens)

        # 3. sample dts in each interval for estimating the integral
        dts_sample_B_Nm1_G = self.make_dtime_loss_samples(dts_BN[:, 1:])

        # 4. evaluate intensity at dt_samples for MC *from the left limit* after decay -> shape (B, N-1, G, M)
        intensity_dts_B_Nm1_G_M = self.layer_intensity(self.get_states(right_hiddens, dts_sample_B_Nm1_G))

        event_ll, non_event_ll, num_events = self.compute_loglikelihood(
            lambda_at_event=intensity_B_Nm1_M,
            lambdas_loss_samples=intensity_dts_B_Nm1_G_M,
            time_delta_seq=dts_BN[:, 1:],
            seq_mask=batch_non_pad_mask[:, 1:],
            type_seq=marks_BN[:, 1:])

        # compute loss to minimize
        loss = - (event_ll - non_event_ll).sum()
        return loss, num_events

    def compute_intensities_at_sample_times(self, time_seqs, time_delta_seqs, type_seqs, sample_dtimes, **kwargs):
        """Compute the intensity at sampled times, not only event times.

        Args:
            time_seqs (tensor): [batch_size, seq_len], times seqs.
            time_delta_seqs (tensor): [batch_size, seq_len], time delta seqs.
            type_seqs (tensor): [batch_size, seq_len], event type seqs.
            sample_dtimes (tensor): [batch_size, seq_len, num_sample], sampled inter-event timestamps.

        Returns:
            tensor: [batch_size, num_times, num_mc_sample, num_event_types],
                    intensity at each timestamp for each event type.
        """

        compute_last_step_only = kwargs.get('compute_last_step_only', False)

        _input = time_seqs, time_delta_seqs, type_seqs, None, None

        # We will need the right limit at the last given event to decay from and get the left limits for sampling
        _, right_hiddens = self.forward(_input)

        c_i, c_bar_i, delta_i, o_i = torch.chunk(right_hiddens, 4, dim=-1)

        if compute_last_step_only:
            interval_t_sample = sample_dtimes[:, -1:, :, None]
            _, h_ts = self.rnn_cell.decay(c_i[:, -1:, None, :],
                                          c_bar_i[:, -1:, None, :],
                                          delta_i[:, -1:, None, :],
                                          o_i[:, -1:, None, :],
                                          interval_t_sample)

            # [batch_size, 1, num_mc_sample, num_marks]
            sampled_intensities = self.layer_intensity(h_ts)

        else:
            # interval_t_sample - [batch_size, seq_len, num_mc_sample, 1]
            interval_t_sample = sample_dtimes[..., None]
            # Use broadcasting to compute the decays at all time steps
            # at all sample points
            # h_ts shape (batch_size, seq_len, num_mc_sample, hidden_dim)
            # cells[:, :, None, :]  (batch_size, seq_len, 1, hidden_dim)
            _, h_ts = self.rnn_cell.decay(c_i[:, :, None, :],
                                          c_bar_i[:, :, None, :],
                                          delta_i[:, :, None, :],
                                          o_i[:, :, None, :],
                                          interval_t_sample)

            # [batch_size, seq_len, num_mc_sample, num_marks]
            sampled_intensities = self.layer_intensity(h_ts)

        return sampled_intensities
