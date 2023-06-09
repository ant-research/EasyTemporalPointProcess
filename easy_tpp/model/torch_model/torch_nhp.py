import torch
from torch import nn

from easy_tpp.model.torch_model.torch_basemodel import TorchBaseModel


class ContTimeLSTMCell(nn.Module):
    """LSTM Cell in Neural Hawkes Process, NeurIPS'17.
    """

    def __init__(self, hidden_dim, beta=1.0):
        """Initialize the continuous LSTM cell.

        Args:
            hidden_dim (int): dim of hidden state.
            beta (float, optional): beta in nn.Softplus. Defaults to 1.0.
        """
        super(ContTimeLSTMCell, self).__init__()
        self.hidden_dim = hidden_dim
        self.init_dense_layer(hidden_dim, bias=True, beta=beta)

    def init_dense_layer(self, hidden_dim, bias, beta):
        """Initialize linear layers given Equations (5a-6c) in the paper.

        Args:
            hidden_dim (int): dim of hidden state.
            bias (bool): whether to use bias term in nn.Linear.
            beta (float): beta in nn.Softplus.
        """

        self.layer_input = nn.Linear(hidden_dim * 2, hidden_dim, bias=bias)
        self.layer_forget = nn.Linear(hidden_dim * 2, hidden_dim, bias=bias)
        self.layer_output = nn.Linear(hidden_dim * 2, hidden_dim, bias=bias)
        self.layer_input_bar = nn.Linear(hidden_dim * 2, hidden_dim, bias=bias)
        self.layer_forget_bar = nn.Linear(hidden_dim * 2, hidden_dim, bias=bias)
        self.layer_pre_c = nn.Linear(hidden_dim * 2, hidden_dim, bias=bias)
        self.layer_decay = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim, bias=bias),
            nn.Softplus(beta=beta))

    def forward(self, x_i, hidden_i_minus, cell_i_minus, cell_bar_i_minus_1):
        """Update the continuous-time LSTM cell.

        Args:
            x_i (tensor): event embedding vector at t_i.
            hidden_i_minus (tensor): hidden state at t_i-
            cell_i_minus (tensor): cell state at t_i-
            cell_bar_i_minus_1 (tensor): cell bar state at t_{i-1}

        Returns:
            list: cell state, cell bar state, decay and output at t_i
        """

        x_i_ = torch.cat((x_i, hidden_i_minus), dim=1)

        # update input gate - Equation (5a)
        gate_input = torch.nn.Sigmoid()(self.layer_input(x_i_))

        # update forget gate - Equation (5b)
        gate_forget = torch.nn.Sigmoid()(self.layer_forget(x_i_))

        # update output gate - Equation (5d)
        gate_output = torch.nn.Sigmoid()(self.layer_output(x_i_))

        # update input bar - similar to Equation (5a)
        gate_input_bar = torch.nn.Sigmoid()(self.layer_input_bar(x_i_))

        # update forget bar - similar to Equation (5b)
        gate_forget_bar = torch.nn.Sigmoid()(self.layer_forget_bar(x_i_))

        # update gate z - Equation (5c)
        gate_pre_c = torch.tanh(self.layer_pre_c(x_i_))

        # update gate decay - Equation (6c)
        gate_decay = self.layer_decay(x_i_)

        # update cell state to t_i+ - Equation (6a)
        cell_i = gate_forget * cell_i_minus + gate_input * gate_pre_c

        # update cell state bar - Equation (6b)
        cell_bar_i = gate_forget_bar * cell_bar_i_minus_1 + gate_input_bar * gate_pre_c

        return cell_i, cell_bar_i, gate_decay, gate_output

    def decay(self, cell_i, cell_bar_i, gate_decay, gate_output, dtime):
        """Cell and hidden state decay according to Equation (7).

        Args:
            cell_i (tensor): cell state at t_i.
            cell_bar_i (tensor): cell bar state at t_i.
            gate_decay (tensor): gate decay state at t_i.
            gate_output (tensor): gate output state at t_i.
            dtime (tensor): delta time to decay.

        Returns:
            list: list of cell and hidden state tensors after the decay.
        """
        c_t = cell_bar_i + (cell_i - cell_bar_i) * torch.exp(-gate_decay * dtime)

        h_t = gate_output * torch.tanh(c_t)

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
        self.bias = model_config.model_specs.get('bias', False)
        self.rnn_cell = ContTimeLSTMCell(self.hidden_size)

        self.layer_intensity = nn.Sequential(
            nn.Linear(self.hidden_size, self.num_event_types, self.bias),
            nn.Softplus(self.beta))

    def init_state(self, batch_size):
        """Initialize hidden and cell states.

        Args:
            batch_size (int): size of batch data.

        Returns:
            list: list of hidden states, cell states and cell bar states.
        """
        h_t, c_t, c_bar = torch.zeros(batch_size,
                                      3 * self.hidden_size).chunk(3, dim=1)
        return h_t, c_t, c_bar

    def forward(self, batch, **kwargs):
        """Call the model.

        Args:
            batch (tuple, list): batch input.

        Returns:
            list: hidden states, [batch_size, seq_len, hidden_dim], states right before the event happens;
                  stacked decay states,  [batch_size, max_seq_length, 4, hidden_dim], states right after
                  the event happens.
        """
        time_seq, time_delta_seq, event_seq, batch_non_pad_mask, _, type_mask = batch

        all_hiddens = []
        all_outputs = []
        all_cells = []
        all_cell_bars = []
        all_decays = []

        max_steps = kwargs.get('max_steps', None)

        max_decay_time = kwargs.get('max_decay_time', 5.0)

        # last event has no time label
        max_seq_length = max_steps if max_steps is not None else event_seq.size(1) - 1

        batch_size = len(event_seq)
        h_t, c_t, c_bar_i = self.init_state(batch_size)

        # if only one event, then we dont decay
        if max_seq_length == 1:
            types_sub_batch = event_seq[:, 0]
            x_t = self.layer_type_emb(types_sub_batch)
            cell_i, c_bar_i, decay_i, output_i = \
                self.rnn_cell(x_t, h_t, c_t, c_bar_i)

            # Append all output
            all_outputs.append(output_i)
            all_decays.append(decay_i)
            all_cells.append(cell_i)
            all_cell_bars.append(c_bar_i)
            all_hiddens.append(h_t)
        else:
            # Loop over all events
            for i in range(max_seq_length):
                if i == event_seq.size(1) - 1:
                    dt = torch.ones_like(time_delta_seq[:, i]) * max_decay_time
                else:
                    dt = time_delta_seq[:, i + 1]  # need to carefully check here
                types_sub_batch = event_seq[:, i]
                x_t = self.layer_type_emb(types_sub_batch)

                # cell_i  (batch_size, process_dim)
                cell_i, c_bar_i, decay_i, output_i = \
                    self.rnn_cell(x_t, h_t, c_t, c_bar_i)

                # States decay - Equation (7) in the paper
                c_t, h_t = self.rnn_cell.decay(cell_i,
                                               c_bar_i,
                                               decay_i,
                                               output_i,
                                               dt[:, None])

                # Append all output
                all_outputs.append(output_i)
                all_decays.append(decay_i)
                all_cells.append(cell_i)
                all_cell_bars.append(c_bar_i)
                all_hiddens.append(h_t)

        # (batch_size, max_seq_length, hidden_dim)
        cell_stack = torch.stack(all_cells, dim=1)
        cell_bar_stack = torch.stack(all_cell_bars, dim=1)
        decay_stack = torch.stack(all_decays, dim=1)
        output_stack = torch.stack(all_outputs, dim=1)

        # [batch_size, max_seq_length, hidden_dim]
        hiddens_stack = torch.stack(all_hiddens, dim=1)

        # [batch_size, max_seq_length, 4, hidden_dim]
        decay_states_stack = torch.stack((cell_stack,
                                          cell_bar_stack,
                                          decay_stack,
                                          output_stack),
                                         dim=2)

        return hiddens_stack, decay_states_stack

    def loglike_loss(self, batch):
        """Compute the loglike loss.

        Args:
            batch (list): batch input.

        Returns:
            list: loglike loss, num events.
        """
        time_seqs, time_delta_seqs, type_seqs, batch_non_pad_mask, _, type_mask = batch

        hiddens_ti, decay_states = self.forward(batch)

        # Num of samples in each batch and num of event time point in the sequence
        batch_size, seq_len, _ = hiddens_ti.size()

        # Lambda(t) right before each event time point
        # lambda_at_event - [batch_size, num_times=max_len-1, num_event_types]
        # Here we drop the last event because it has no delta_time label (can not decay)
        lambda_at_event = self.layer_intensity(hiddens_ti)

        # Compute the big lambda integral in Equation (8)
        # 1 - take num_mc_sample rand points in each event interval
        # 2 - compute its lambda value for every sample point
        # 3 - take average of these sample points
        # 4 - times the interval length

        # interval_t_sample - [batch_size, num_times=max_len-1, num_mc_sample]
        # for every batch and every event point => do a sampling (num_mc_sampling)
        # the first dtime is zero, so we use time_delta_seq[:, 1:]
        interval_t_sample = self.make_dtime_loss_samples(time_delta_seqs[:, 1:])

        # [batch_size, num_times = max_len - 1, num_mc_sample, hidden_size]
        state_t_sample = self.compute_states_at_sample_times(decay_states, interval_t_sample)

        # [batch_size, num_times = max_len - 1, num_mc_sample, event_num]
        lambda_t_sample = self.layer_intensity(state_t_sample)

        event_ll, non_event_ll, num_events = self.compute_loglikelihood(lambda_at_event=lambda_at_event,
                                                                        lambdas_loss_samples=lambda_t_sample,
                                                                        time_delta_seq=time_delta_seqs[:, 1:],
                                                                        seq_mask=batch_non_pad_mask[:, 1:],
                                                                        lambda_type_mask=type_mask[:, 1:])

        # (num_samples, num_times)
        loss = - (event_ll - non_event_ll).sum()
        return loss, num_events

    def compute_states_at_sample_times(self, decay_states, sample_dtimes):
        """Compute the states at sampling times.

        Args:
            decay_states (tensor): states right after the events.
            sample_dtimes (tensor): delta times in sampling.

        Returns:
            tensor: hiddens states at sampling times.
        """
        # update the states given last event
        # cells (batch_size, num_times, hidden_dim)
        cells, cell_bars, decays, outputs = decay_states.unbind(dim=-2)

        # Use broadcasting to compute the decays at all time steps
        # at all sample points
        # h_ts shape (batch_size, num_times, num_mc_sample, hidden_dim)
        # cells[:, :, None, :]  (batch_size, num_times, 1, hidden_dim)
        _, h_ts = self.rnn_cell.decay(cells[:, :, None, :],
                                      cell_bars[:, :, None, :],
                                      decays[:, :, None, :],
                                      outputs[:, :, None, :],
                                      sample_dtimes[..., None])

        return h_ts

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

        input_ = time_seqs, time_delta_seqs, type_seqs, None, None, None

        # forward to the last but one event
        hiddens_ti, decay_states = self.forward(input_, **kwargs)

        # Num of samples in each batch and num of event time point in the sequence
        batch_size, seq_len, _ = hiddens_ti.size()

        # update the states given last event
        # cells (batch_size, num_times, hidden_dim)
        cells, cell_bars, decays, outputs = decay_states.unbind(dim=-2)

        if compute_last_step_only:
            interval_t_sample = sample_dtimes[:, -1:, :, None]
            _, h_ts = self.rnn_cell.decay(cells[:, -1:, None, :],
                                          cell_bars[:, -1:, None, :],
                                          decays[:, -1:, None, :],
                                          outputs[:, -1:, None, :],
                                          interval_t_sample)

            # [batch_size, 1, num_mc_sample, num_event_types]
            sampled_intensities = self.layer_intensity(h_ts)

        else:
            # interval_t_sample - [batch_size, num_times, num_mc_sample, 1]
            interval_t_sample = sample_dtimes[..., None]
            # Use broadcasting to compute the decays at all time steps
            # at all sample points
            # h_ts shape (batch_size, num_times, num_mc_sample, hidden_dim)
            # cells[:, :, None, :]  (batch_size, num_times, 1, hidden_dim)
            _, h_ts = self.rnn_cell.decay(cells[:, :, None, :],
                                          cell_bars[:, :, None, :],
                                          decays[:, :, None, :],
                                          outputs[:, :, None, :],
                                          interval_t_sample)

            # [batch_size, num_times, num_mc_sample, num_event_types]
            sampled_intensities = self.layer_intensity(h_ts)

        return sampled_intensities
