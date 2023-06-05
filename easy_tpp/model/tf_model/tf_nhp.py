import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from easy_tpp.model.tf_model.tf_basemodel import TfBaseModel

if tf.__version__ >= '2.0':
    tf = tf.compat.v1
    tf.disable_v2_behavior()


class ContTimeLSTMCell(keras.Model):
    """LSTM Cell in Neural Hawkes Process.
    """

    def __init__(self, hidden_size):
        """Initialize the continuous LSTM Cell

        Args:
            hidden_size (int): size of hidden states.
        """
        super(ContTimeLSTMCell, self).__init__()

        with tf.variable_scope('ContTimeLSTMCell'):
            self.hidden_size = hidden_size
            self.init_dense_layer()

    def init_dense_layer(self):
        """Initialize related dense layers.
        """
        self.layer_input = layers.Dense(self.hidden_size,
                                        activation=tf.nn.sigmoid,
                                        name='layer_input')
        self.layer_forget = layers.Dense(self.hidden_size,
                                         activation=tf.nn.sigmoid,
                                         name='layer_forget')
        self.layer_output = layers.Dense(self.hidden_size,
                                         activation=tf.nn.sigmoid,
                                         name='layer_output')
        self.layer_input_bar = layers.Dense(self.hidden_size,
                                            activation=tf.nn.sigmoid,
                                            name='layer_input_bar')
        self.layer_forget_bar = layers.Dense(self.hidden_size,
                                             activation=tf.nn.sigmoid,
                                             name='layer_forget_bar')

        self.layer_pre_c = layers.Dense(self.hidden_size,
                                        activation=tf.nn.tanh,
                                        name='layer_z')
        self.layer_decay = layers.Dense(self.hidden_size,
                                        activation=tf.nn.softplus,
                                        name='layer_decay')

    def init_state(self, batch_size):
        """Initialize hidden and cell states with zeros.

        Args:
            batch_size (tensor): size of the batch.

        Returns:
            tuple: rnn state, a tuple of three tensors and decay states, a tuple of four tensors.
        """
        zero_dims = tf.stack([batch_size, self.hidden_size])
        rnn_state = (tf.fill(zero_dims, 0.0),
                     tf.fill(zero_dims, 0.0),
                     tf.fill(zero_dims, 0.0))
        decay_state = (tf.fill(zero_dims, 0.0),
                       tf.fill(zero_dims, 0.0),
                       tf.fill(zero_dims, 0.0),
                       tf.fill(zero_dims, 0.0))
        return rnn_state, decay_state

    def call(self, x_t, dtime_t, initial_state):
        """Update the continuous time LSTM cell.

        Args:
            x_t (tensor): [batch_size, hidden_size]
            dtime_t (tensor): [batch_size, 1]
            initial_state (tuple): states initialized in function init_state.

        Returns:
            tuple: updated hidden state and tuple of rnn and decay states.
        """
        # parameter process
        h_t, c_func_t, c_bar_t = initial_state[0]
        input_t = tf.concat([x_t, h_t], axis=-1)

        # update input gate - Equation (5a)
        gate_input = self.layer_input(input_t)

        # update forget gate - Equation (5b)
        gate_forget = self.layer_forget(input_t)

        # update output gate - Equation (5d)
        gate_output = self.layer_output(input_t)

        # update input bar - similar to Equation (5a)
        gate_input_bar = self.layer_input_bar(input_t)

        # update forget bar - similar to Equation (5b)
        gate_forget_bar = self.layer_forget_bar(input_t)

        # update gate decay - Equation (6c)
        gate_decay = self.layer_decay(input_t)

        # update gate z - Equation (5c)
        z_t = self.layer_pre_c(input_t)

        # update cell state to t_i+ - Equation (6a)
        c_t = gate_forget * c_func_t + gate_input * z_t

        # update cell state bar - Equation (6b)
        c_bar_t = gate_forget_bar * c_bar_t + gate_input_bar * z_t

        c_func_t, h_t = ContTimeLSTMCell.decay(c_t, c_bar_t, gate_decay, gate_output, dtime_t)

        rnn_state = (h_t, c_func_t, c_bar_t)
        decay_state = (c_t, c_bar_t, gate_decay, gate_output)
        return h_t, (rnn_state, decay_state)

    @staticmethod
    def decay(cell_i, cell_bar_i, gate_decay, gate_output, dtime):
        """Cell and hidden state decay - Equation (7)

        Args:
            cell_i (tensor): cell state, [batch_size, hidden_size].
            cell_bar_i (tensor): cell bar state, [batch_size, hidden_size].
            gate_decay (tensor): decay state, [batch_size, hidden_size].
            gate_output (tensor): output state, [batch_size, hidden_size].
            dtime (tensor): delta time, , [batch_size, 1].

        Returns:
            tuple: cell state and hidden state.
        """
        c_t = cell_bar_i + (cell_i - cell_bar_i) * tf.math.exp(-gate_decay * dtime)
        h_t = gate_output * tf.tanh(c_t)
        return c_t, h_t

    def dynamic_run(self, seq_type_embed, dtime):
        """Update the continuous time LSTM for all time steps.

        Args:
            seq_type_embed (tensor): [batch_size, seq_len, hidden_size].
            dtime (tensor): [batch_size, seq_len].

        Returns:
            tuple: hidden state, [batch_size, seq_len, hidden_size] and decay state,
            [batch_size, 4, seq_len, hidden_size].
        """

        def move_forward_fn(accumulator, item):
            init_state = accumulator[1]
            x_t = item[0]
            dtime_t = item[1]
            h_t, init_state = self.call(x_t, dtime_t, initial_state=init_state)
            return h_t, init_state

        initial_state = self.init_state(tf.shape(seq_type_embed)[0])
        initial_h_t = initial_state[0][0]

        # Scan(move forward) along times axis
        h_ts, cell_states = tf.scan(move_forward_fn,
                                    (tf.transpose(seq_type_embed, perm=[1, 0, 2]),
                                     tf.transpose(tf.expand_dims(dtime, -1), perm=[1, 0, 2])),
                                    initializer=(initial_h_t, initial_state))

        # Transpose the tensor so that batch_size is in the first dimension
        h_ts = tf.transpose(h_ts, perm=[1, 0, 2])
        decay_states = tf.stack(cell_states[1])
        decay_states = tf.transpose(decay_states, perm=[2, 1, 0, 3])

        return h_ts, decay_states


class NHP(TfBaseModel):

    def __init__(self,
                 model_config):
        """Initialize the model

        Args:
            model_config (dict): config of model specs.
        """
        super(NHP, self).__init__(model_config)

    def build_graph(self):
        """Build up the network
        """
        with tf.variable_scope('NHP'):
            self.build_input_graph()

            # Initialize the rnn cell
            self.rnn_cell = ContTimeLSTMCell(self.hidden_size)

            self.layer_intensity = layers.Dense(self.num_event_types, activation=tf.nn.softplus)

        # Compute the loss
        self.loss, self.num_event = self.loglike_loss()

        # Make predictions
        if self.event_sampler and self.gen_config.num_step_gen == 1:
            self.dtime_predict_one_step, self.type_predict_one_step = \
                self.predict_one_step_at_every_event(self.time_seqs,
                                                     self.time_delta_seqs,
                                                     self.type_seqs)

        if self.event_sampler and self.gen_config.num_step_gen > 1:
            # make generations
            self.dtime_generation, self.type_generation = \
                self.predict_multi_step_since_last_event(self.time_seqs,
                                                         self.time_delta_seqs,
                                                         self.type_seqs,
                                                         num_step=self.gen_config.num_step_gen)

    def forward(self, dtimes_seq, event_seq, len_seq=None):
        """ Move forward through the network """

        # shape - (batch_size, seq_len, hidden_size)
        types_seq_emb = self.layer_type_emb(event_seq)

        h_ts, decay_states = self.rnn_cell.dynamic_run(types_seq_emb,
                                                       dtimes_seq)

        if len_seq is not None:
            # Get last decay state for every seq
            # Find out the index position of last decay states for each sequence
            ind_shape = tf.shape(len_seq)
            ndind = tf.concat([tf.range(ind_shape[0])[:, None], len_seq], axis=-1)

            # shape  (batch_size, 4, hidden_size)
            last_decay_states = tf.gather_nd(decay_states, ndind)
        else:
            last_decay_states = decay_states[:, -1, :, :]

        # h_ts   (batch_size, seq_len, event_num)
        # decay_states (batch_size, seq_len, 4, hidden_size)
        # last_decay_states (batch_size, 4, hidden_size)
        return h_ts, decay_states, last_decay_states

    def loglike_loss(self):
        """Compute the loglike loss.

        Returns:
            tuple: loglike loss and num of events.
        """
        hiddens_ti, decay_states, _ = self.forward(self.time_delta_seqs[:, 1:], self.type_seqs[:, :-1])

        # Lambda(t) right before each event time point
        # lambda_at_event - [batch_size, seq_len = max_len-1, num_event_types]
        # Here we drop the last event because it has no delta_time label (can not decay)
        lambda_at_event = self.layer_intensity(hiddens_ti)

        # interval_t_sample - [batch_size, seq_len = max_len-1, num_mc_sample]
        # for every batch and every event point => do a sampling (num_mc_sampling)
        # the first dtime is zero, so we use time_delta_seq[:, 1:]
        interval_t_sample = self.make_dtime_loss_samples(self.time_delta_seqs[:, 1:])

        # [batch_size, seq_len = max_len - 1, num_mc_sample]
        state_t_sample = self.compute_states_at_sampled_times(decay_states, interval_t_sample)

        # [batch_size, seq_len = max_len - 1, num_mc_sample, event_num]
        lambda_t_sample = self.layer_intensity(state_t_sample)

        event_ll, non_event_ll, num_events = self.compute_loglikelihood(lambda_at_event=lambda_at_event,
                                                                        lambdas_loss_samples=lambda_t_sample,
                                                                        time_delta_seq=self.time_delta_seqs[:, 1:],
                                                                        seq_mask=self.batch_non_pad_mask[:, 1:],
                                                                        lambda_type_mask=self.type_mask[:, 1:])

        # (num_samples, num_times)
        loss = - tf.reduce_sum(event_ll - non_event_ll)

        return loss, num_events

    def compute_states_at_sampled_times(self, decay_states, sample_dtimes, compute_last_step_only=False):
        """Compute the hidden states at sampled times.

        Args:
            decay_states (tensor): [batch_size, 4, seq_len, hidden_size].
            sample_dtimes (tensor): [batch_size, seq_len, num_samples].
            compute_last_step_only (bool, optional): whether to compute the last step only. Defaults to False.

        Returns:
            tensor: hidden state at each sampled time.
        """

        # update the states given last event
        # cells (batch_size, num_times, hidden_dim)
        cells, cell_bars, decays, outputs = tf.unstack(decay_states, 4, axis=-2)

        if compute_last_step_only:
            _, h_ts = self.rnn_cell.decay(cells[:, -1:, None, :],
                                          cell_bars[:, -1:, None, :],
                                          decays[:, -1:, None, :],
                                          outputs[:, -1:, None, :],
                                          sample_dtimes[:, -1:, :, None])

        else:
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

    def compute_intensities_at_sample_times(self, time_seqs, time_delta_seqs, type_seqs, sampled_dtimes,
                                            **kwargs):
        """Compute the intensity at sampled times.

        Args:
            time_seqs (tensor): [batch_size, seq_len], sequences of timestamps.
            time_delta_seqs (tensor): [batch_size, seq_len], sequences of delta times.
            type_seqs (tensor): [batch_size, seq_len], sequences of event types.
            sampled_dtimes (tensor): [batch_size, seq_len, num_sample], sampled time delta sequence.

        Returns:
            tensor: intensities as sampled_dtimes, [batch_size, seq_len, num_samples, event_num].
        """

        compute_last_step_only = kwargs.get('compute_last_step_only', False)

        hiddens_ti, decay_states, _ = self.forward(time_delta_seqs, type_seqs)

        # [batch_size, seq_len, num_mc_sample, hidden_dim]
        state_t_sample = self.compute_states_at_sampled_times(decay_states, sampled_dtimes, compute_last_step_only)

        # [batch_size, seq_len, num_samples, event_num]
        lambdas = self.layer_intensity(state_t_sample)

        return lambdas
