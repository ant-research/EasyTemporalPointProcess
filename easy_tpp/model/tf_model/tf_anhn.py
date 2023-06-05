import tensorflow as tf
from tensorflow.python.keras import layers

from easy_tpp.model.tf_model.tf_baselayer import MultiHeadAttention
from easy_tpp.model.tf_model.tf_basemodel import TfBaseModel
from easy_tpp.utils.tf_utils import get_shape_list, tensordot, swap_axes, create_tensor

if tf.__version__ >= '2.0':
    tf = tf.compat.v1
    tf.disable_v2_behavior()


class ANHN(TfBaseModel):
    """Tensorflow implementation of Attentive Neural Hawkes Network, IJCNN 2021.
        http://arxiv.org/abs/2211.11758
    """

    def __init__(self, model_config):
        """Initialize the model

        Args:
            model_config (dict): config of model specs.
        """
        super(ANHN, self).__init__(model_config)
        self.d_time = model_config['time_emb_size']
        self.use_norm = model_config['use_ln']

        self.n_layers = model_config['num_layers']
        self.n_head = model_config['num_heads']
        self.dropout = model_config['dropout']

    def build_graph(self):
        """ Build up the network """
        with tf.variable_scope('ANHN'):
            self.build_input_graph()

            self.layer_rnn = layers.LSTM(self.hidden_size,
                                         return_state=False,
                                         return_sequences=True,
                                         name='layer_rnn')

            self.lambda_w = tf.get_variable('lambda_w',
                                            shape=[self.hidden_size, self.num_event_types],
                                            dtype=tf.float32,
                                            initializer=tf.glorot_normal_initializer())
            self.lambda_b = tf.get_variable('lambda_b',
                                            shape=[self.num_event_types],
                                            dtype=tf.float32,
                                            initializer=tf.constant_initializer(0.1))

            self.layer_time_delta = layers.Dense(self.hidden_size,
                                                 activation=tf.nn.softplus,
                                                 name='layer_time_delta')
            self.layer_base_intensity = layers.Dense(self.hidden_size,
                                                     activation=tf.nn.sigmoid,
                                                     name='layer_mu')

            self.layer_att = MultiHeadAttention(self.hidden_size,
                                                self.n_head,
                                                self.dropout)

            self.layer_intensity = layers.Dense(self.num_event_types, activation=tf.nn.softplus)

            self.loss, self.num_event = self.loglike_loss()

            # Make predictions
            if self.gen_config and self.gen_config['num_step_gen'] == 1:
                self.dtime_predict_one_step, self.type_predict_one_step = \
                    self.predict_one_step_at_every_event(self.time_seqs,
                                                         self.time_delta_seqs,
                                                         self.type_seqs)

            if self.gen_config and self.gen_config['num_step_gen'] > 1:
                # make generations
                self.dtime_generation, self.type_generation = \
                    self.predict_multi_step_since_last_event(self.time_seqs,
                                                             self.time_delta_seqs,
                                                             self.type_seqs,
                                                             num_step=self.gen_config['num_step_gen'])

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
        rnn_output = self.layer_rnn(event_emb)

        # [batch_size, seq_len, hidden_size]
        # mu in Equation (3)
        intensity_base = self.layer_base_intensity(rnn_output)

        # [batch_size, num_head, seq_len, seq_len]
        _, att_weight = self.layer_att(rnn_output, attention_mask=attention_mask, output_weight=True)

        # reformat the tensor
        # [batch_size, seq_len, seq_len, 1]
        att_weight = tf.reduce_mean(att_weight, axis=1)[..., None]

        # At each step, alpha and delta reply on all previous event embeddings because there is a cumsum in Equation
        # (3), therefore the alpha and beta have shape [batch_size, seq_len, seq_len, hidden_size] when performing
        # matrix operations.
        # [batch_size, seq_len, seq_len, hidden_size]
        # alpha in Equation (3)
        intensity_alpha = att_weight * rnn_output[:, None, :, :]

        # compute delta
        max_len = get_shape_list(event_emb)[1]

        # [batch_size, seq_len, seq_len, hidden_dim]
        left = tf.tile(rnn_output[:, None, :, :], [1, max_len, 1, 1])
        right = tf.tile(rnn_output[:, :, None, :], [1, 1, max_len, 1])

        # [batch_size, seq_len, seq_len, hidden_dim * 2]
        cur_prev_concat = tf.concat([left, right], axis=-1)
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

    def loglike_loss(self):
        """Compute the loglike loss.

        Args:
            batch (list): batch input.

        Returns:
            tuple: loglikelihood loss and num of events.
        """

        # 1. compute event-loglik
        imply_lambdas, (intensity_base, intensity_alpha, intensity_delta), (base_dtime, target_cumsum_dtime) \
            = self.forward(self.time_delta_seqs[:, 1:],
                           self.type_seqs[:, :-1],
                           self.attention_mask[:, 1:, :-1])
        lambda_at_event = self.layer_intensity(imply_lambdas)

        # 2. compute non-event-loglik (using MC sampling to compute integral)
        # 2.1 sample times
        # [batch_size, seq_len, num_sample]
        sample_dtimes = self.make_dtime_loss_samples(self.time_delta_seqs[:, 1:])

        # [batch_size, num_times = max_len - 1, num_sample, event_num]
        state_t_sample = self.compute_states_at_sample_times(intensity_base, intensity_alpha, intensity_delta,
                                                             base_dtime, sample_dtimes)
        lambda_t_sample = self.layer_intensity(state_t_sample)

        event_ll, non_event_ll, num_events = self.compute_loglikelihood(lambda_at_event=lambda_at_event,
                                                                        lambdas_loss_samples=lambda_t_sample,
                                                                        time_delta_seq=self.time_delta_seqs[:, 1:],
                                                                        seq_mask=self.batch_non_pad_mask[:, 1:],
                                                                        lambda_type_mask=self.type_mask[:, 1:])

        # return enc_inten to compute accuracy
        loss = - tf.reduce_sum(event_ll - non_event_ll)

        return loss, num_events

    def compute_cumsum_dtime(self, dtime_seqs):
        """Compute cumulative delta times.

        Args:
            dtime_seqs (tensor): [batch_size, seq_len].

        Returns:
            tensor: [batch_size, seq_len].
        """
        # [batch_size, seq_len, num_sample]
        #  dtime_seqs here = dtime_seqs[:, 1:]
        # [dt_1, dt_2, dt_3] => [dt_1 + dt_2, dt_2, 0]
        cum_dtimes = tf.cumsum(dtime_seqs, axis=1, reverse=True, exclusive=True)

        # [batch_size, seq_len, seq_len, 1] (lower triangular: positive, upper: negative, diagonal: zero)
        base_elapses = tf.expand_dims(cum_dtimes[:, None, :] - cum_dtimes[:, :, None], axis=-1)

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
        elapse = tf.abs(cumsum_dtimes)

        # [batch_size, seq_len, hidden_dim]
        cumsum_term = tf.reduce_sum(intensity_alpha * tf.exp(-intensity_delta * elapse), axis=-2)
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
        # [max_len, batch_size, n_sample, hidden_dim]
        mus_trans = tf.transpose(intensity_base, perm=[1, 0, 2])[:, :, None]

        # [max_len, batch_size, n_sample, max_len, hidden_dim]
        alphas_trans = tf.transpose(intensity_alpha, perm=[1, 0, 2, 3])[:, :, None]
        deltas_trans = tf.transpose(intensity_delta, perm=[1, 0, 2, 3])[:, :, None]

        base_elapses_trans = tf.transpose(base_dtime, perm=[1, 0, 2, 3])[:, :, None]

        batch_size, num_sample_per_step, _ = get_shape_list(sample_dtimes)

        state_scan_initializer = create_tensor([batch_size,
                                                num_sample_per_step,
                                                self.hidden_size],
                                               0.0)

        # [seq_len, batch_size, num_sample, hidden_size]
        states_samples = tf.scan(fn=self.get_compute_lambda_forward_fn(),
                                 elems=[
                                     mus_trans,
                                     alphas_trans,
                                     deltas_trans,
                                     base_elapses_trans,
                                     swap_axes(sample_dtimes[:, :, :, None, None], 0, 1)],
                                 initializer=state_scan_initializer)

        # [batch_size, seq_len, num_sample, hidden_size]
        states_samples = swap_axes(states_samples, 1, 0)
        return states_samples

    def get_compute_lambda_forward_fn(self):
        """Compute the lambda using scan function.

        Returns:
            function: a forward function used in tf.scan.
        """
        compute_states_fn = self.compute_states_at_event_times

        def forward_fn(acc, item):
            mu, alpha, delta, elapse, elapse_bias = item
            return compute_states_fn(mu, alpha, delta, elapse + elapse_bias)

        return forward_fn

    def layer_intensity(self, hidden_states):
        """Compute the intensity based on the hidden states.

        Args:
            hidden_states (tensor): [batch_size, seq_len, hidden_size].

        Returns:
            tensor: [batch_size, seq_len, num_event_type_no_pad].
        """
        return tf.nn.softplus(tensordot(hidden_states, self.lambda_w) + self.lambda_b)

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
            batch_size, seq_len = get_shape_list(time_seqs)
            attention_mask = tf.ones((seq_len, seq_len))
            # only keep the strict upper triangular
            lower_diag_masks = tf.linalg.LinearOperatorLowerTriangular(tf.ones_like(attention_mask)).to_dense()
            attention_mask = tf.where(tf.equal(lower_diag_masks, 0),
                                      attention_mask,
                                      tf.zeros_like(attention_mask))
            attention_mask = tf.tile(attention_mask[None, ...], (batch_size, 1, 1))
            attention_mask = tf.cast(attention_mask, tf.int32)

        # [batch_size, seq_len, num_samples]
        imply_lambdas, (intensity_base, intensity_alpha, intensity_delta), (base_dtime, target_cumsum_dtime) \
            = self.forward(time_delta_seqs, type_seqs, attention_mask)

        # [batch_size, seq_len, num_samples, hidden_size]
        encoder_output = self.compute_states_at_sample_times(intensity_base, intensity_alpha, intensity_delta,
                                                             base_dtime, sample_dtimes)

        if compute_last_step_only:
            lambdas = self.layer_intensity(encoder_output[:, -1:, :, :])
        else:
            # [batch_size, seq_len, num_samples, num_event_types]
            lambdas = self.layer_intensity(encoder_output)
        return lambdas
