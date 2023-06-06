import tensorflow as tf

from easy_tpp.model.tf_model.tf_baselayer import EncoderLayer, TimeShiftedPositionalEncoding
from easy_tpp.model.tf_model.tf_basemodel import TfBaseModel
from easy_tpp.utils.tf_utils import get_shape_list

if tf.__version__ >= '2.0':
    tf = tf.compat.v1
    tf.disable_v2_behavior()


class SAHP(TfBaseModel):
    """Tensorflow implementation of Self-Attentive Hawkes Process, ICML 2020.
    https://arxiv.org/abs/1907.07561
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

        self.n_layers = model_config.num_layers
        self.n_head = model_config.num_heads
        self.dropout = model_config.dropout_rate

    def build_graph(self):
        """Build up the network
        """
        with tf.variable_scope('THP'):
            self.build_input_graph()

            self.layer_position_emb = TimeShiftedPositionalEncoding(self.hidden_size)

            # Equation (12) - (14)
            # Equation (12): mu
            self.mu = tf.get_variable(name='mu',
                                      shape=[self.d_model, self.num_event_types],
                                      initializer=tf.keras.initializers.glorot_uniform())
            # Equation (13): eta
            self.eta = tf.get_variable(name='eta',
                                       shape=[self.d_model, self.num_event_types],
                                       initializer=tf.keras.initializers.glorot_uniform())
            # Equation (14): gamma
            self.gamma = tf.get_variable(name='gamma',
                                         shape=[self.d_model, self.num_event_types],
                                         initializer=tf.keras.initializers.glorot_uniform())

            self.stack_layers = [EncoderLayer(hidden_size=self.d_model,
                                              num_heads=self.n_head,
                                              dropout_rate=self.dropout) for _ in range(self.n_layers)]

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
        # i did not use the exp operation here because it can easily explode!
        states = tf.matmul(encode_state, mu) + (
                tf.matmul(encode_state, eta) - tf.matmul(encode_state, mu)) * tf.matmul(encode_state, gamma) * duration_t

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
        position_embedding = self.layer_position_emb((time_seqs, time_delta_seqs))

        enc_output = type_embedding + position_embedding

        for enc_layer in self.stack_layers:
            enc_output = enc_layer(
                (enc_output,
                 attention_mask))

        # [batch_size, seq_len, hidden_dim]
        return enc_output

    def loglike_loss(self):
        """Compute the loglike loss.

        Returns:
            tuple: loglike loss and num of events.
        """
        # 1. compute event-loglik
        enc_out = self.forward(self.time_seqs[:, 1:],
                               self.time_delta_seqs[:, 1:],
                               self.type_seqs[:, :-1],
                               self.attention_mask[:, 1:, :-1])

        cell_t = self.state_decay(encode_state=enc_out,
                                  mu=self.mu[None, ...],
                                  eta=self.eta[None, ...],
                                  gamma=self.gamma[None, ...],
                                  duration_t=self.time_delta_seqs[:, 1:, None])

        # [batch_size, seq_len, num_event_types]
        lambda_at_event = tf.nn.softplus(cell_t)

        # 2. compute non-event-loglik (using MC sampling to compute integral)
        # 2.1 sample times
        # [batch_size, seq_len, num_sample]
        sample_dtimes = self.make_dtime_loss_samples(self.time_delta_seqs[:, 1:])

        # 2.2 compute intensities at sampled times
        # [batch_size, num_times = max_len - 1, num_sample, event_num]
        state_t_sample = self.compute_states_at_sample_times(encode_state=enc_out,
                                                             sample_dtimes=sample_dtimes)
        lambda_t_sample = tf.nn.softplus(state_t_sample)

        event_ll, non_event_ll, num_events = self.compute_loglikelihood(lambda_at_event=lambda_at_event,
                                                                        lambdas_loss_samples=lambda_t_sample,
                                                                        time_delta_seq=self.time_delta_seqs[:, 1:],
                                                                        seq_mask=self.batch_non_pad_mask[:, 1:],
                                                                        lambda_type_mask=self.type_mask[:, 1:])

        # return enc_inten to compute accuracy
        loss = - tf.reduce_sum(event_ll - non_event_ll)

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
        enc_out = self.forward(time_seqs, time_delta_seqs, type_seqs, attention_mask)

        # [batch_size, seq_len, num_samples, hidden_size]
        encoder_output = self.compute_states_at_sample_times(enc_out, sample_dtimes)

        if compute_last_step_only:
            lambdas = tf.nn.softplus(encoder_output[:, -1:, :, :])
        else:
            # [batch_size, seq_len, num_samples, num_event_types]
            lambdas = tf.nn.softplus(encoder_output)
        return lambdas
