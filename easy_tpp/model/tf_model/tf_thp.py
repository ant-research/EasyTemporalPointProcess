import tensorflow as tf
from tensorflow.python.keras import layers

from easy_tpp.model.tf_model.tf_baselayer import EncoderLayer, TimePositionalEncoding
from easy_tpp.model.tf_model.tf_basemodel import TfBaseModel
from easy_tpp.utils.tf_utils import get_shape_list

if tf.__version__ >= '2.0':
    tf = tf.compat.v1
    tf.disable_v2_behavior()


class THP(TfBaseModel):
    """Tensorflow implementation of Transformer Hawkes Process, ICML 2020, https://arxiv.org/abs/2002.09291.
    """

    def __init__(self, model_config):
        """Intialiaze the model

        Args:
            model_config (EasyTPP.ModelConfig): config of model specs.
        """
        super(THP, self).__init__(model_config)
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

            self.layer_temporal_encoding = TimePositionalEncoding(self.d_model)

            self.factor_intensity_base = tf.get_variable(name='intensity_base', shape=[1, self.num_event_types])
            self.factor_intensity_decay = tf.get_variable(name='intensity_decay',
                                                          shape=[1, self.num_event_types])

            # convert hidden vectors into event-type-sized vector
            self.layer_intensity_hidden = layers.Dense(self.num_event_types)
            self.softplus = tf.nn.softplus

            self.layer_intensity = layers.Dense(self.num_event_types, activation=tf.nn.softplus)

            self.stack_layers = [EncoderLayer(hidden_size=self.d_model,
                                              num_heads=self.n_head,
                                              dropout_rate=self.dropout) for _ in range(self.n_layers)]

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
                                                             num_step=self.gen_config.num_step_gen)

    def forward(self, time_seqs, type_seqs, attention_mask):
        """Call the model

        Args:
            time_seqs (tensor): [batch_size, seq_len], timestamp seqs.
            time_delta_seqs (tensor): [batch_size, seq_len], inter-event time seqs.
            type_seqs (tensor): [batch_size, seq_len], event type seqs.
            attention_mask (tensor): [batch_size, seq_len, hidden_size], attention masks.

        Returns:
            tensor: hidden states at event times.
        """
        # [batch_size, seq_len, hidden_size]
        tem_enc = self.layer_temporal_encoding(time_seqs)
        enc_output = self.layer_type_emb(type_seqs)

        # [batch_size, seq_len, hidden_size]
        for enc_layer in self.stack_layers:
            enc_output += tem_enc
            enc_output = enc_layer((
                enc_output,
                attention_mask))
        return enc_output

    def loglike_loss(self):
        """Compute the loglike loss.

        Args:
            batch (list): batch input.

        Returns:
            list: loglike loss, num events.
        """
        # 1. compute event-loglik
        enc_out = self.forward(self.time_seqs[:, :-1],
                               self.type_seqs[:, :-1],
                               self.attention_mask[:, 1:, :-1])

        # [1, 1, num_event_types]
        factor_intensity_decay = self.factor_intensity_decay[None, ...]
        factor_intensity_base = self.factor_intensity_base[None, ...]

        # update time decay based on Equation (6)
        # [batch_size, seq_len, num_event_types]
        intensity_states = factor_intensity_decay * self.time_delta_seqs[:, 1:, None] + self.layer_intensity_hidden(
            enc_out) + factor_intensity_base

        lambda_at_event = self.softplus(intensity_states)

        # 2. compute non-event-loglik (using MC sampling to compute integral)
        # 2.1 sample times
        # [batch_size, seq_len, num_sample]
        sample_dtimes = self.make_dtime_loss_samples(self.time_delta_seqs[:, 1:])

        # 2.2 compute intensities at sampled times
        # [batch_size, num_times = max_len - 1, num_sample, event_num]
        state_t_sample = self.compute_states_at_sample_times(event_states=enc_out,
                                                             sample_dtimes=sample_dtimes)
        lambda_t_sample = self.softplus(state_t_sample)

        event_ll, non_event_ll, num_events = self.compute_loglikelihood(lambda_at_event=lambda_at_event,
                                                                        lambdas_loss_samples=lambda_t_sample,
                                                                        time_delta_seq=self.time_delta_seqs[:, 1:],
                                                                        seq_mask=self.batch_non_pad_mask[:, 1:],
                                                                        lambda_type_mask=self.type_mask[:, 1:])

        # return enc_inten to compute accuracy
        loss = - tf.reduce_sum(event_ll - non_event_ll)

        return loss, num_events

    def compute_states_at_sample_times(self, event_states, sample_dtimes):
        """Compute the hidden states at sampled times.

        Args:
            event_states (tensor): [batch_size, seq_len, hidden_size].
            sample_dtimes (tensor): [batch_size, seq_len, num_samples].

        Returns:
            tensor: hidden state at each sampled time.
        """
        # [batch_size, seq_len, 1, hidden_size]
        event_states = event_states[:, :, None, :]

        # [batch_size, seq_len, num_samples, 1]
        sample_dtimes = sample_dtimes[..., None]

        # [1, 1, 1, num_event_types]
        factor_intensity_decay = self.factor_intensity_decay[None, None, ...]
        factor_intensity_base = self.factor_intensity_base[None, None, ...]

        # update time decay based on Equation (6)
        # [batch_size, seq_len, num_samples, num_event_types]
        intensity_states = factor_intensity_decay * sample_dtimes + self.layer_intensity_hidden(
            event_states) + factor_intensity_base

        return intensity_states

    def compute_intensities_at_sample_times(self,
                                            time_seqs,
                                            time_delta_seqs,
                                            type_seqs,
                                            sample_dtimes,
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
        enc_out = self.forward(time_seqs, type_seqs, attention_mask)

        # [batch_size, seq_len, num_samples, hidden_size]
        encoder_output = self.compute_states_at_sample_times(enc_out, sample_dtimes)

        if compute_last_step_only:
            lambdas = self.layer_intensity(encoder_output[:, -1:, :, :])
        else:
            # [batch_size, seq_len, num_samples, num_event_types]
            lambdas = self.layer_intensity(encoder_output)
        return lambdas
