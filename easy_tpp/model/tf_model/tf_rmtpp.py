import tensorflow as tf
from tensorflow.keras import layers

from easy_tpp.model.tf_model.tf_basemodel import TfBaseModel

if tf.__version__ >= '2.0':
    tf = tf.compat.v1
    tf.disable_v2_behavior()


class RMTPP(TfBaseModel):
    """Tensorflow implementation of Recurrent Marked Temporal Point Process, KDD 2016.
    https://www.kdd.org/kdd2016/papers/files/rpp1081-duA.pdf
    """

    def __init__(self, model_config):
        """Initialize the model

        Args:
            model_config (dict): config of model specs.
        """
        super(RMTPP, self).__init__(model_config)

    def build_graph(self):
        """Build up the network
        """
        with tf.variable_scope('RMTPP'):
            self.build_input_graph()

            # Initialize the rnn cell
            self.layer_rnn = layers.SimpleRNN(self.hidden_size,
                                              return_state=False,
                                              return_sequences=True)

            self.layer_temporal_emb = layers.Dense(self.hidden_size)

            self.layer_hidden = layers.Dense(self.num_event_types)

            self.factor_intensity_base = tf.get_variable(name='intensity_base',
                                                         shape=[1, 1, self.num_event_types],
                                                         initializer=tf.keras.initializers.glorot_uniform())
            self.factor_intensity_current_influence = tf.get_variable(name='intensity_current_influence',
                                                                      shape=[1, 1, self.num_event_types],
                                                                      initializer=tf.keras.initializers.glorot_uniform())

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

    def state_decay(self, states_to_decay, duration_t):
        """Equation (11), which computes the intensity
        """

        # [batch_size, seq_len, num_event_types]
        states_to_decay_ = self.layer_hidden(states_to_decay)

        # [batch_size, seq_len, num_event_types]
        intensity = tf.exp(
            states_to_decay_ + self.factor_intensity_current_influence * duration_t + self.factor_intensity_base)
        return intensity

    def forward(self, time_seqs, time_delta_seqs, type_seqs):
        """Call the model.

        Args:
            time_seqs (tensor): [batch_size, seq_len], timestamp seqs.
            time_delta_seqs (tensor): [batch_size, seq_len], inter-event time seqs.
            type_seqs (tensor): [batch_size, seq_len], event type seqs.

        Returns:
            list: hidden states, [batch_size, seq_len, hidden_dim], states right before the event happens;
                  stacked decay states,  [batch_size, max_seq_length, 4, hidden_dim], states right after
                  the event happens.
        """

        # [batch_size, seq_len, hidden_size]
        type_emb = self.layer_type_emb(type_seqs)

        # [batch_size, seq_len, hidden_size]
        temporal_emb = self.layer_temporal_emb(time_seqs[..., None])

        # [batch_size, seq_len, hidden_size]
        # states right after the event
        decay_states = self.layer_rnn(type_emb + temporal_emb)

        # States decay - Equation (7) in the paper
        # states before the happening of the next event
        h_t = self.state_decay(decay_states, time_delta_seqs[..., None])

        return h_t, decay_states

    def loglike_loss(self):
        """Compute the loglike loss.

        Returns:
            tuple: loglike loss and num of events.
        """

        lambda_at_event, decay_states = self.forward(self.time_seqs[:, :-1], self.time_delta_seqs[:, 1:],
                                                     self.type_seqs[:, :-1])

        # interval_t_sample - [batch_size, num_times=max_len-1, num_mc_sample]
        # for every batch and every event point => do a sampling (num_mc_sampling)
        # the first dtime is zero, so we use time_delta_seq[:, 1:]
        interval_t_sample = self.make_dtime_loss_samples(self.time_delta_seqs[:, 1:])

        # [batch_size, num_times = max_len - 1, num_mc_sample]
        lambda_t_sample = self.compute_states_at_sample_times(decay_states, interval_t_sample)

        event_ll, non_event_ll, num_events = self.compute_loglikelihood(lambda_at_event=lambda_at_event,
                                                                        lambdas_loss_samples=lambda_t_sample,
                                                                        time_delta_seq=self.time_delta_seqs[:, 1:],
                                                                        seq_mask=self.batch_non_pad_mask[:, 1:],
                                                                        lambda_type_mask=self.type_mask[:, 1:])

        # (num_samples, num_times)
        loss = - tf.reduce_sum(event_ll - non_event_ll)

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

        _, decay_states = self.forward(time_seqs, time_delta_seqs, type_seqs)

        if compute_last_step_only:
            interval_t_sample = sampled_dtimes[:, -1:, :, None]
            # [batch_size, 1, num_mc_sample, num_event_types]
            sampled_intensities = self.state_decay(decay_states[:, -1:, None, :],
                                                   interval_t_sample)

        else:
            # interval_t_sample - [batch_size, num_times, num_mc_sample, 1]
            interval_t_sample = sampled_dtimes[..., None]
            # Use broadcasting to compute the decays at all time steps
            # at all sample points
            # sampled_intensities shape (batch_size, num_times, num_mc_sample, hidden_dim)
            # decay_states[:, :, None, :]  (batch_size, num_times, 1, hidden_dim)
            sampled_intensities = self.state_decay(decay_states[..., None, :],
                                                   interval_t_sample)

        return sampled_intensities
