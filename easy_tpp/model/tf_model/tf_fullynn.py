import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from easy_tpp.model.tf_model.tf_basemodel import TfBaseModel
from easy_tpp.utils.tf_utils import get_shape_list

if tf.__version__ >= '2.0':
    tf = tf.compat.v1
    tf.disable_v2_behavior()


class CumulHazardFunctionNetwork(keras.Model):
    """Cumulative Hazard Function Network
    ref: https://github.com/wassname/torch-neuralpointprocess
    """

    def __init__(self, model_config):
        super(CumulHazardFunctionNetwork, self).__init__()
        self.hidden_size = model_config.hidden_size
        self.num_mlp_layers = model_config.data_specs['num_mlp_layers']
        self.num_event_types = model_config.num_event_types

        # transform inter-event time embedding
        self.layer_dense_1 = layers.Dense(self.hidden_size,
                                          kernel_constraint=keras.constraints.NonNeg(),
                                          activation=tf.nn.softplus)

        # concat rnn states and inter-event time embedding
        self.layer_dense_2 = layers.Dense(self.hidden_size,
                                          kernel_constraint=keras.constraints.NonNeg(),
                                          activation=tf.nn.softplus)

        # mlp layers
        self.module_list = [layers.Dense(self.hidden_size, kernel_constraint=keras.constraints.NonNeg(),
                                         activation=tf.nn.softplus) for _ in
                            range(self.num_mlp_layers - 1)]

        self.layer_dense_3 = layers.Dense(self.num_event_types, kernel_constraint=keras.constraints.NonNeg(),
                                          activation=tf.nn.softplus)

        self.params_eps = 1e-5  # ensure positiveness of parameters

    def call(self, hidden_states, time_delta_seqs):
        # [batch_size, seq_len, hidden_size]
        t = self.layer_dense_1(time_delta_seqs[..., None])

        # [batch_size, seq_len, hidden_size]
        out = self.layer_dense_2(tf.concat([hidden_states, t], axis=-1))
        for layer in self.module_list:
            out = layer(out)

        # [batch_size, seq_len, num_event_types]
        integral_lambda = self.layer_dense_3(out)

        # [batch_size, seq_len]
        derivative_integral_lambda = tf.gradients(
            tf.reduce_mean(tf.reduce_sum(integral_lambda, axis=-1)),
            time_delta_seqs)[0]

        return integral_lambda, derivative_integral_lambda


class FullyNN(TfBaseModel):
    """Tensorflow implementation of
    Fully Neural Network based Model for General Temporal Point Processes, NeurIPS 2019.
    https://arxiv.org/abs/1905.09690
    """

    def __init__(self, model_config):
        """Initialize the model

        Args:
            model_config (EasyTPP.ModelConfig): config of model specs.
        """
        super(FullyNN, self).__init__(model_config)
        self.rnn_type = model_config.rnn_type
        self.rnn_dict = {'lstm': layers.LSTM,
                         'rnn': layers.SimpleRNN,
                         'gru': layers.GRU}

    def build_graph(self):
        """Build up the network
        """
        with tf.variable_scope('FullyRNN'):
            self.build_input_graph()

            self.layer_intensity = CumulHazardFunctionNetwork(model_config=self.model_config)

            # self.layer_intensity_
            # we can add a method to do this
            sub_rnn_class = self.rnn_dict[self.rnn_type.lower()]
            self.layer_rnn = sub_rnn_class(self.hidden_size,
                                           return_state=False,
                                           return_sequences=True,
                                           activation='tanh')

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

    def forward(self, time_seqs, time_delta_seqs, type_seqs):
        """Call the model

        Args:
            time_seqs (tensor): [batch_size, seq_len], timestamp seqs.
            time_delta_seqs (tensor): [batch_size, seq_len], inter-event time seqs.
            type_seqs (tensor): [batch_size, seq_len], event type seqs.

        Returns:
            tensor: hidden states at event times.
        """
        # [batch_size, seq_len, hidden_size]
        type_embedding = self.layer_type_emb(type_seqs)

        # [batch_size, seq_len, hidden_size + 1]
        rnn_input = tf.concat([type_embedding, time_seqs[..., None]], axis=-1)

        # [batch_size, seq_len, hidden_size]
        # states right after the event
        hidden_states = self.layer_rnn(rnn_input)

        integral_lambda, derivative_integral_lambda = self.layer_intensity(hidden_states, time_delta_seqs)

        # [batch_size, num_event_types, seq_len]
        return integral_lambda, derivative_integral_lambda

    def loglike_loss(self):
        """Compute the loglike loss.

        Returns:
            list: loglike loss, num events.
        """

        # [batch_size, seq_len, num_event_types]
        integral_lambda, derivative_integral_lambda = self.forward(self.time_delta_seqs[:, 1:],
                                                                   self.time_delta_seqs[:, 1:],
                                                                   self.type_seqs[:, :-1])
        seq_mask = self.batch_non_pad_mask[:, 1:]

        # [batch_size, seq_len]
        # A temp fix -> make it positive
        event_lambdas = tf.maximum(tf.boolean_mask(derivative_integral_lambda, seq_mask), self.eps)

        event_ll = tf.reduce_sum(tf.log(event_lambdas))

        # [batch_size, seq_len]
        # multiplied by sequence mask
        non_event_ll = tf.reduce_sum(tf.reduce_sum(integral_lambda, axis=-1) * tf.cast(seq_mask, tf.float32))

        num_events = get_shape_list(event_lambdas)[0]

        loss = - (event_ll - non_event_ll)

        return loss, num_events

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

        compute_last_step_only = kwargs.get('compute_last_step_only', False)
        _, seq_len, num_samples = get_shape_list(sample_dtimes)

        self.test = sample_dtimes

        # [batch_size, seq_len, hidden_size, num_samples]
        type_emb = tf.tile(self.layer_type_emb(type_seqs)[..., None], (1, 1, 1, num_samples))

        # [batch_size, seq_len, hidden_size + 1, num_samples]
        rnn_input = tf.concat([type_emb, sample_dtimes[:, :, None, :]], axis=-2)

        # [batch_size, num_samples， seq_len, hidden_size + 1]
        rnn_input = tf.transpose(rnn_input, perm=[0, 3, 1, 2])

        # [batch_size * num_samples， seq_len, hidden_size + 1]
        rnn_input = tf.reshape(rnn_input, (-1, seq_len, self.hidden_size + 1))

        # [batch_size * num_samples, seq_len, hidden_size]
        # states right after the event
        hidden_states = self.layer_rnn(rnn_input)

        # [batch_size, num_samples， seq_len, hidden_size]
        hidden_states = tf.reshape(hidden_states,
                                   (-1, num_samples, seq_len, self.hidden_size))

        # [batch_size, seq_len, num_sample, hidden_size]
        hidden_states = tf.transpose(hidden_states, perm=(0, 2, 1, 3))

        # [batch_size, seq_len, num_samples]
        _, derivative_integral_lambda = self.layer_intensity(hidden_states, sample_dtimes)

        # FIX: need to fix this later
        # [batch_size, seq_len, num_samples, num_event_types]
        derivative_integral_lambda = tf.tile(derivative_integral_lambda[..., None],
                                             [1, 1, 1, self.num_event_types])

        if compute_last_step_only:
            lambdas = derivative_integral_lambda[:, -1:, :, :]
        else:
            # [batch_size, seq_len, num_samples, num_event_types]
            lambdas = derivative_integral_lambda
        return lambdas
