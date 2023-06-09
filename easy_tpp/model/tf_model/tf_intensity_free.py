import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import layers

from easy_tpp.model.tf_model.tf_basemodel import TfBaseModel
from easy_tpp.utils.tf_utils import get_shape_list

tfd = tfp.distributions
tfb = tfp.bijectors

if tf.__version__ >= '2.0':
    tf = tf.compat.v1
    tf.disable_v2_behavior()


class MixtureSameFamily(tfd.MixtureSameFamily):
    """Mixture (same-family) distribution, redefined `log_cdf` and `log_survival_function`.
    """

    def log_cdf(self, x):
        x = x[..., None]
        log_cdf_x = self.components_distribution.log_cdf(x)
        mix_logits = self.mixture_distribution.logits
        return tf.reduce_logsumexp(log_cdf_x + mix_logits, axis=-1)

    def log_survival_function(self, x):
        x = x[..., None]
        log_sf_x = self.components_distribution.log_survival_function(x)
        mix_logits = self.mixture_distribution.logits
        return tf.reduce_logsumexp(log_sf_x + mix_logits, axis=-1)


class LogNormalMixtureDistribution:
    """
    Mixture of log-normal distributions.

    Args:
        locs (tensor): [batch_size, seq_len, num_mix_components].
        log_scales (tensor): [batch_size, seq_len, num_mix_components].
        log_weights (tensor): [batch_size, seq_len, num_mix_components].
        mean_log_inter_time (float): Average log-inter-event-time.
        std_log_inter_time (float): Std of log-inter-event-times.
    """

    def __init__(self, locs, log_scales, log_weights, mean_log_inter_time, std_log_inter_time, validate_args=None):
        mixture_dist = tfd.Categorical(logits=log_weights)
        component_dist = tfd.Normal(loc=locs, scale=tf.exp(log_scales))
        self.GMM = MixtureSameFamily(mixture_dist, component_dist)
        self.mean_log_inter_time = mean_log_inter_time
        self.std_log_inter_time = std_log_inter_time

        self.transformed_distribution = tfd.TransformedDistribution(self.GMM,
                                                                    bijector=tfb.Exp(),
                                                                    validate_args=validate_args)

    def log_prob(self, x):
        return self.transformed_distribution.log_prob(x)

    def log_survival_function(self, x):
        return self.transformed_distribution.log_survival_function(x)


class IntensityFree(TfBaseModel):
    """Tensorflow implementation of Intensity-Free Learning of Temporal Point Processes, ICLR 2020.
    https://openreview.net/pdf?id=HygOjhEYDH

    reference: https://github.com/shchur/ifl-tpp
    """

    def __init__(self, model_config):
        """Initialize the model

        Args:
            model_config (EasyTPP.ModelConfig): config of model specs.
        """
        super(IntensityFree, self).__init__(model_config)

        self.num_mix_components = model_config.data_specs['num_mix_components']
        self.num_features = 1 + self.hidden_size

    def build_graph(self):
        """Build up the network
        """
        with tf.variable_scope('IntensityFree'):
            self.build_input_graph()

            self.layer_rnn = layers.GRU(self.hidden_size,
                                        return_state=False,
                                        return_sequences=True)
            # activation='tanh')

            self.context_init = tf.zeros(self.hidden_size)[None, None, :]
            self.mark_linear = layers.Dense(self.num_event_types_pad)
            self.linear = layers.Dense(3 * self.num_mix_components)

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

    def forward(self, time_delta_seqs, type_seqs):
        """Call the model.

        Args:
            time_delta_seqs (tensor): [batch_size, seq_len], inter-event time seqs.
            type_seqs (tensor): [batch_size, seq_len], event type seqs.

        Returns:
            tensor: hidden states, [batch_size, seq_len, hidden_dim], states right before the event happens.
        """
        # [batch_size, seq_len, 1]
        temporal_seqs = tf.log(time_delta_seqs + self.eps)[..., None]

        # [batch_size, seq_len, hidden_size]
        type_emb = self.layer_type_emb(type_seqs)

        # [batch_size, seq_len, hidden_size + 1]
        features = tf.concat([temporal_seqs, type_emb], axis=-1)

        # [batch_size, seq_len, hidden_size]
        context = self.layer_rnn(features)

        batch_size, seq_len, hidden_size = get_shape_list(context)

        # (batch_size, 1, hidden_size)
        context_init = tf.tile(self.context_init, [batch_size, 1, 1])

        # (batch_size, seq_len + 1, hidden_size)
        context = tf.concat([context_init, context], axis=1)

        return context

    def loglike_loss(self):
        """Compute the loglike loss.

        Returns:
            tuple: loglikelihood loss and num of events.

        """

        time_delta_seqs = self.time_delta_seqs
        type_seqs = self.type_seqs
        batch_non_pad_mask = self.batch_non_pad_mask

        mean_log_inter_time = tf.reduce_mean(tf.log(time_delta_seqs))
        std_log_inter_time = tf.math.reduce_std(tf.log(time_delta_seqs))

        # [batch_size, seq_len, hidden_size]
        # seq_len = time_delta_seqs[:, 1:].size()[1]
        context = self.forward(time_delta_seqs[:, 1:], type_seqs[:, :-1])

        # (batch_size, seq_len, 3 * num_mix_components)
        raw_params = self.linear(context)
        locs = raw_params[..., :self.num_mix_components]
        log_scales = raw_params[..., self.num_mix_components: (2 * self.num_mix_components)]
        log_weights = raw_params[..., (2 * self.num_mix_components):]

        log_weights = tf.nn.log_softmax(log_weights, dim=-1)
        inter_time_dist = LogNormalMixtureDistribution(
            locs=locs,
            log_scales=log_scales,
            log_weights=log_weights,
            mean_log_inter_time=mean_log_inter_time,
            std_log_inter_time=std_log_inter_time
        )

        inter_times = tf.clip_by_value(time_delta_seqs, 1e-10, 1e10)
        # (batch_size, seq_len)
        log_p = inter_time_dist.log_prob(inter_times)

        # (batch_size, 1)
        # last_event_idx = tf.cast(tf.reduce_sum(batch_non_pad_mask, axis=-1, keepdims=True),
        #                          tf.int32) - 1

        log_surv_all = inter_time_dist.log_survival_function(inter_times)

        self.inter_times = log_surv_all

        #
        # # (batch_size,)
        # log_surv_last = tf.gather(log_surv_all, axis=-1, indices=last_event_idx)[..., None]

        # (batch_size, seq_len, num_marks)
        mark_logits = tf.nn.log_softmax(self.mark_linear(context), dim=-1)
        mark_dist = tfd.Categorical(logits=mark_logits)
        log_p += mark_dist.log_prob(type_seqs)

        # (batch_size, seq_len)
        log_p = tf.boolean_mask(log_p, batch_non_pad_mask) + self.eps
        # (batch_size,)
        loss = -tf.reduce_sum(log_p)

        num_events = get_shape_list(log_p)[0]

        return loss, num_events
