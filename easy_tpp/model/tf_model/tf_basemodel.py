""" Base model with common functionality  """

import tensorflow as tf

from easy_tpp.model.tf_model.tf_thinning import EventSampler
from easy_tpp.utils.tf_utils import get_shape_list

if tf.__version__ >= '2.0':
    tf = tf.compat.v1
    tf.disable_v2_behavior()


class TfBaseModel(tf.keras.Model):
    def __init__(self, model_config):
        super(TfBaseModel, self).__init__()
        """Initialize the model

        Args:
            model_config (EasyTPP.ModelConfig): config of model specs.
        """
        self.model_config = model_config

        self.loss_integral_num_sample_per_step = model_config.loss_integral_num_sample_per_step
        self.hidden_size = model_config.hidden_size
        self.num_event_types = model_config.num_event_types  # not include [PAD]
        self.num_event_types_pad = model_config.num_event_types_pad  # include [PAD]
        self.event_pad_index = model_config.pad_token_id
        self.dropout_rate = model_config.dropout_rate

        self.eps = 1e-7

        self.layer_type_emb = tf.keras.layers.Embedding(self.num_event_types_pad,
                                                        self.hidden_size)

        self.gen_config = model_config.thinning
        self.event_sampler = None
        if self.gen_config:
            self.event_sampler = EventSampler(num_sample=self.gen_config.num_sample,
                                              num_exp=self.gen_config.num_exp,
                                              over_sample_rate=self.gen_config.over_sample_rate,
                                              patience_counter=self.gen_config.patience_counter,
                                              num_samples_boundary=self.gen_config.num_samples_boundary,
                                              dtime_max=self.gen_config.dtime_max)

    def build_input_graph(self):
        """Build up the network
        """
        with tf.variable_scope('BaseModel'):
            # Input placeholder
            # shape - (batch_size, max_len)
            # max_len - sequence length including time zero padding
            self.time_delta_seqs = tf.placeholder(tf.float32, shape=[None, None])

            # shape - (batch_size, max_len)
            self.time_seqs = tf.placeholder(tf.float32, shape=[None, None])

            # shape - (batch_size, max_len)
            self.type_seqs = tf.placeholder(tf.int32, shape=[None, None])

            # shape - (batch_size, max_len)
            self.batch_non_pad_mask = tf.placeholder(tf.int32, shape=[None, None])

            # shape - (batch_size, max_len, max_len)
            self.attention_mask = tf.placeholder(tf.int32, shape=[None, None, None])

            # Event type one-hot code
            # shape - (batch_size, max_len, num_event_types)
            self.type_mask = tf.placeholder(tf.float32, shape=[None, None, None])

            # shape - (batch_size 1)
            self.seq_len = tf.cast(tf.reduce_sum(tf.cast(self.batch_non_pad_mask, tf.float32), axis=1, keepdims=True),
                                   tf.int32)

            self.is_training = tf.placeholder(tf.bool)

        return

    @staticmethod
    def generate_model_from_config(model_config):
        """Generate the model in derived class based on model config.

        Args:
            model_config (EasyTPP.Config): config of model specs.
        """
        model_id = model_config.model_id

        for subclass in TfBaseModel.__subclasses__():
            if subclass.__name__ == model_id:
                return subclass(model_config)

        raise RuntimeError('No model named ' + model_id)

    def compute_intensities_at_sample_times(self, time_seqs, time_delta_seqs, event_seqs, sampled_dtimes):
        raise NotImplementedError

    def make_dtime_loss_samples(self, time_delta_seq):
        """Generate the time point samples for every interval.

        Args:
            time_delta_seq (tensor): [batch_size, seq_len].

        Returns:
            tensor: [batch_size, seq_len, n_samples]
        """
        # shape  (1, 1, n_samples)
        dtimes_ratio_sampled = tf.linspace(start=0.0,
                                           stop=1.0,
                                           num=self.loss_integral_num_sample_per_step)[None, None, :]

        # shape (batch_size, max_len, n_samples)
        sampled_dtimes = time_delta_seq[:, :, None] * dtimes_ratio_sampled

        return sampled_dtimes

    def compute_loglikelihood(self, time_delta_seq, lambda_at_event, lambdas_loss_samples, seq_mask,
                              lambda_type_mask):
        """Compute the loglikelihood of the event sequence based on Equation (8) of NHP paper.

        Args:
            time_delta_seq (tensor): [batch_size, seq_len], time_delta_seq from model input.
            lambda_at_event (tensor): [batch_size, seq_len, num_event_types], unmasked intensity at
            (right after) the event.
            lambdas_loss_samples (tensor): [batch_size, seq_len, num_sample, num_event_types],
            intensity at sampling times.
            seq_mask (tensor): [batch_size, seq_len], sequence mask vector to mask the padded events.
            lambda_type_mask (tensor): [batch_size, seq_len, num_event_types], type mask matrix to mask
            the padded event types.

        Returns:
            tuple: event loglike, non event loglike, intensity at event with padding events masked
        """

        # Sum of lambda over every type
        # [batch_size, seq_len]
        event_lambdas = tf.reduce_sum(lambda_at_event * lambda_type_mask, axis=-1)

        # mask the pad event
        event_lambdas = tf.boolean_mask(event_lambdas, seq_mask)

        # Sum of lambda over every event point
        # [num_unmasked_events]
        event_ll = tf.reduce_sum(tf.log(event_lambdas))

        # Compute the big lambda integral in Equation (8) of NHP paper
        # 1 - take num_mc_sample rand points in each event interval
        # 2 - compute its lambda value for every sample point
        # 3 - take average of these sample points
        # 4 - times the interval length

        # [batch_size, max_len, n_loss_sample]
        lambdas_total_samples = tf.reduce_sum(lambdas_loss_samples, axis=-1)

        # interval_integral - [batch_size, num_times]
        # interval_integral = length_interval * average of sampled lambda(t)
        non_event_ll = tf.reduce_mean(lambdas_total_samples, axis=-1) * time_delta_seq * tf.cast(seq_mask, tf.float32)

        non_event_ll = tf.reduce_sum(non_event_ll)

        num_events = get_shape_list(event_lambdas)[0]

        return event_ll, non_event_ll, num_events

    def predict_one_step_at_every_event(self, time_seqs, time_delta_seqs, type_seqs):
        """One-step prediction for every event in the sequence.

        Args:
            time_seqs (tensor): [batch_size, seq_len].
            time_delta_seqs (tensor): [batch_size, seq_len].
            type_seqs (tensor): [batch_size, seq_len].

        Returns:
            tuple: tensors of dtime and type prediction, [batch_size, seq_len].
        """

        # remove the last event, as the prediction based on the last event has no label
        time_seqs, time_delta_seqs, type_seqs = time_seqs[:, :-1], time_delta_seqs[:, 1:], type_seqs[:, :-1]

        # (batch_size, seq_len)
        dtime_boundary = time_delta_seqs + self.event_sampler.dtime_max

        # [batch_size, seq_len, num_sample]
        accepted_dtimes, weights = self.event_sampler.draw_next_time_one_step(time_seqs,
                                                                              time_delta_seqs,
                                                                              type_seqs,
                                                                              dtime_boundary,
                                                                              self.compute_intensities_at_sample_times,
                                                                              compute_last_step_only=False)

        # [batch_size, seq_len]
        dtimes_pred = tf.reduce_sum(accepted_dtimes * weights, axis=-1)

        # [batch_size, seq_len, 1, event_num]
        intensities_at_times = self.compute_intensities_at_sample_times(time_seqs,
                                                                        time_delta_seqs,
                                                                        type_seqs,
                                                                        dtimes_pred[:, :, None])

        # [batch_size, seq_len, event_num]
        intensities_at_times = tf.squeeze(intensities_at_times, axis=-2)

        # [batch_size, seq_len]
        types_pred = tf.argmax(intensities_at_times, axis=-1)

        return dtimes_pred, types_pred

    def predict_multi_step_since_last_event(self, time_seqs, time_delta_seqs, type_seqs, num_step):
        """Multi-step prediction for every event in the sequence.

        Args:
            time_seqs (tensor): [batch_size, seq_len].
            time_delta_seqs (tensor): [batch_size, seq_len].
            type_seqs (tensor): [batch_size, seq_len].
            num_step (int): num of steps for prediction.

        Returns:
            tuple: tensors of dtime and type prediction, [batch_size, seq_len].
        """
        i = tf.constant(0)

        time_seqs_ = time_seqs
        time_delta_seqs_ = time_delta_seqs
        type_seqs_ = type_seqs

        def while_condition(i, time_seqs_, time_delta_seqs_, type_seqs_):
            return tf.less(i, num_step)

        def body(i, time_seqs_, time_delta_seqs_, type_seqs_):
            # [batch_size, seq_len]
            dtime_boundary = time_delta_seqs_ + self.event_sampler.dtime_max

            # [batch_size, seq_len, num_sample]
            accepted_dtimes, weights = \
                self.event_sampler.draw_next_time_one_step(time_seqs_,
                                                           time_delta_seqs_,
                                                           type_seqs_,
                                                           dtime_boundary,
                                                           self.compute_intensities_at_sample_times,
                                                           compute_last_step_only=True)

            # [batch_size, seq_len]
            dtimes_pred = tf.reduce_sum(accepted_dtimes * weights, axis=-1)

            # [batch_size, seq_len, 1, event_num]
            intensities_at_times = self.compute_intensities_at_sample_times(time_seqs_,
                                                                            time_delta_seqs_,
                                                                            type_seqs_,
                                                                            dtimes_pred[:, :, None])

            # [batch_size, seq_len, event_num]
            intensities_at_times = tf.squeeze(intensities_at_times, axis=-2)

            # [batch_size, seq_len]
            types_pred = tf.argmax(intensities_at_times, axis=-1)

            # [batch_size, 1]
            types_pred_ = tf.cast(types_pred[:, -1:], tf.int32)
            dtimes_pred_ = dtimes_pred[:, -1:]
            time_pred_ = time_seqs_[:, -1:] + dtimes_pred_

            # concat to the prefix sequence
            time_seqs_ = tf.concat([time_seqs_, time_pred_], axis=-1)
            time_delta_seqs_ = tf.concat([time_delta_seqs_, dtimes_pred_], axis=-1)
            type_seqs_ = tf.concat([type_seqs_, types_pred_], axis=-1)

            return [tf.add(i, 1), time_seqs_, time_delta_seqs_, type_seqs_]

        _, _, time_delta_seqs_, type_seqs_ = tf.while_loop(while_condition,
                                                           body,
                                                           [i, time_seqs_, time_delta_seqs_, type_seqs_])
        # to be consistent with the torch version
        return time_delta_seqs_[:, -num_step - 1:], type_seqs_[:, -num_step - 1:]
