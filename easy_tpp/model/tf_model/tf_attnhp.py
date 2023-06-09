import math

import numpy as np
import tensorflow as tf
from tensorflow.python.keras import layers

from easy_tpp.model.tf_model.tf_baselayer import EncoderLayer
from easy_tpp.model.tf_model.tf_basemodel import TfBaseModel
from easy_tpp.utils.tf_utils import get_shape_list

if tf.__version__ >= '2.0':
    tf = tf.compat.v1
    tf.disable_v2_behavior()


class AttNHP(TfBaseModel):
    def __init__(self, model_config):
        super(AttNHP, self).__init__(model_config)
        self.d_model = model_config.hidden_size
        self.d_time = model_config.data_specs['time_emb_size']
        self.use_norm = model_config.data_specs['use_ln']

        self.n_layers = model_config.data_specs['num_layers']
        self.n_head = model_config.data_specs['num_heads']
        self.dropout = model_config.dropout_rate

        # position vector, used for temporal encoding
        self.div_term = self.make_div_term()

    def make_div_term(self):
        """Initialize the division term used in temporal embedding.

        Returns:
            np.array: diviser.
        """
        div_term_ = np.exp(np.arange(0, self.d_time, 2) * -(math.log(10000.0) / self.d_time))
        div_term_ = np.reshape(div_term_, (1, 1, -1))

        div_term = np.ones((1, 1, self.d_time))

        for i in range(self.d_time):
            div_term[..., i] = div_term_[..., i // 2]

        return div_term

    def build_graph(self):
        """Build up the network
        """
        with tf.variable_scope('AttNHP'):
            self.build_input_graph()

            self.layer_intensity = layers.Dense(self.num_event_types, activation=tf.nn.softplus)

            self.heads = []
            for i in range(self.n_head):
                self.heads.append([EncoderLayer(hidden_size=self.d_model,
                                                num_heads=self.n_head,
                                                dropout_rate=self.dropout) for _ in range(self.n_layers)])

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

    def compute_temporal_embedding(self, time_seqs):
        """Compute the temporal embedding.

        Args:
            time_seqs (tensor): [batch_size, seq_len].

        Returns:
            tensor: [batch_size, seq_len, emb_size].
        """

        batch_size, seq_len = get_shape_list(time_seqs)  # dynamic

        # [self.d_time]
        position_mask = np.array([1] * self.d_time)
        position_mask[1::2] = 0  # dim 2i+1
        position_mask = tf.convert_to_tensor(position_mask, tf.int32)

        # [batch_size, max_len, d_time]
        position_mask = tf.tile(position_mask[None, None, ...], [batch_size, seq_len, 1])

        time_seqs_ = time_seqs[..., None]

        position_enc = tf.where(tf.equal(position_mask, 0),  # dim 2i+1
                                tf.cos(time_seqs_ * self.div_term),  # dim 2i+1
                                tf.sin(time_seqs_ * self.div_term))  # dim 2i

        # [batch_size, max_len, hidden_dim]
        return position_enc

    def seq_encoding(self, time_seqs, type_seqs):
        """Encode the sequence.

        Args:
            time_seqs (tensor): time seqs input, [batch_size, seq_len].
            event_seqs (_type_): event type seqs input, [batch_size, seq_len].

        Returns:
            tuple: event embedding, time embedding and type embedding.
        """

        # [batch_size, seq_len, hidden_size]
        time_emb = self.compute_temporal_embedding(time_seqs)
        # [batch_size, seq_len, hidden_size]
        type_emb = tf.tanh(self.layer_type_emb(type_seqs))
        # [batch_size, seq_len, hidden_size*2]
        event_emb = tf.concat([type_emb, time_emb], axis=-1)

        return event_emb, time_emb, type_emb

    def make_layer_mask(self, attention_mask):
        """Create a tensor to do masking on layers.

        Args:
            attention_mask (tensor): mask for attention operation, [batch_size, seq_len, seq_len]

        Returns:
            tensor: aim to keep the current layer, the same size of attention mask
            a diagonal matrix, [batch_size, seq_len, seq_len]
        """
        # [batch_size, seq_len, seq_len]
        layer_mask = tf.eye(get_shape_list(attention_mask)[1]) < 1
        layer_mask = layer_mask[None, ...]
        layer_mask = tf.tile(layer_mask, [get_shape_list(attention_mask)[0], 1, 1])
        return tf.cast(layer_mask, tf.int32)

    def make_combined_att_mask(self, attention_mask, layer_mask):
        """Combined attention mask and layer mask.

        Args:
            attention_mask (tensor): mask for attention operation, [batch_size, seq_len, seq_len]
            layer_mask (tensor): mask for other layers, [batch_size, seq_len, seq_len]

        Returns:
            tensor: [batch_size, seq_len * 2, seq_len * 2]
        """

        # [batch_size, seq_len, seq_len * 2]
        combined_mask = tf.concat([attention_mask, layer_mask], axis=-1)
        # [batch_size, seq_len, seq_len * 2]
        contextual_mask = tf.concat([attention_mask, tf.ones_like(layer_mask)], axis=-1)
        # [batch_size, seq_len * 2, seq_len * 2]
        combined_mask = tf.concat([contextual_mask, combined_mask], axis=1)
        return combined_mask

    def forward_pass(self, init_cur_layer, time_emb, sample_time_emb, event_emb, combined_mask):
        """update the structure sequentially.

        Args:
            init_cur_layer (tensor): [batch_size, seq_len, hidden_size]
            time_emb (tensor): [batch_size, seq_len, hidden_size]
            sample_time_emb (tensor): [batch_size, seq_len, hidden_size]
            event_emb (tensor): [batch_size, seq_len, hidden_size]
            combined_mask (tensor): [batch_size, seq_len, hidden_size]

        Returns:
            tensor: [batch_size, seq_len, hidden_size*2]
        """
        cur_layers = []
        seq_len = get_shape_list(time_emb)[1]

        for head_i in range(self.n_head):
            # [batch_size, seq_len, hidden_size]
            cur_layer_ = init_cur_layer
            for layer_i in range(self.n_layers):
                # each layer concats the temporal emb
                # [batch_size, seq_len, hidden_size*2]
                layer_ = tf.concat([cur_layer_, sample_time_emb], axis=-1)
                # make combined input from event emb + layer emb
                # [batch_size, seq_len*2, hidden_size*2]
                _combined_input = tf.concat([event_emb, layer_], axis=1)

                enc_layer = self.heads[head_i][layer_i]
                # compute the output
                enc_output = enc_layer((_combined_input, combined_mask),
                                       training=self.is_training)

                # the layer output
                # [batch_size, seq_len, hidden_size]
                _cur_layer_ = enc_output[:, seq_len:, :]
                # add residual connection
                cur_layer_ = tf.tanh(_cur_layer_) + cur_layer_

                # event emb
                event_emb = tf.concat([enc_output[:, :seq_len, :], time_emb], axis=-1)

                if self.use_norm:
                    cur_layer_ = self.norm(cur_layer_)
            cur_layers.append(cur_layer_)
        cur_layer_ = tf.concat(cur_layers, axis=-1)

        return cur_layer_

    def forward(self, time_seqs, type_seqs, attention_mask, sample_times=None):
        """ Move forward through the network """

        # [batch_size, seq_len, hidden_size]
        event_emb, time_emb, type_emb = self.seq_encoding(time_seqs, type_seqs)
        init_cur_layer = tf.zeros_like(type_emb)
        layer_mask = self.make_layer_mask(attention_mask)

        if sample_times is None:
            sample_time_emb = time_emb
        else:
            sample_time_emb = self.compute_temporal_embedding(sample_times)
        combined_mask = self.make_combined_att_mask(attention_mask, layer_mask)

        encoder_output = self.forward_pass(init_cur_layer, time_emb, sample_time_emb, event_emb, combined_mask)

        return encoder_output

    def loglike_loss(self):
        # 1. compute event-loglik
        enc_out = self.forward(self.time_seqs[:, :-1],
                               self.type_seqs[:, :-1],
                               self.attention_mask[:, 1:, :-1],
                               self.time_seqs[:, 1:])

        # [batch_size, seq_len, num_event_types]
        lambda_at_event = self.layer_intensity(enc_out)

        # 2. compute non-event-loglik (using MC sampling to compute integral)
        # 2.1 sample times
        # [batch_size, seq_len, num_sample]
        temp_time = self.make_dtime_loss_samples(self.time_delta_seqs[:, 1:])

        # [batch_size, seq_len, num_sample]
        sample_times = temp_time + self.time_seqs[:, :-1][..., None]

        # 2.2 compute intensities at sampled times
        # [batch_size, num_times = max_len - 1, num_sample, event_num]
        lambda_t_sample = self.compute_intensities_at_sample_times(self.time_seqs[:, :-1],
                                                                   self.time_delta_seqs[:, :-1],  # not used
                                                                   self.type_seqs[:, :-1],
                                                                   sample_times,
                                                                   attention_mask=self.attention_mask[:, 1:, :-1])

        event_ll, non_event_ll, num_events = self.compute_loglikelihood(lambda_at_event=lambda_at_event,
                                                                        lambdas_loss_samples=lambda_t_sample,
                                                                        time_delta_seq=self.time_delta_seqs[:, 1:],
                                                                        seq_mask=self.batch_non_pad_mask[:, 1:],
                                                                        lambda_type_mask=self.type_mask[:, 1:])

        # return enc_inten to compute accuracy
        loss = - tf.reduce_sum(event_ll - non_event_ll)

        return loss, num_events

    def compute_states_at_sample_times(self,
                                       time_seqs,
                                       type_seqs,
                                       attention_mask,
                                       sample_times):
        """

        Args:
            time_seqs: [batch_size, seq_len]
            type_seqs: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len, seq_len]
            sample_times: [batch_size, seq_len, num_samples]

        Returns:
            hidden states at all sampled times: [batch_size, seq_len, num_samples, hidden_size]

        """
        batch_size, seq_len = get_shape_list(type_seqs)
        num_samples = get_shape_list(sample_times)[-1]

        # [num_samples, batch_size, seq_len]
        sample_times = tf.transpose(sample_times, perm=(2, 0, 1))
        # [num_samples * batch_size, seq_len]
        _sample_time = tf.reshape(sample_times, (num_samples * batch_size, -1))
        # [num_samples * batch_size, seq_len]
        _types = tf.reshape(tf.tile(type_seqs[None, ...], (num_samples, 1, 1)), (num_samples * batch_size, -1))
        # [num_samples * batch_size, seq_len]
        _times = tf.reshape(tf.tile(time_seqs[None, ...], (num_samples, 1, 1)), (num_samples * batch_size, -1))
        # [num_samples * batch_size, seq_len]
        _attn_mask = tf.tile(attention_mask[None, ...], (num_samples, 1, 1, 1))
        _attn_mask = tf.reshape(_attn_mask, (num_samples * batch_size, seq_len, seq_len))

        # [num_samples * batch_size, seq_len, hidden_size]
        encoder_output = self.forward(_times,
                                      _types,
                                      _attn_mask,
                                      _sample_time)

        # [num_samples, batch_size, seq_len, hidden_size]
        encoder_output = tf.reshape(encoder_output, (num_samples, batch_size, seq_len, -1))
        # [batch_size, seq_len, num_samples, hidden_size]
        encoder_output = tf.transpose(encoder_output, perm=(1, 2, 0, 3))
        return encoder_output

    def compute_intensities_at_sample_times(self, time_seqs, time_delta_seqs, type_seqs, sample_times, **kwargs):
        """
        Args:
            time_seqs: [batch_size, seq_len]
            time_delta_seqs: [batch_size, seq_len]
            type_seqs: [batch_size, seq_len]
            sample_times: [batch_size, seq_len, num_samples]

        Returns:
            intensities at sample times: [batch_size, seq_len, num_samples, num_event_types]
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

        # [batch_size, seq_len, num_samples, hidden_size]
        encoder_output = self.compute_states_at_sample_times(time_seqs, type_seqs, attention_mask, sample_times)

        if compute_last_step_only:
            lambdas = self.layer_intensity(encoder_output[:, -1:, :, :])
        else:
            # [batch_size, seq_len, num_samples, num_event_types]
            lambdas = self.layer_intensity(encoder_output)
        return lambdas
