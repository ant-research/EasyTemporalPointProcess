import numpy as np
import tensorflow as tf
from tensorflow.python.keras import layers

from easy_tpp.utils import py_assert
from easy_tpp.utils.tf_utils import get_shape_list

if tf.__version__ >= '2.0':
    tf = tf.compat.v1
    tf.disable_v2_behavior()


def gelu(x):
    """Gaussian Error Linear Unit.
    This is a smoother version of the RELU.
    Original paper: https://arxiv.org/abs/1606.08415
    Args:
        x: float Tensor to perform activation.
    Returns:
        `x` with the GELU activation applied.
    """
    cdf = 0.5 * (1.0 + tf.tanh((np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
    return x * cdf


def swish(x):
    return x * tf.sigmoid(x)


def null_activation(x):
    return x


def activation_layer(activation):
    if activation.lower() == 'null':
        return null_activation
    if activation.lower() == 'gelu':
        act_layer = gelu
    elif activation.lower() == 'swish':
        act_layer = swish
    elif isinstance(activation, str):
        act_layer = tf.keras.layers.Activation(activation)
    elif issubclass(activation, layers.Layer):
        act_layer = activation()
    else:
        raise ValueError(
            "Invalid activation,found %s.You should use a str or a Activation Layer Class." % (activation))
    return act_layer


def append_tensor_alias(tensor, alias):
    """Append an alias to the list of aliases of the tensor.
    Args:
      tensor: A `Tensor`.
      alias: String, to add to the list of aliases of the tensor.
    Returns:
      The tensor with a new alias appended to its list of aliases.
    """
    # Remove ending '/' if present.
    if alias[-1] == '/':
        alias = alias[:-1]
    if hasattr(tensor, 'aliases'):
        tensor.aliases.append(alias)
    else:
        tensor.aliases = [alias]
    return tensor


class LayerNormalization(layers.Layer):
    """

    ref: https://github.com/shenweichen/DeepCTR/blob/master/deepctr/layers/normalization.py

    """

    def __init__(self, axis=-1, eps=1e-9, center=True,
                 scale=True, **kwargs):
        self.axis = axis
        self.eps = eps
        self.center = center
        self.scale = scale
        super(LayerNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        self.gamma = self.add_weight(name='gamma', shape=input_shape[-1:],
                                     initializer=tf.keras.initializers.Ones(), trainable=True)
        self.beta = self.add_weight(name='beta', shape=input_shape[-1:],
                                    initializer=tf.keras.initializers.Zeros(), trainable=True)
        super(LayerNormalization, self).build(input_shape)

    def call(self, inputs):
        mean = tf.keras.backend.mean(inputs, axis=self.axis, keepdims=True)
        variance = tf.keras.backend.mean(tf.keras.backend.square(inputs - mean), axis=-1, keepdims=True)
        std = tf.keras.backend.sqrt(variance + self.eps)
        outputs = (inputs - mean) / std
        if self.scale:
            outputs *= self.gamma
        if self.center:
            outputs += self.beta
        return outputs


class MultiHeadAttention(layers.Layer):
    def __init__(self, hidden_size, num_heads, dropout_rate, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        py_assert(hidden_size % num_heads == 0, ValueError, 'In Attention, hidden size '
                                                            'must be a multiplier of num_heads')
        self.head_size = int(hidden_size / num_heads)
        self.dropout = layers.Dropout(dropout_rate)

    def build(self, input_shape):
        """Build up weight tensors.

        Args:
            input_shape (tensor): shape of the input.
        """
        self.q_head_weight = self.add_weight(
            shape=(self.hidden_size, self.head_size * self.num_heads),
            initializer=tf.keras.initializers.TruncatedNormal(),
            dtype=tf.float32,
            name='query/kernel')

        self.k_head_weight = self.add_weight(
            shape=(self.hidden_size, self.head_size * self.num_heads),
            initializer=tf.keras.initializers.TruncatedNormal(),
            dtype=tf.float32,
            name='key/kernel')

        self.v_head_weight = self.add_weight(
            shape=(self.hidden_size, self.head_size * self.num_heads),
            initializer=tf.keras.initializers.TruncatedNormal(),
            dtype=tf.float32,
            name='value/kernel')

        self.q_head_bias = self.add_weight(
            shape=(self.head_size * self.num_heads,),
            initializer=tf.keras.initializers.TruncatedNormal(),
            dtype=tf.float32,
            name='query/bias')

        self.k_head_bias = self.add_weight(
            shape=(self.head_size * self.num_heads,),
            initializer=tf.keras.initializers.TruncatedNormal(),
            dtype=tf.float32,
            name='key/bias')

        self.v_head_bias = self.add_weight(
            shape=(self.head_size * self.num_heads,),
            initializer=tf.keras.initializers.TruncatedNormal(),
            dtype=tf.float32,
            name='value/bias')

        super(MultiHeadAttention, self).build(input_shape)

    def _abs_attn_core(self, q_head, k_head, v_head, attn_mask, training,
                       scale):
        """Compute the attention weight score.

        Args:
            q_head (tensor): [batch_size, q_seq_len, num_head, head_size].
            k_head (tensor): [batch_size, kv_seq_len, num_head, head_size].
            v_head (tensor): [batch_size, kv_seq_len, num_head, head_size].
            attn_mask (tensor): [batch_size, q_seq_len, kv_seq_len].
            training (bool): whether in training mode.
            scale (float): equals to 1 / (head_size ** 0.5), which scales down the attention score.

        Returns:
            tuple: attention context vector, [batch_size, q_seq_len, num_head, head_size];
             and attention weight [batch_size, num_head, q_seq_len, kv_seq_len].
        """
        # [batch_size, num_head, q_seq_len, kv_seq_len]
        attn_score = tf.einsum('bind,bjnd->bnij', q_head, k_head)

        # [batch_size, num_head, q_seq_len, kv_seq_len]
        attn_score = tf.multiply(attn_score, scale)

        # [batch_size, 1, q_seq_len, kv_seq_len]
        attn_mask = tf.expand_dims(attn_mask, axis=[1])

        # elements 1 => need to mask, 0 => no mask
        # adder = (1.0 - tf.cast(attn_mask, tf.float32)) * -10000.0
        adder = tf.cast(attn_mask, tf.float32) * -10000.0
        attn_score += adder

        # [batch_size, num_head, q_seq_len, kv_seq_len]
        attn_prob = tf.nn.softmax(attn_score)
        attn_prob = self.dropout(attn_prob, training=training)

        # [batch_size, q_seq_len, num_head, head_size]
        attn_vec = tf.einsum('bnij,bjnd->bind', attn_prob, v_head)
        return attn_vec, attn_prob

    def call(self, attention_input, attention_mask, kv=None, training=False, output_weight=False, **kwargs):

        q_input = attention_input
        if kv is None:
            k_input = attention_input
            v_input = attention_input
        else:
            k_input = v_input = kv

        batch_size, q_seq_length, kv_seq_length = get_shape_list(attention_mask)

        q_head_h = tf.einsum('bih,hx->bix', q_input, self.q_head_weight)
        q_head_h = tf.nn.bias_add(q_head_h, self.q_head_bias)

        k_head_h = tf.einsum('bih,hx->bix', k_input, self.k_head_weight)
        k_head_h = tf.nn.bias_add(k_head_h, self.k_head_bias)

        v_head_h = tf.einsum('bih,hx->bix', v_input, self.v_head_weight)
        v_head_h = tf.nn.bias_add(v_head_h, self.v_head_bias)

        q_head_h = tf.reshape(q_head_h, [batch_size, q_seq_length, self.num_heads, self.head_size])
        k_head_h = tf.reshape(k_head_h, [batch_size, kv_seq_length, self.num_heads, self.head_size])
        v_head_h = tf.reshape(v_head_h, [batch_size, kv_seq_length, self.num_heads, self.head_size])

        scale = 1 / (self.head_size ** 0.5)
        attn_vec, attn_prob = self._abs_attn_core(q_head_h, k_head_h, v_head_h, attention_mask, training, scale)
        attn_vec = tf.reshape(attn_vec, [batch_size, q_seq_length, self.head_size * self.num_heads])
        if output_weight:
            return attn_vec, attn_prob
        else:
            return attn_vec


class DenseDropoutLayernorm(layers.Layer):
    def __init__(self, hidden_size, dropout_rate, **kwargs):
        super(DenseDropoutLayernorm, self).__init__(**kwargs)
        self.dense = layers.Dense(hidden_size)
        self.LayerNorm = LayerNormalization()
        self.dropout = layers.Dropout(dropout_rate)

    def call(self, inputs, training=False):
        hidden_states, input_tensor = inputs
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states, training=training)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class EncoderLayer(layers.Layer):
    def __init__(self, hidden_size, num_heads, dropout_rate, **kwargs):
        super(EncoderLayer, self).__init__(**kwargs)
        self.input_layer = layers.Dense(hidden_size)
        self.self_attention = MultiHeadAttention(hidden_size, num_heads, dropout_rate)
        self.dense_output = DenseDropoutLayernorm(hidden_size, dropout_rate)

    def call(self, inputs, training=False):
        input_tensor, attention_mask = inputs
        input_tensor = self.input_layer(input_tensor)
        self_outputs = self.self_attention(input_tensor, attention_mask, training=training)
        attention_output = self.dense_output([self_outputs, input_tensor], training=training)
        return attention_output


class TimePositionalEncoding(layers.Layer):
    def __init__(self, hidden_size, max_len=5000):
        """Standard positional encoder,
        source code is from https://www.tensorflow.org/text/tutorials/transformer#the_encoder_layer
        Args:
            hidden_size (int):
            max_len:
        """
        super(TimePositionalEncoding, self).__init__()

        depth = hidden_size / 2
        # [max_len, 1]
        positions = np.arange(max_len)[:, None]
        # [1, depth]
        depths = np.arange(depth)[None, :] / depth
        # [1, depth]
        angle_rates = 1 / (10000 ** depths)
        # [max_len, depth]
        self.angle_rads = positions * angle_rates

        pos_encoding = np.concatenate(
            [np.sin(self.angle_rads), np.cos(self.angle_rads)],
            axis=-1)

        # [1, max_len, model_dim]
        self.pe = tf.cast(pos_encoding, dtype=tf.float32)[None, ...]

    def call(self, inputs, **kwargs):
        """Compute time positional encoding defined in Equation (2) in THP model.

        Args:
            inputs: (tensor), [batch_size, seq_len]
            **kwargs:

        Returns:
            positional encoding

        """

        _, seq_len = get_shape_list(inputs)

        return self.pe[:, :seq_len]


class TimeShiftedPositionalEncoding(TimePositionalEncoding):
    def __init__(self, hidden_size, max_len=5000):
        super(TimeShiftedPositionalEncoding, self).__init__(hidden_size, max_len)
        self.hidden_size = hidden_size

        self.layer_time_delta = layers.Dense(hidden_size // 2)

        depth = self.hidden_size / 2

        # [1, depth]
        self.depths = tf.cast(tf.range(depth)[None, :] / depth, tf.float32)

    def call(self, inputs, **kwargs):
        """Compute time shifted positional encoding defined in Equation (8) in SAHP model

        Args:
            inputs: (time_seqs, time_delta_seqs), [batch_size, seq_len]
            **kwargs:

        Returns:
            positional encoding

        """
        time_seqs, time_delta_seqs = inputs
        _, seq_len = get_shape_list(time_seqs)

        # [batch_size, seq_len, model_dim //2]
        phi = self.layer_time_delta(time_delta_seqs[..., None])

        # [seq_len, 1]
        self.positions = tf.cast(tf.range(seq_len)[:, None], tf.float32)

        # [1, model_dim // 2]
        angle_rates = 1 / (10000 ** self.depths)
        arc = (tf.multiply(self.positions, angle_rates))[None, ...]

        pe_sin = tf.sin(arc + phi)
        pe_cos = tf.cos(arc + phi)
        pe = tf.concat([pe_sin, pe_cos], axis=-1)

        return pe


class DNN(layers.Layer):
    """The Multi Layer Percetron
      Input shape
        - nD tensor with shape: ``(batch_size, ..., input_dim)``.
        The most common situation would be a 2D input with shape ``(batch_size, input_dim)``.
      Output shape
        - nD tensor with shape: ``(batch_size, ..., hidden_size[-1])``.
        For instance, for a 2D input with shape ``(batch_size, input_dim)``,
        the output would have shape ``(batch_size, hidden_size[-1])``.
      Arguments
        - **hidden_units**:list of positive integer, the layer number and units in each layer.
        - **activation**: Activation function to use.
        - **l2_reg**: float between 0 and 1. L2 regularizer strength applied to the kernel weights matrix.
        - **dropout_rate**: float in [0,1). Fraction of the units to dropout.
        - **use_bn**: bool. Whether use BatchNormalization before activation or not.
        - **output_activation**: Activation function to use in the last layer.If ``None``,
        it will be same as ``activation``.
        - **seed**: A Python integer to use as random seed.
    """

    def __init__(self, hidden_size, activation='relu', l2_reg=0, dropout_rate=0, use_bn=False,
                 output_activation=None,
                 **kwargs):
        self.hidden_size = hidden_size if isinstance(hidden_size, list) else [hidden_size]
        self.activation = activation
        self.l2_reg = l2_reg
        self.dropout_rate = dropout_rate
        self.use_bn = use_bn
        self.output_activation = output_activation

        self.dense_layers = [layers.Dense(self.hidden_size[i]) for i in range(len(self.hidden_size))]
        if self.use_bn:
            self.bn_layers = [tf.keras.layers.BatchNormalization() for _ in range(len(self.hidden_size))]

        self.dropout_layers = [tf.keras.layers.Dropout(self.dropout_rate) for _ in
                               range(len(self.hidden_size))]

        self.activation_layers = [activation_layer(self.activation) for _ in range(len(self.hidden_size))]

        if self.output_activation:
            self.activation_layers[-1] = activation_layer(self.output_activation)

        super(DNN, self).__init__(**kwargs)

    def call(self, inputs, training=None, **kwargs):

        deep_input = inputs

        for i in range(len(self.hidden_size)):
            fc = self.dense_layers[i](deep_input)

            if self.use_bn:
                fc = self.bn_layers[i](fc, training=training)

            fc = self.activation_layers[i](fc)

            fc = self.dropout_layers[i](fc, training=training)
            deep_input = fc

        return deep_input
