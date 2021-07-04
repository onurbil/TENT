import time

import numpy as np
import tensorflow as tf
import tensorflow.keras as kr


"""
Based on: A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin, 
'Attention is all you need' 
in Advances in neural information processing systems, 2017, pp. 5998–6008.
"""

def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)


def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Args:
      q: query shape == (..., seq_len_q, depth)
      k: key shape == (..., seq_len_k, depth)
      v: value shape == (..., seq_len_v, depth_v)
      mask: Float tensor with shape broadcastable
            to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
      output, attention_weights
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

        # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, input_size, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(input_size)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention,
                                        perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights


def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, input_size, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(input_size, d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(input_size, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        attn_output, attn_weights = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2, attn_weights


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        # enc_output.shape == (batch_size, input_seq_len, d_model)

        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        attn2, attn_weights_block2 = self.mha2(
            enc_output, enc_output, out1, padding_mask)  # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)

        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)

        return out3, attn_weights_block1, attn_weights_block2


class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, input_size, d_model, num_heads, dff,
                 maximum_position_encoding, rate=0.1):
        super(Encoder, self).__init__()

        self.input_size = input_size
        self.d_model = d_model
        self.num_layers = num_layers

        self.pos_encoding = positional_encoding(maximum_position_encoding, input_size)

        self.enc_layers = [EncoderLayer(input_size, d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        seq_len = tf.shape(x)[1]
        attention_weights = {}

        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, attn_weights = self.enc_layers[i](x, training, mask)
            attention_weights['encoder_layer{}'.format(i + 1)] = attn_weights

        return x, attention_weights  # (batch_size, input_seq_len, d_model)


class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size,
                 maximum_position_encoding, rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)

        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        seq_len = tf.shape(x)[1]
        attention_weights = {}

        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training,
                                                   look_ahead_mask, padding_mask)

            attention_weights['decoder_layer{}_block1'.format(i + 1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i + 1)] = block2

        return x, attention_weights


class Transformer(tf.keras.Model):
    def __init__(self, input_size, num_layers, d_model, num_heads, dff,
                 input_length, output_size, rate=0.1):
        super(Transformer, self).__init__()

        self.encoder = Encoder(num_layers, input_size, d_model, num_heads, dff,
                               input_length, rate)

        self.flatten_layer = tf.keras.layers.Flatten()
        self.final_layer = tf.keras.layers.Dense(output_size, activation='sigmoid')
        self.stop_training = False

    def call(self, inp, training):
        enc_output, attention_weights = self.encoder(inp, training, None)  # (batch_size, inp_seq_len, d_model)

        flatten_output = self.flatten_layer(enc_output)
        final_output = self.final_layer(flatten_output)  # (batch_size, tar_seq_len, target_vocab_size)

        return final_output, attention_weights

    def fit(self, train_x, train_y, epochs, optimizer, loss, validation_data=None, metrics=None, callbacks=None):
        self.stop_training = False
        logs = {}

        train_loss = kr.metrics.Mean(name='loss')
        train_metrics = None
        if metrics is not None:
            train_metrics = [kr.metrics.Mean(name=f'{m}') for m in metrics]

        val_loss = None
        val_metrics = None
        if validation_data is not None:
            val_loss = kr.metrics.Mean(name='val_loss')
            if metrics is not None:
                val_metrics = [kr.metrics.Mean(name=f'val_{m}') for m in metrics]

        if callbacks is not None:
            for callback in callbacks:
                callback.set_model(self)
                callback.on_train_begin()

        train_step_signature = [
            tf.TensorSpec(shape=train_x.shape[1:], dtype=tf.float32),
            tf.TensorSpec(shape=train_y.shape[1:], dtype=tf.float32),
        ]

        # @tf.function(input_signature=train_step_signature)
        def train_step(inp, tar):
            with tf.GradientTape() as tape:
                predictions, _ = self(inp, True)
                loss_value = loss(tar, predictions)

            gradients = tape.gradient(loss_value, self.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self.trainable_variables))

            train_loss(loss_value)
            if metrics is not None:
                for m, tm in zip(metrics.values(), train_metrics):
                    tm(m(tar, predictions))

        for epoch in range(epochs):
            logs.clear()

            start = time.time()

            train_loss.reset_states()
            if metrics is not None:
                for tm in train_metrics:
                    tm.reset_states()

            for (batch, (inp, tar)) in enumerate(zip(train_x, train_y)):
                train_step(inp, tar)

                if batch % 50 == 0:
                    self.print_batch(epoch, epochs, batch, train_loss, train_metrics)

            logs['loss'] = train_loss.result()
            if train_metrics is not None:
                for tm in train_metrics:
                    logs[tm.name] = tm.result()

            if validation_data is not None:
                val_loss.reset_states()
                if metrics is not None:
                    for vm in val_metrics:
                        vm.reset_states()

                for valid_x, valid_y in zip(validation_data[0], validation_data[1]):
                    valid_predictions, _ = self(valid_x, False)
                    valid_loss_value = loss(valid_y, valid_predictions)
                    val_loss(valid_loss_value)

                    if metrics is not None:
                        for m, vm in zip(metrics.values(), val_metrics):
                            vm(m(valid_y, valid_predictions))

                logs[val_loss.name] = val_loss.result()
                if val_metrics is not None:
                    for vm in val_metrics:
                        logs[vm.name] = vm.result()

            self.print_epoch(epoch, epochs, train_loss, train_metrics, val_loss, val_metrics)
            training_time = time.time() - start
            print(f'Time taken for 1 epoch: {training_time} secs\n')

            if callbacks is not None:
                for callback in callbacks:
                    callback.on_epoch_end(epoch, logs)

            if self.stop_training:
                print('Model stop_training set to True. Stopping!')
                break

    def print_batch(self, epoch, epochs, batch, train_loss, train_metrics):
        train_string = f'{train_loss.name}: {train_loss.result():.4f}'
        if train_metrics is not None:
            for tm in train_metrics:
                train_string += f' - {tm.name}: {tm.result():.4f}'

        print(f'{epoch + 1} / {epochs} - Batch {batch} - {train_string}')

    def print_epoch(self, epoch, epochs, train_loss, train_metrics, valid_loss, valid_metrics):
        train_string = f'{train_loss.name}: {train_loss.result():.4f}'
        if train_metrics is not None:
            for tm in train_metrics:
                train_string += f' - {tm.name}: {tm.result():.4f}'

        valid_string = ''
        if valid_loss is not None:
            valid_string += f' - {valid_loss.name}: {valid_loss.result():.4f}'
            if valid_metrics is not None:
                for vm in valid_metrics:
                    valid_string += f' - {vm.name}: {vm.result():.4f}'

        print(f'Epoch {epoch + 1} / {epochs} {train_string}{valid_string}')


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
