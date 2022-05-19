import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
import tensorflow.keras as kr
import common.paths


"""
The implementation of TENT model.
"""


def get_angles(pos, i, c):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(c))
    return pos * angle_rates


class PositionalEncoding(kr.layers.Layer):
    def __init__(self, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)

    def build(self, input_shape):
        self.t = input_shape[-3]
        self.c = input_shape[-2]
        self.model_shape = input_shape
        self.batch_size = input_shape[0]

    def call(self, input_data):
        angle_rads = get_angles(np.arange(self.t)[:, np.newaxis],
                                np.arange(self.c)[np.newaxis, :],
                                self.c)

        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        angle_rads = np.broadcast_to(np.expand_dims(angle_rads, -1), input_data.shape[1:])

        pos_encoding = tf.broadcast_to(angle_rads, tf.shape(input_data))
        pos_encoding = tf.cast(pos_encoding, input_data.dtype)

        return tf.math.add(input_data, pos_encoding)

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class EncoderLayer(kr.layers.Layer):
    def __init__(self,
                 input_length,
                 d_model,
                 head_num,
                 dense_units,
                 initializer,
                 softmax_type=3,
                 batch_size=32,
                 save_attention=False,
                 **kwargs):
        super(EncoderLayer, self).__init__(**kwargs)

        assert (d_model % head_num == 0), 'The division of d_k and head number must be an integer.'

        self.input_length = input_length
        self.d_model = d_model
        self.head_num = head_num
        self.dense_units = dense_units
        self.initializer1 = initializer
        self.initializer = tf.keras.initializers.get(initializer)
        self.softmax_type = softmax_type
        self.batch_size = batch_size
        self.save_attention = save_attention

        self.wq = None
        self.wk = None
        self.wv = None
        self.wo = None
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.flatten = tf.keras.layers.Flatten()
        self.dense_hidden = tf.keras.layers.Dense(dense_units, activation='relu')
        self.dense_out: tf.keras.layers.Dense = None
        self.reshape: tf.keras.layers.Reshape = None

    def build(self, input_shape):
        self.inp_shape = input_shape

        self.z_all = tf.zeros([input_shape[-2], input_shape[-1], 0])
        self.wq = self.add_weight(
            shape=(input_shape[-2], input_shape[-1], self.d_model),
            initializer=self.initializer,
            name="wq",
            trainable=True,
        )
        self.wk = self.add_weight(
            shape=(input_shape[-2], input_shape[-1], self.d_model),
            initializer=self.initializer,
            name="wk",
            trainable=True,
        )
        self.wv = self.add_weight(
            shape=(input_shape[-2], input_shape[-1], self.d_model),
            initializer=self.initializer,
            name="wv",
            trainable=True,
        )
        self.wo = self.add_weight(
            shape=(self.input_length, self.d_model, input_shape[-1]),
            initializer=self.initializer,
            name="wo",
            trainable=True,
        )
        self.dense_out = tf.keras.layers.Dense(tf.reduce_prod(input_shape[-3:]), activation='relu')
        self.reshape = tf.keras.layers.Reshape(input_shape[-3:])

        if self.softmax_type in (1, 2):
            attention_shape = self.head_num, self.batch_size, self.input_length, self.input_length

        if self.softmax_type == 3:
            attention_shape = (self.head_num, self.batch_size, self.input_length, self.input_length, input_shape[-2])

        self.attention_weights = tf.Variable(initial_value=tf.raw_ops.Empty(shape=attention_shape, dtype=tf.float32),
                                             trainable=False)

        self.d_k = int(self.d_model / self.head_num)

    def call(self, inputs, training=True):
        inputs = tf.expand_dims(inputs, -2)
        q = tf.squeeze(tf.matmul(inputs, self.wq), -2)
        k = tf.squeeze(tf.matmul(inputs, self.wk), -2)
        v = tf.squeeze(tf.matmul(inputs, self.wv), -2)
        inputs = tf.squeeze(inputs, -2)

        zs = []
        aw_list = []
        batch_size = tf.shape(inputs)[0]

        for i in range(self.head_num):
            index = i * self.d_k

            zz, aw = self.self_attention(batch_size, q[..., index:index + self.d_k],
                                         k[..., index:index + self.d_k],
                                         v[..., index:index + self.d_k],
                                         softmax_type=self.softmax_type)

            zs.append(zz)
            aw_list.append(aw)

        z = tf.concat(zs, axis=-1)
        if self.save_attention and not training:
            aww = tf.stack(aw_list, axis=0)
            self.attention_weights.assign(aww)

        z = tf.matmul(z, self.wo)

        # 1. Residual:
        sum = tf.math.add(inputs, z)
        norm = self.layer_norm(sum)

        # linear
        x = self.flatten(norm)
        x = self.dense_hidden(x)
        x = self.dense_out(x)
        out = self.reshape(x)
        # 2. Residual:
        sum = tf.math.add(norm, out)
        out = self.layer_norm(sum)

        return out

    def self_attention(self, batch_size, q, k, v, softmax_type=1):
        """
        Calculates self attention:
        """
        q_shape = tf.shape(q)
        q_expanded_shape = (q_shape[0], q_shape[1], q_shape[1], q_shape[2], q_shape[3])
        k_expanded_shape = (q_shape[0], q_shape[1], q_shape[1], q_shape[3], q_shape[2])
        kt = tf.transpose(k, perm=(0, 1, 3, 2))
        qe = tf.expand_dims(q, axis=-3)
        qe = tf.broadcast_to(qe, q_expanded_shape)
        ke = tf.expand_dims(kt, axis=-4)
        ke = tf.broadcast_to(ke, k_expanded_shape)
        z = tf.matmul(qe, ke)

        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        z = z / tf.math.sqrt(dk)

        if softmax_type == 1:
            z = tf.reduce_sum(z, axis=[-1, -2])
            se = tf.nn.softmax(z, axis=-1)
            # Attention weights to plot:
            attention_weights = se
            se = tf.expand_dims(se, -1)


        elif softmax_type == 2:
            # Softmax over all measurements:
            z = tf.reduce_sum(z, axis=[-1, -2])
            z_shape = tf.shape(z)
            z = tf.reshape(z, [z_shape[0], -1])
            se = tf.nn.softmax(z, axis=-1)
            se = tf.reshape(se, [z_shape[0], z_shape[1], z_shape[2]])
            # Attention weights to plot:
            attention_weights = se
            se = tf.expand_dims(se, -1)

        elif softmax_type == 3:
            z = tf.reduce_sum(z, axis=-1)
            se = tf.nn.softmax(z, axis=-1)
            # Attention weights to plot:
            attention_weights = se

        se = tf.expand_dims(se, -1)

        v = tf.expand_dims(v, 1)
        ve = tf.broadcast_to(v, (batch_size, v.shape[-3], v.shape[-3], v.shape[-2], v.shape[-1]))
        se = tf.broadcast_to(se, (batch_size, v.shape[-3], v.shape[-3], v.shape[-2], v.shape[-1]))

        z = tf.multiply(se, ve)
        z = tf.reduce_sum(z, axis=2)

        return z, attention_weights

    def get_config(self):
        config = {
            'input_length': self.input_length,
            'd_model': self.d_model,
            'head_num': self.head_num,
            'dense_units': self.dense_units,
            'initializer': self.initializer1,
            'softmax_type': self.softmax_type,
            'save_attention': self.save_attention,
        }
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self,
                 d_model,
                 warmup_steps=50,
                 factor1=-0.6,
                 factor2=-1.5):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model1 = tf.cast(d_model, tf.float32)

        self.warmup_steps = warmup_steps
        self.factor1 = factor1
        self.factor2 = factor2

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** self.factor2)

        return (self.d_model1 ** self.factor1) * tf.math.minimum(arg1, arg2)

    def get_config(self):
        config = {
            'd_model': self.d_model,
            'warmup_steps': self.warmup_steps,
            'factor1': self.factor1,
            'factor2': self.factor2,
        }
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer._decayed_lr(tf.float32) # I use ._decayed_lr method instead of .lr
    return lr


def main():
    # Load USA+Canada dataset:
    import experiment_tools.load_dataset as load_dataset
    from common.variables import city_labels

    input_length = 16
    prediction_time = 4
    y_feature = 4
    y_city = 0
    num_cities = 30
    remove_last_from_test = 800
    valid_size = 1024

    dataset, dataset_params = load_dataset.get_usa_dataset(common.paths.PROCESSED_DATASET_DIR,
                                                           input_length, prediction_time,
                                                           y_feature, y_city,
                                                           end_city=num_cities,
                                                           remove_last_from_test=remove_last_from_test,
                                                           valid_split=valid_size, split_random=1337)

    denorm_min, denorm_max = load_dataset.get_usa_normalization(common.paths.PROCESSED_DATASET_DIR, y_feature)

    Xtr, Ytr, Xvalid, Yvalid, Xtest, Ytest = dataset
    print('Xtr.shape', Xtr.shape)
    print('Ytr.shape', Ytr.shape)
    print('Xvalid.shape', Xvalid.shape)
    print('Yvalid.shape', Yvalid.shape)
    print('Xtest.shape', Xtest.shape)
    print('Ytest.shape', Ytest.shape)

    print('denorm_min', denorm_min)
    print('denorm_max', denorm_max)

    # Create and train the TENT model
    import experiment_tools.tt_training as tt_training
    from visualization_tools.AW_save import save_weights
    import datetime

    save_aw = False  ## To store the attention weights set this variable to true
    folder = datetime.datetime.now().strftime("%Y%m%d") + '_' + datetime.datetime.now().strftime("%H%M%S")

    # model
    softmax_type = 3
    epoch = 300
    patience = 20
    num_layers = 3
    head_num = 32
    d_model = 256
    dense_units = 128
    batch_size = 16
    loss = 'mse'

    model, model_params, history = tt_training.train_model(dataset,
                                                           softmax_type, epoch, patience,
                                                           num_layers, head_num, d_model, dense_units,
                                                           batch_size, loss, use_tpu=False, save_aw=save_aw)
    if save_aw:
        save_weights(model, city_labels, layer=1,
                     folder_name=common.paths.WORKING_DIR + folder)


if __name__ == '__main__':
    main()
