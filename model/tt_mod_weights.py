import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from debugging_tools import *

import numpy as np
import tensorflow as tf
import tensorflow.keras as kr
import matplotlib.pyplot as plt
import dataset_tools.split
import attention.self_attention
import common.paths
from visualization_tools.visualization import visualize_pos_encoding, attention_plotter, attention_3d_plotter
from tensorflow.keras.callbacks import LambdaCallback

def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates

class PositionalEncoding(kr.layers.Layer):
    def __init__(self,
                 broadcast,
                 **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.broadcast = broadcast

    def build(self, input_shape):
        self.position = input_shape[-3]
        self.angle_dim = input_shape[-2]
        if not self.broadcast:
            self.angle_dim *= input_shape[-1]

        self.model_shape = input_shape
        self.batch_size = input_shape[0]

    def call(self, input_data):
        angle_rads = get_angles(np.arange(self.position)[:, np.newaxis],
                                np.arange(self.angle_dim)[np.newaxis, :],
                                self.angle_dim)
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        if self.broadcast:
            angle_rads = np.broadcast_to(np.expand_dims(angle_rads, -1), input_data.shape[1:])
        else:
            new_shape = angle_rads.shape[:-1] + self.model_shape
            angle_rads = np.reshape(angle_rads, new_shape)

        pos_encoding = tf.broadcast_to(angle_rads, tf.shape(input_data))
        pos_encoding = tf.cast(pos_encoding, input_data.dtype)

        return tf.math.add(input_data, pos_encoding)

class EncoderLayer(kr.layers.Layer):
    def __init__(self,
                 input_length,
                 d_model,
                 head_num,
                 dense_units,
                 initializer,
                 softmax_type,
                 **kwargs):
        super(EncoderLayer, self).__init__(**kwargs)

        # TODO: add meaningful description to the assertion
        assert (d_model % head_num == 0)

        self.input_length = input_length
        self.d_model = d_model
        self.head_num = head_num
        self.initializer = tf.keras.initializers.get(initializer)
        self.softmax_type = softmax_type

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
            attention_shape = self.head_num, 0, self.input_length, self.input_length

        if self.softmax_type == 3:
            attention_shape = (self.head_num, 0, self.input_length, self.input_length, input_shape[-2])

        self.attention_weights = tf.Variable(initial_value=tf.raw_ops.Empty(shape=attention_shape, dtype=tf.float32),
                                             trainable=False)

        # self.attention_weights = tf.Variable(initial_value=tf.raw_ops.Empty(
        #     shape=(self.head_num,0,self.input_length, self.input_length),
        #     dtype=tf.float32),
        #     trainable=False)
        self.d_k = int(self.d_model / self.head_num)

    def call(self, inputs):
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

    def self_attention(self, batch_size, q, k, v, mask=None, softmax_type=1):
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

        # if mask is not None:
        #     z += (mask * -1e9)

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

        # z = tf.matmul(se, v)
        z = tf.multiply(se, ve)
        z = tf.reduce_sum(z, axis=2)

        return z, attention_weights


def custom_loss_function(lambada):
    
    def mse_loss_function(y_true, y_pred):
        
        loss = tf.math.reduce_mean(tf.square(y_true-y_pred)) + lambada * tf.square(tf.math.reduce_mean(y_true-y_pred))   
        return loss
    
    return mse_loss_function




if __name__ == '__main__':
    # Load dataset:
    filename = 'dataset_tensor.npy'
    file_path = os.path.join(common.paths.PROCESSED_DATASET_DIR, filename)
    dataset = np.load(file_path, allow_pickle=True)
    print(dataset.shape)

    """
    ###### ALL PARAMETERS HERE######:
    """

    softmax_type = 2
    input_length = 16
    lag = 4
    epoch = 100

    learning_rate = 0.0001
    head_num = 16
    d_model = 32
    dense_units = 64
    batch_size = 64

    num_examples = 10000
    num_valid_examples = 500
    initializer = 'RandomNormal'

    train, test = dataset_tools.split.split_train_test(dataset)
    x_train, y_train = dataset_tools.split.get_xy(train, input_length=input_length, lag=lag)
    x_test, y_test = dataset_tools.split.get_xy(test, input_length=input_length, lag=lag)

    #x_train = x_train.astype('float32')
    x_train = tf.reshape(x_train, (x_train.shape[0], x_train.shape[1], dataset.shape[1], dataset.shape[2]))
    y_train = tf.reshape(y_train, (y_train.shape[0], dataset.shape[1], dataset.shape[2]))
    x_test = tf.reshape(x_test, (x_test.shape[0], x_test.shape[1], dataset.shape[1], dataset.shape[2]))
    y_test = tf.reshape(y_test, (y_test.shape[0], dataset.shape[1], dataset.shape[2]))

    # Choosing first 29 cities
    x_train = x_train[:, :, :29, :]
    y_train = y_train[:, :29, :]
    x_test = x_test[:, :, :29, :]
    y_test = y_test[:, :29, :]

    input_shape = (input_length, x_train.shape[-2], x_train.shape[-1])
    output_shape = (1, 1)

    # Choosing temperature as output
    y_train = y_train[..., 0, 4]
    y_test = y_test[..., 0, 4]

    print(f'x_train.shape: {x_train.shape}')
    print(f'x_test.shape: {x_test.shape}')

    model = kr.Sequential([
        kr.Input(shape=input_shape),
        PositionalEncoding(broadcast=True),
        EncoderLayer(input_length, d_model, head_num, dense_units, initializer, softmax_type),
        EncoderLayer(input_length, d_model, head_num, dense_units, initializer, softmax_type),
        EncoderLayer(input_length, d_model, head_num, dense_units, initializer, softmax_type),
        EncoderLayer(input_length, d_model, head_num, dense_units, initializer, softmax_type),
        EncoderLayer(input_length, d_model, head_num, dense_units, initializer, softmax_type),
        EncoderLayer(input_length, d_model, head_num, dense_units, initializer, softmax_type),
        kr.layers.Flatten(),
        kr.layers.Dense(tf.reduce_prod(output_shape), activation='linear'),
        kr.layers.Reshape(output_shape),
    ])

    model.summary()
    model.compile(optimizer=kr.optimizers.Adam(learning_rate=learning_rate), loss='mse', metrics=['mae'])

    x_valid = x_train[-num_examples - num_valid_examples:-num_examples, ...]
    y_valid = y_train[-num_examples - num_valid_examples:-num_examples]
    print(f'x_valid.shape: {x_valid.shape}')

    x_train = x_train[-num_examples:]
    y_train = y_train[-num_examples:]

    # Callbacks
    print_attention_weights = kr.callbacks.LambdaCallback(
        on_train_end=lambda batch: print(model.layers[1].attention_weights))
    early_stopping = kr.callbacks.EarlyStopping(patience=10,
                                                restore_best_weights=True,
                                                verbose=1)

    history = model.fit(
        x_train, y_train,
        epochs=epoch,
        batch_size=batch_size,
        validation_data=(x_valid, y_valid),
        callbacks=[early_stopping]
    )


    labels = np.arange(model.layers[1].attention_weights.shape[-2]).tolist()
    
    if (softmax_type == 1 or softmax_type == 2):
        attention_plotter(tf.reshape(model.layers[1].attention_weights[1][0], (input_length,-1)), labels)
        attention_plotter(tf.reshape(model.layers[1].attention_weights[2][0], (input_length,-1)), labels)
        attention_plotter(tf.reshape(model.layers[1].attention_weights[3][0], (input_length,-1)), labels)        
        attention_plotter(tf.reshape(model.layers[1].attention_weights[4][0], (input_length,-1)), labels)        

    elif softmax_type == 3:
        # print(model.layers[1].attention_weights[0][3].numpy())
        attention_3d_plotter(model.layers[1].attention_weights[0][3].numpy(), city_labels)
    else:
        pass
        
        

    preds = []
    for i in range(x_valid.shape[0]):
        if (i + 1) % 100 == 0:
            print(f'prediction: {i + 1}/{x_valid.shape[0]}')
        preds.append(model.predict(x_valid[i][np.newaxis, ...]))
    pred = np.concatenate(preds, axis=0)
    mse = np.mean(kr.metrics.mse(y_valid, pred))
    mae = np.mean(kr.metrics.mae(y_valid, pred))
    print(f'mse: {mse}, mae: {mae}')

    plt.figure(figsize=(14, 8))
    plt.plot(range(pred.size), pred.flatten(), label='pred')
    plt.plot(range(len(y_valid)), y_valid, label='true')
    plt.legend()
    plt.show()
    
    
    print("\n\n######################## Model description ################################")
    model.summary()
    print("softmax_type = ", softmax_type)
    print("Input_length = ", input_length)
    print("Lag = ", lag)
    print("Epoch = ", epoch)

    print("LR = ", learning_rate)
    print("Head_num = ", head_num)
    print("d_model = ", d_model)
    print("dense_units = ", dense_units)
    print("batch_size = ", batch_size)

    print("num_examples = ", num_examples)
    print("num_valid_examples = ", num_valid_examples)
    print("input_shape = ", input_shape)

    pred = model.predict(x_test)
    mae = kr.metrics.mae(y_test.numpy().flatten(), pred.flatten())
    print("\n\n######################## Results ##########################################")
    print(f'test mae: {np.mean(mae)}')

