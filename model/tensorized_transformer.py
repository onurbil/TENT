import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from debugging_tools import *

import numpy as np
import tensorflow as tf
import tensorflow.keras as kr

import dataset_tools.split
import attention.self_attention
import common.paths
from visualization_tools.visualization import visualize_pos_encoding, attention_plotter
from keras.callbacks import LambdaCallback


      
# class SelfAttentionLayer(kr.layers.Layer):
#     def __init__(self,
#                  input_length,
#                  d_model,
#                  head_num,
#                  input_shape,
#                  #mask,
#                  **kwargs):
#         super(SelfAttentionLayer, self).__init__(**kwargs)
#
#         self.input_length = input_length
#         self.d_model = d_model
#         self.head_num = head_num
#
#         self.wq = None
#         self.wk = None
#         self.wv = None
#         self.wo = None
#
#     def build(self, input_shape):
#         # self.z_all = tf.zeros([input_shape[-2], input_shape[-1], 0])
#         self.wq = tf.random.normal([input_shape[-1], d_model], mean=0.0, stddev=1.0)
#         self.wk = tf.random.normal([input_shape[-1], d_model], mean=0.0, stddev=1.0)
#         self.wv = tf.random.normal([input_shape[-1], d_model], mean=0.0, stddev=1.0)
#         self.wo = tf.random.normal([self.input_length, self.d_model, input_shape[-1]], mean=0.0, stddev=1.0)
#
#     def call(self, inputs):
#         q = tf.matmul(inputs, self.wq)
#         k = tf.matmul(inputs, self.wk)
#         v = tf.matmul(inputs, self.wv)
#
#         qe = tf.broadcast_to(q, (q.shape[-3], q.shape[-3], q.shape[-2], q.shape[-1]))
#         kt = tf.transpose(k, perm=(0, 1, 3, 2))
#         kt = tf.broadcast_to(kt, (kt.shape[-3], kt.shape[-3], kt.shape[-2], kt.shape[-1]))
#         z = tf.matmul(qe, kt)
#
#         dk = tf.cast(tf.shape(k)[-1], tf.float32)
#         z = z / tf.math.sqrt(dk)
#         # if mask is not None:
#         #     z += (mask * -1e9)
#
#         z = tf.reduce_sum(z, axis=[-1, -2])
#         se = tf.nn.softmax(z, axis=-1)
#         se = tf.expand_dims(se, -1)
#         se = tf.expand_dims(se, -1)
#
#         ve = tf.broadcast_to(v, [v.shape[-3], v.shape[-3], v.shape[-2], v.shape[-1]])
#         se = tf.broadcast_to(se, (v.shape[-3], v.shape[-3], v.shape[-2], v.shape[-1]))
#         # z = tf.matmul(se, v)
#         z = tf.multiply(se, ve)
#         z = tf.reduce_sum(z, axis=1)
#
#         z = tf.matmul(z, self.wo)
#
#         return z

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
        
        pos_encoding = tf.broadcast_to(angle_rads,tf.shape(input_data))
        pos_encoding = tf.cast(pos_encoding, input_data.dtype)
                
        return tf.math.add(input_data, pos_encoding)


class EncoderLayer(kr.layers.Layer):
    def __init__(self,
                 input_length,
                 d_model,
                 head_num,
                 dense_units,
                 initializer,
                 **kwargs):
        super(EncoderLayer, self).__init__(**kwargs)

        # TODO: add meaningful description to the assertion
        assert (d_model % head_num == 0)

        self.input_length = input_length
        self.d_model = d_model
        self.head_num = head_num
        self.initializer = tf.keras.initializers.get(initializer)
        
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
        self.z_all = tf.zeros([input_shape[-2], input_shape[-1], 0])
        self.wq = self.add_weight(
            shape=(input_shape[-1], self.d_model),
            initializer=self.initializer,
            name="wq",
            trainable=True,
        )
        self.wk = self.add_weight(
            shape=(input_shape[-1], self.d_model),
            initializer=self.initializer,
            name="wk",
            trainable=True,
        )
        self.wv = self.add_weight(
            shape=(input_shape[-1], self.d_model),
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
        
        self.attention_weights = tf.Variable(initial_value=tf.zeros((0,self.input_length, self.input_length)), trainable=False)

    def call(self, inputs):

        q = tf.matmul(inputs, self.wq)
        k = tf.matmul(inputs, self.wk)
        v = tf.matmul(inputs, self.wv)

        d_k = int(self.d_model / self.head_num)
        zs = []
        batch_size = tf.shape(inputs)[0]
        
        for i in range(self.head_num):
            index = i * d_k
            zs.append(self.self_attention(batch_size, q[..., index:index + d_k],
                                          k[..., index:index + d_k],
                                          v[..., index:index + d_k]))
        z = tf.concat(zs, axis=-1)
        # z = self.self_attention(q, k, v)
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

    def self_attention(self, batch_size, q, k, v, mask=None):
        """
        Calculates self attention:
        """
        # q = tf.expand_dims(q, 1)
        # qe = tf.broadcast_to(q, (batch_size, q.shape[-3], q.shape[-3], q.shape[-2], q.shape[-1]))
        # kt = tf.transpose(k, perm=(0, 1, 3, 2))
        # kt = tf.expand_dims(kt, 1)
        # kt = tf.broadcast_to(kt, (batch_size, kt.shape[-3], kt.shape[-3], kt.shape[-2], kt.shape[-1]))
        # z = tf.matmul(qe, kt)

        kt = tf.transpose(k, perm=(0, 1, 3, 2))
        
        d1 = tf.raw_ops.Empty(shape=(0,q.shape[1], q.shape[1], q.shape[2], q.shape[2]), dtype=q.dtype)
        for b in range(batch_size):
            tf.autograph.experimental.set_loop_options(shape_invariants=[(d1, tf.TensorShape((None,q.shape[1], q.shape[1], q.shape[2], q.shape[2])))])

        
            d2 = tf.raw_ops.Empty(shape=(0,q.shape[1], q.shape[2], q.shape[2]), dtype=q.dtype)
            for t in range(q.shape[1]):
                # tf.autograph.experimental.set_loop_options(shape_invariants=[(d2, tf.TensorShape((None,q.shape[1], q.shape[2], q.shape[2]), dtype=q.dtype)))])
                
                aa = tf.matmul(q[b,t],kt[b])
                aa = tf.expand_dims(aa,0)
                d2 = tf.concat([d2,aa], axis=0)
                # tf.print(tf.shape(d2))
            
            d2 = tf.expand_dims(d2,0) 
            d1 = tf.concat([d1,d2], axis=0)    
        
        # tf.print(d1)
        
        z=d1
        
        # q = tf.expand_dims(q, 1)
        # qe = tf.broadcast_to(q, (batch_size, q.shape[-3], q.shape[-3], q.shape[-2], q.shape[-1]))
        # kt = tf.transpose(k, perm=(0, 1, 3, 2))
        # kt = tf.expand_dims(kt, 1)
        # kt = tf.broadcast_to(kt, (batch_size, kt.shape[-3], kt.shape[-3], kt.shape[-2], kt.shape[-1]))
        # z = tf.matmul(qe, kt)
        
        
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        z = z / tf.math.sqrt(dk)
        if mask is not None:
            z += (mask * -1e9)
        z = tf.reduce_sum(z, axis=[-1, -2])
        
        se = tf.nn.softmax(z, axis=-1)
        # Attention weights to plot:
        self.attention_weights.assign(se)
                        
        se = tf.expand_dims(se, -1)
        se = tf.expand_dims(se, -1)
        
        v = tf.expand_dims(v, 1)
        ve = tf.broadcast_to(v, (batch_size, v.shape[-3], v.shape[-3], v.shape[-2], v.shape[-1]))
        se = tf.broadcast_to(se, (batch_size, v.shape[-3], v.shape[-3], v.shape[-2], v.shape[-1]))

        # z = tf.matmul(se, v)
        z = tf.multiply(se, ve)
        z = tf.reduce_sum(z, axis=2)

        return z


# def get_config(self):
#     # Implement get_config to enable serialization. This is optional.
#     # base_config = super(Antirectifier, self).get_config()
#     # config = {"initializer": keras.initializers.serialize(self.initializer)}
#     # return dict(list(base_config.items()) + list(config.items()))
#     pass



def custom_loss_function(lambada):
    
    # print(model.predict(x_train[batch][np.newaxis, ...]))
    # tf.print(previous_pred[0])
    # tf.print(lambada)
    
    def mse_loss_function(y_true, y_pred):

        loss = tf.keras.backend.mean(tf.math.reduce_sum(tf.square(y_true-y_pred))) #+ lambada * (y_pred-previous_pred)            
        return loss
    
    return mse_loss_function
        

if __name__ == '__main__':
    # Load dataset:
    filename = 'dataset_tensor.npy'
    file_path = os.path.join(common.paths.PROCESSED_DATASET_DIR, filename)
    dataset = np.load(file_path, allow_pickle=True)








    # Get x_train, y_train, x_test, y_test:
    input_length = 24
    lag = 4
    train, test = dataset_tools.split.split_train_test(dataset)
    x_train, y_train = dataset_tools.split.get_xy(train, input_length=input_length, lag=lag)
    x_test, y_test = dataset_tools.split.get_xy(test, input_length=input_length, lag=lag)

    x_train = x_train.astype('float32')
    x_train = tf.reshape(x_train, (x_train.shape[0], x_train.shape[1], dataset.shape[1], dataset.shape[2]))
    y_train = tf.reshape(y_train, (y_train.shape[0], dataset.shape[1], dataset.shape[2]))
    x_test = tf.reshape(x_test, (x_test.shape[0], x_test.shape[1], dataset.shape[1], dataset.shape[2]))
    y_test = tf.reshape(y_test, (y_test.shape[0], dataset.shape[1], dataset.shape[2]))

    print(f'x_train.shape: {x_train.shape}')
    print(f'x_test.shape: {x_test.shape}')

    # Parameters:
    epoch = 20
    learning_rate = 0.001
    d_model = 1
    head_num = 1
    dense_units = 64
    batch_size = 64
    input_shape = (input_length, x_train.shape[-2], x_train.shape[-1])
    # output_shape = (36, x_train.shape[-1])
    # output_shape = (1, x_train.shape[-1])
    output_shape = (1, 1)
    y_train = y_train[..., 0, 2]
    y_test = y_test[..., 0, 2]
    initializer = 'RandomNormal'

    # x_train = np.zeros((1,) + input_shape)
    # y_train = np.zeros((1,) + input_shape[1:])

    model = kr.Sequential([
        kr.Input(input_shape),
        PositionalEncoding(broadcast=True),
        EncoderLayer(input_length, d_model, head_num, dense_units, initializer),
        # EncoderLayer(input_length, d_model, head_num, dense_units, initializer),
        # EncoderLayer(input_length, d_model, head_num, dense_units, initializer),
        kr.layers.Flatten(),
        kr.layers.Dense(tf.reduce_prod(output_shape), activation='linear'),
        kr.layers.Reshape(output_shape),
    ])

    model.summary()
    # model.compile(optimizer='sgd', loss='mse', metrics=['mae'])
    model.compile(optimizer=kr.optimizers.Adam(learning_rate=learning_rate), loss='mse', metrics=['mae'])


    num_examples = 1000
    x_train = x_train[:num_examples]
    y_train = y_train[:num_examples]

    num_test_examples = 100
    x_test = x_test[:num_test_examples, ...]
    y_test = y_test[:num_test_examples]

    # print_weights = kr.callbacks.LambdaCallback(on_epoch_end=lambda batch, logs: print(model.layers[1].get_weights()[0]))
    print_attention_weights = kr.callbacks.LambdaCallback(on_train_end=lambda batch: print(model.layers[1].attention_weights))

    model.fit(x_train, y_train, epochs=epoch, batch_size=batch_size, callbacks=[print_attention_weights])
    # model.fit(x_train, y_train, epochs=epoch, batch_size=batch_size, validation_data=(x_test,y_test), callbacks=[print_attention_weights])

    # print(model.layers[1].attention_weights)
    labels = np.arange(model.layers[1].attention_weights.shape[1]).tolist()
    print(tf.shape(model.layers[1].attention_weights))
    attention_plotter(model.layers[1].attention_weights[5], labels)
    
    # pred = model.predict(x_test[0])

    preds = []
    for i in range(x_test.shape[0]):
        if (i + 1) % 100 == 0:
            print(f'prediction: {i + 1}/{x_test.shape[0]}')
        preds.append(model.predict(x_test[i][np.newaxis, ...]))
    pred = np.concatenate(preds, axis=0)
    mse = np.mean(kr.metrics.mse(y_test, pred))
    mae = np.mean(kr.metrics.mae(y_test, pred))
    print(f'mse: {mse}, mae: {mae}')
    # print(pred.flatten())
    # print(model.layers[1].get_weights()[0])


    import matplotlib.pyplot as plt
    
    print(pred.flatten().shape)
    print(y_test.shape)
    
    plt.plot(range(pred.size), pred.flatten(), label='pred')
    plt.plot(range(len(y_test)), y_test, label='true')
    plt.legend()
    plt.show()
