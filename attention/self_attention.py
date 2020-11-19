import numpy as np
import os

from numpy.core._multiarray_umath import broadcast

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import tensorflow.keras.backend
import keras.layers
from common.paths import PROCESSED_DATASET_DIR
from dataset_tools.split import split_train_test, get_xy
from visualization_tools.visualization import visualize_pos_encoding
from debugging_tools import *


"""
Self attention implementation:
"""

def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates

def positional_encoding(position, model_shape, broadcast=True):
    """
    Copied from tutorial. The dimension doesnt change the implementation of
    the positional encoding. Only the correct dimension (time dimension) must
    sent as "position" input to positional_encoding.
    d_model is hyperparameter.
    """

    angle_dim = model_shape[0]
    if not broadcast:
        angle_dim *= model_shape[1]

    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(angle_dim)[np.newaxis, :],
                            angle_dim)
    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    if broadcast:
        angle_rads = np.broadcast_to(np.expand_dims(angle_rads, -1), angle_rads.shape + (model_shape[1],))
    else:
        new_shape = angle_rads.shape[:-1] + model_shape
        angle_rads = np.reshape(angle_rads, new_shape)

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


def qkv_matrices(x,d_model):
    """
    Calculate query, key and value vectors:
    d_model: Hyperparameter
    """
    wq = tf.random.normal([x.shape[2],d_model], mean=0.0, stddev=1.0)
    q = tf.matmul(x,wq)

    wk = tf.random.normal([x.shape[2],d_model], mean=0.0, stddev=1.0)
    k = tf.matmul(x,wk)

    wv = tf.random.normal([x.shape[2],d_model], mean=0.0, stddev=1.0)
    v = tf.matmul(x,wv)
    
    return q,k,v

def self_attention(q,k,v,mask=None):
    """
    Calculates self attention:
    """
    qe = tf.broadcast_to(q, [q.shape[0], q.shape[0], q.shape[1], q.shape[2]])
    kt = tf.transpose(k, perm=[0, 2, 1])
    kt = tf.broadcast_to(kt, [kt.shape[0], kt.shape[0], kt.shape[1], kt.shape[2]])
    z = tf.matmul(qe,kt)

    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    z = z / tf.math.sqrt(dk)
    if mask is not None:
        z += (mask * -1e9) 
    # !!! Test reduce_sum vs reduce_mean
    z = tf.reduce_sum(z, axis=[-1, -2])  
    se = tf.nn.softmax(z, axis=-1) 
    se = tf.expand_dims(se, -1)
    se = tf.expand_dims(se, -1)
     
    # Option 1 (Comment option 1 or option 2): 
    # ve = tf.broadcast_to(v, [v.shape[0], v.shape[0], v.shape[1], v.shape[2]])    
    # se = tf.broadcast_to(se, [v.shape[0], v.shape[0], v.shape[1], v.shape[2]])
    # z = tf.multiply(se, ve)
    ##
    # Option 2:     
    se = tf.broadcast_to(se, [v.shape[0], v.shape[0], v.shape[1], v.shape[1]])
    z = tf.matmul(se,v)
    ##
    
    z = tf.reduce_sum(z, axis=1)
        
    return z


def multihead_self_attention(x,d_model,head_num):
    """
    Run self_attention() 'head_num' different times and concatenate to axis=2.
    Initialize wo matrix. 
    Later this function will be embedded to self_attention as 4th dimension
    for a better runtime.
    x: input
    d_model: hyperparameter
    head_num: Number of heads.
    """
    z_all = tf.zeros([x.shape[0],x.shape[1],0])
    for i in range(head_num):
        
        q,k,v = qkv_matrices(x, d_model)
        z = self_attention(q,k,v)
        z_all = tf.concat([z_all, z], axis=2)
    
    wo = tf.random.normal([z_all.shape[0],z_all.shape[2],x.shape[2]], mean=0.0, stddev=1.0)
    z = tf.matmul(z_all,wo)
    return z


def encoder(x, d_model, head_num=1, units=64):
        
    # x = x.astype('float32')
    z = multihead_self_attention(x,d_model,head_num)
    sum = tf.math.add(x, z)
    norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)(sum)
    model = tf.keras.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units,activation='relu'),
    # tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(tf.multiply(z.shape[-1],z.shape[-2]) ,activation='relu'),
    # tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Reshape(z.shape[-2:])
    ])        
    dense = model(norm)

    output = tf.keras.layers.LayerNormalization(epsilon=1e-6)(dense+norm)

    return output

def stack_encoders(num_encoders, x, d_model, head_num, units):
    r = x
    for e in range(num_encoders):
        r = encoder(r, d_model, head_num, units)
    return r


# Load dataset:
filename = 'dataset_tensor.npy'
file_path = os.path.join(PROCESSED_DATASET_DIR, filename)
dataset = np.load(file_path, allow_pickle=True)

# Get x_train, y_train, x_test, y_test:
input_length=24
train, test = split_train_test(dataset)
x_train, y_train = get_xy(train, input_length=input_length)
x_test, y_test = get_xy(test, input_length=input_length)

# Run transformer:
d_model = 2
x_train = x_train.astype('float32')
x_train = x_train[:64,:,:]
# test_encoder = encoder(x_train, d_model, head_num=1, units=64)
stacked_encoders = stack_encoders(6, x_train, d_model, head_num=1, units=64)


# Visualization Test:
# bb = positional_encoding(x_train.shape[0],x_train.shape, broadcast=True)
# bb = bb.numpy()
# bb = bb.reshape((bb.shape[3], bb.shape[1], bb.shape[2]))
# visualize_pos_encoding(bb[0])

# Encoder Test:
# aa = np.arange(144).reshape((12,4,3))
# aa = aa.astype('float32')
# bb = encoder(aa, d_model, head_num=1, units=64)