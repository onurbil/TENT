import numpy as np
import os

from numpy.core._multiarray_umath import broadcast

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import tensorflow.keras.backend
import keras.layers
from common.paths import PROCESSED_DATASET_DIR
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
        debug(new_shape)
        angle_rads = np.reshape(angle_rads, new_shape)

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


def qkv_matrices(x,hp):
    """
    Calculate query, key and value vectors:
    hp: Hyperparameter
    """
    wq = tf.random.normal([x.shape[2],hp], mean=0.0, stddev=1.0)
    q = tf.matmul(x,wq)

    wk = tf.random.normal([x.shape[2],hp], mean=0.0, stddev=1.0)
    k = tf.matmul(x,wk)

    wv = tf.random.normal([x.shape[2],hp], mean=0.0, stddev=1.0)
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
    # !!! dk check how to calculate in tensors!!! over axis [-1] correct???
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


def multihead_self_attention(x,hp,loop):
    """
    Run self_attention() 'loop' different times and concatenate to axis=2.
    Initialize wo matrix. 
    Later this function will be embedded to self_attention as 4th dimension
    for a better runtime.
    x: input
    hp: hyperparameter
    loop: repeating number. 
    """
    z_all = tf.zeros([x.shape[0],x.shape[1],0])
    for i in range(loop):
        
        q,k,v = qkv_matrices(x, hp)
        z = self_attention(q,k,v)
        z_all = tf.concat([z_all, z], axis=2)
    
    wo = tf.random.normal([z_all.shape[0],z_all.shape[2],x.shape[2]], mean=0.0, stddev=1.0)
    z = tf.matmul(z_all,wo)
    return z


def encoder(x,z,units=64):

    d_model = z.shape[-1]
    # Input 3. dimension also 512 after
    # encoding?
    # 3. Dimension mismatch for residuals!
    sum = tf.math.add(x, z)
    norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)(sum)
    model = tf.keras.Sequential([
    tf.keras.layers.Dense(units,activation='relu'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(d_model ,activation='relu'),
    tf.keras.layers.Dropout(0.1),
    ])        
    dense = model(norm)
    output = tf.keras.layers.LayerNormalization(epsilon=1e-6)(dense+norm)

    return output


def split_train_test(dataset, tr_batch_count=284, te_batch_count=69,
                     batch_size=128):
    """
    Returns x_train, y_train, x_test, y_test. Flattens last dimesion.
    Test is last te_batch_count*batch_size rows of the dataset.
    Train is tr_batch_count*batch_size rows before test.
    The first rows of the dataset is not used.
    Inputs:
    dataset: dataset with shape (x,y,z)
    tr_batch_count: Batch count for train data.
    te_batch_count: Batch count for test data.
    batch_size: Size of each batch.     
    """
    dataset = dataset.reshape(dataset.shape[0],-1)
    train_range = tr_batch_count*batch_size
    test_range = te_batch_count*batch_size    
    train = dataset[-(train_range+test_range):-test_range]
    test = dataset[-test_range:]
    
    return train, test
    
    
def make_recurrent(array, input_length=24):
    """
    Make make_recurrent arrays from given array:
    x1, x2, x3, ..., xn
    x2, x3, x4, ..., xn
    x3, x4, x5, ..., xn
    with n=input_length.
    """
    shape0 = array.shape[0]
    shape1 = array.shape[1]
    recurrent = np.zeros((shape0, input_length, shape1))
    
    for i in range(input_length):
        rec = array[i:]
        zeros_arr = np.zeros((shape0,shape1))
        zeros_arr[:shape0-i,:] = rec
        recurrent[:,i,:] = zeros_arr
    # -1 because last array is needed for y:
    recurrent = recurrent[:-(input_length-1),:,:]
        
    return recurrent


# Load dataset:
filename = 'dataset_tensor.npy'
file_path = os.path.join(PROCESSED_DATASET_DIR, filename)
dataset = np.load(file_path, allow_pickle=True)

train, test = split_train_test(dataset)

input_length=24
recurrent_tr = make_recurrent(train, input_length=input_length)
x_train = recurrent_tr[:-1]
y_train = recurrent_tr[1:,input_length-1,:]

recurrent_te = make_recurrent(test, input_length=input_length)
x_test = recurrent_te[:-1]
y_test = recurrent_te[1:,input_length-1,:]



# Or use a random array for test:
# x = tf.keras.backend.constant(np.arange(60).reshape(5,4,3))
# z = multihead_self_attention(x, hp=10, loop=8)
# out = encoder(x,z)
# print(out)