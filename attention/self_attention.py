import numpy as np
import os
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


def positional_encoding(position, d_model):
    """
    Copied from tutorial. The dimension doesnt change the implementation of
    the positional encoding. Only the correct dimension (time dimension) must
    sent as "position" input to positional_encoding.
    d_model is hyperparameter.
    """
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)
    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
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
    # Positional encoding missing! Input 3. dimension also 512 after
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





# Load dataset:
filename = 'dataset_tensor.npy'  
file_path = os.path.join(PROCESSED_DATASET_DIR, filename)
dataset = np.load(file_path, allow_pickle=True)
# Or use a random array for test:
x = tf.keras.backend.constant(np.arange(60).reshape(5,4,3))
z = multihead_self_attention(x, hp=10, loop=8)
out = encoder(x,z)
print(out)