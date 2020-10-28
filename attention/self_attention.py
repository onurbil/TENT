import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
import tensorflow.keras.backend
import keras.layers
from common.paths import PROCESSED_DATASET_DIR



"""
Self attention implementation:
"""

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
    ve = tf.broadcast_to(v, [v.shape[0], v.shape[0], v.shape[1], v.shape[2]])
    
    se = tf.expand_dims(se, -1)
    se = tf.expand_dims(se, -1)
    se = tf.broadcast_to(se, [v.shape[0], v.shape[0], v.shape[1], v.shape[2]])
    z = tf.multiply(se, ve)
    # !!! reduce_sum vs reduce_mean: Reduce mean --> smaller variance --> Distinction harder.
    # Here reduce_sum seems better.
    z_sum = tf.reduce_sum(z, axis=1)    
    
    return z_sum


# Load dataset:
filename = 'dataset_tensor.npy'  
file_path = os.path.join(PROCESSED_DATASET_DIR, filename)
dataset = np.load(file_path, allow_pickle=True)
# Or use a random array for test:
x = tf.keras.backend.constant(np.arange(60).reshape(5,4,3))
q,k,v = qkv_matrices(x, hp=10)

output = self_attention(q,k,v)
print(output)