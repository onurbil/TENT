import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
import tensorflow.keras.backend
import keras.layers

"""
Self attention implementation:
Not sure about dk axis=? and softmax axis=?
"""

def qkv_matrices(x):
    """
    Calculate query, key and value vectors:
    """
    wq = tf.keras.backend.random_normal([2,2,3], mean=0.0, stddev=1.0)
    q = keras.layers.Dot(axes=(2,1))([x, wq])

    wk = tf.keras.backend.random_normal([2,2,3], mean=0.0, stddev=1.0)
    k = keras.layers.Dot(axes=(2,1))([x, wk])

    wv = tf.keras.backend.random_normal([2,2,3], mean=0.0, stddev=1.0)
    v = keras.layers.Dot(axes=(2,1))([x, wv])
    
    return q,k,v


def self_attention(q,k,v,mask=None):
    """
    Calculates self attention and attention weights:
    """
    kt = tf.transpose(k, perm=[0, 2, 1])
    z = keras.layers.Dot(axes=(2,1))([q, kt])
    # !!! dk check how to calculate in tensors!!! over axis [-1] correct???
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    z = z / tf.math.sqrt(dk)
    
    if mask is not None:
        z += (mask * -1e9) 
    
    # !!! softmax check how to calculate in tensors!!! axis=-1 correct???
    sm = tf.keras.activations.softmax(z, axis=-1)
    z = keras.layers.Dot(axes=(2,1))([sm, v])
    
    return z


x = tf.keras.backend.constant(np.arange(12).reshape(2,3,2))
q,k,v = qkv_matrices(x)
output = self_attention(q,k,v)
print(output)
