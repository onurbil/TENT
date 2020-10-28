import numpy as np
import os

import tensorflow as tf
import tensorflow.keras as kr

###############
#  No batch   #
###############

b = 6  # batch size
t = 5  # lag
c = 4  # cities
m = 3  # measurements
d = 2  # internal dimension

x = tf.ones([t, c, m], dtype=tf.float32)
# print(f'x: {x.shape}')

wq = tf.ones([m, d], dtype=tf.float32)
wk = tf.ones([m, d], dtype=tf.float32)
wv = tf.ones([m, d], dtype=tf.float32)
# print(f'wq: {wq.shape}')
# print(f'wk: {wk.shape}')
# print(f'wv: {wv.shape}')

q = tf.matmul(x, wq)
k = tf.matmul(x, wk)
v = tf.matmul(x, wv)
# print(f'q: {q.shape}')
# print(f'k: {k.shape}')
# print(f'v: {v.shape}')

from debugging_tools import *
print(q)
kt = tf.transpose(k, perm=[0, 2, 1])
# kt = tf.expand_dims(kt, axis=0)
kt = tf.broadcast_to(kt, [kt.shape[0], kt.shape[0], kt.shape[1], kt.shape[2]])
print(kt)
pause()
print(f'kt: {kt.shape}')

qe = tf.broadcast_to(q, [q.shape[0], q.shape[0], q.shape[1], q.shape[2]])
print(f'qe: {qe.shape}')

s = tf.matmul(qe, kt)
print(f's: {s.shape}')

s_sum = tf.reduce_sum(s, axis=[-1, -2])
print(f's_sum: {s_sum.shape}')

softmax = kr.layers.Softmax(axis=-1)
s_softmax = softmax(s_sum)
print(f's_softmax: {s_softmax.shape}')

ve = tf.broadcast_to(v, [v.shape[0], v.shape[0], v.shape[1], v.shape[2]])
print(f've: {ve.shape}')

se = s_softmax
se = tf.expand_dims(se, -1)
se = tf.expand_dims(se, -1)
se = tf.broadcast_to(se, [v.shape[0], v.shape[0], v.shape[1], v.shape[2]])
print(f'se: {se.shape}')

z = tf.multiply(se, ve)
print(f'z: {z.shape}')

z_sum = tf.reduce_sum(z, axis=1)
print(f'z_sum: {z_sum.shape}')
print(z_sum)

# NN to produce output tensor t x c x m


###############
#  With batch #
###############

x = tf.ones([b, t, c, m], dtype=tf.float32)
print(f'x: {x.shape}')

wq = tf.ones([m, d], dtype=tf.float32)
wk = tf.ones([m, d], dtype=tf.float32)
wv = tf.ones([m, d], dtype=tf.float32)
print(f'wq: {wq.shape}')
print(f'wk: {wk.shape}')
print(f'wv: {wv.shape}')

q = tf.matmul(x, wq)
k = tf.matmul(x, wk)
v = tf.matmul(x, wv)
print(f'q: {q.shape}')
print(f'k: {k.shape}')
print(f'v: {v.shape}')

kt = tf.transpose(k, perm=[0, 1, 3, 2])
kt = tf.expand_dims(kt, axis=2)
kt = tf.broadcast_to(kt, [kt.shape[0], kt.shape[1], kt.shape[1], kt.shape[3], kt.shape[4]])
print(f'kt: {kt.shape}')

qe = q
qe = tf.expand_dims(qe, axis=2)
qe = tf.broadcast_to(qe, [qe.shape[0], qe.shape[1], qe.shape[1], qe.shape[3], qe.shape[4]])
print(f'qe: {qe.shape}')

s = tf.matmul(qe, kt)
print(f's: {s.shape}')

s_sum = tf.reduce_sum(s, axis=[-1, -2])
print(f's_sum: {s_sum.shape}')

softmax = kr.layers.Softmax(axis=-1)
s_softmax = softmax(s_sum)
print(f's_softmax: {s_softmax.shape}')

ve = v
ve = tf.expand_dims(ve, axis=2)
ve = tf.broadcast_to(ve, [ve.shape[0], ve.shape[1], ve.shape[1], ve.shape[3], ve.shape[4]])
print(f've: {ve.shape}')

se = s_softmax
se = tf.expand_dims(se, -1)
se = tf.expand_dims(se, -1)
se = tf.broadcast_to(se, [ve.shape[0], ve.shape[1], ve.shape[1], ve.shape[3], ve.shape[4]])
print(f'se: {se.shape}')

z = tf.multiply(se, ve)
print(f'z: {z.shape}')

z_sum = tf.reduce_sum(z, axis=1)
print(f'z_sum: {z_sum.shape}')
print(z_sum)

# NN to produce output tensor t x c x m
