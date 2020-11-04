import time
import os

import numpy as np
import tensorflow as tf
import tensorflow.keras as kr

import common.paths
import vanilla_transformer as vt

dataset = np.load(os.path.join(common.paths.PROCESSED_DATASET_DIR, 'dataset_tensor.npy'), allow_pickle=True)
dataset = np.transpose(dataset, axes=[0, 2, 1])
dataset = np.reshape(dataset, (dataset.shape[0], dataset.shape[1] * dataset.shape[2]))
dataset = dataset[:1000, ...]
print(dataset.shape)

batch_size = 32
input_length = 10
lag = 1

num_examples = dataset.shape[0] - input_length - lag
input_sequences = []
output_rows = []
for i in range(num_examples):
    input_sequences.append(dataset[i:i + input_length])
    output_rows.append(dataset[i + input_length + lag])

batches_x = []
batches_y = []
for b in range(0, len(input_sequences)-batch_size, batch_size):
    batch_x = np.stack(input_sequences[b:b+batch_size], axis=0)
    batch_y = np.stack(output_rows[b:b+batch_size], axis=0)

    batches_x.append(batch_x)
    batches_y.append(batch_y)

train_x = np.stack(batches_x, axis=0).astype(np.float32)
train_y = np.stack(batches_y, axis=0).astype(np.float32)
print(train_x.shape)
print(train_y.shape)

input_size = dataset.shape[-1]
output_size = dataset.shape[-1]

EPOCHS = 50
num_layers = 4
d_model = 64
dff = 64
num_heads = 8
dropout_rate = 0.1

transformer = vt.Transformer(input_size, num_layers, d_model, num_heads, dff, input_length, output_size,
                             rate=dropout_rate)

learning_rate = vt.CustomSchedule(d_model)
optimizer = kr.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
# train_loss = kr.metrics.Mean(name='train_loss')

transformer.fit(train_x, train_y, EPOCHS, optimizer, loss=kr.losses.MeanSquaredError())

pred = transformer(train_x[0], False)[0]
print(np.round(pred.numpy(), 3))
print(train_y[0])
