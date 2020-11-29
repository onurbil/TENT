import time
import os

import numpy as np
import tensorflow as tf
import tensorflow.keras as kr

import common.paths
from vanilla_transformer.transformer import Transformer, CustomSchedule

"""
    Size of dataset := N
    Number of cities := C
    Number of timesteps := T
    Number of features := F
"""

# N x C(5) x T(4) x F(4)
wind_train_x = np.load(common.paths.PROCESSED_DATASET_DIR + "/siamak/train_step1_x.npy", allow_pickle=True)
wind_train_y = np.load(common.paths.PROCESSED_DATASET_DIR + "/siamak/train_step1_y.npy", allow_pickle=True)
wind_test_x = np.load(common.paths.PROCESSED_DATASET_DIR + "/siamak/test_step1_x.npy", allow_pickle=True)
wind_test_y = np.load(common.paths.PROCESSED_DATASET_DIR + "/siamak/test_step1_y.npy", allow_pickle=True)

# N x T(4) x C(5) x F(4)
wind_train_x = np.transpose(wind_train_x, axes=[0, 2, 1, 3])
wind_test_x = np.transpose(wind_test_x, axes=[0, 2, 1, 3])

# N x T(4) x CF(20)
wind_train_x = np.reshape(wind_train_x, (wind_train_x.shape[0], wind_train_x.shape[1], wind_train_x.shape[2] * wind_train_x.shape[3]))
wind_test_x = np.reshape(wind_test_x, (wind_test_x.shape[0], wind_test_x.shape[1], wind_test_x.shape[2] * wind_test_x.shape[3]))

batch_size = 32
input_length = 4
lag = 1

num_cities = 5
num_features = 4

batches_x = []
batches_y = []

for b in range(0, len(wind_train_x) - batch_size, batch_size):
    batch_x = np.stack(wind_train_x[b:b + batch_size], axis=0)
    batch_y = np.stack(wind_train_y[b:b + batch_size], axis=0)

    batches_x.append(batch_x)
    batches_y.append(batch_y)

# N(N/batch_size) x batch_size x T(4) x CF(20)
train_x = np.stack(batches_x, axis=0).astype(np.float32)
train_y = np.stack(batches_y, axis=0).astype(np.float32)

# Not relevant for now as every input has been used
feature_index = None
city_index = None

if feature_index and city_index:
    train_y = train_y[:, :, num_cities * feature_index + city_index]
    train_y = np.reshape(train_y, (train_y.shape[0], train_y.shape[1], 1))
elif feature_index:
    feature_indexes = [feature_index * num_cities + n for n in range(num_cities)]
    train_y = np.take(train_y, feature_indexes, axis=2)

elif city_index:
    city_indexes = [city_index + n * num_cities for n in range(num_features)]
    train_y = np.take(train_y, city_indexes, axis=2)


print(train_x.shape)
print(train_y.shape)

input_size = train_x.shape[-1]
output_size = train_y.shape[-1]

EPOCHS = 50
num_layers = 4
d_model = 64
dff = 64
num_heads = 8
dropout_rate = 0.1

transformer = Transformer(input_size, num_layers, d_model, num_heads, dff, input_length, output_size,
                             rate=dropout_rate)

learning_rate = CustomSchedule(d_model)
optimizer = kr.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
# train_loss = kr.metrics.Mean(name='train_loss')

checkpoint_path = "./checkpoints/train"
ckpt = tf.train.Checkpoint(transformer=transformer,
                           optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# if a checkpoint exists, restore the latest checkpoint.
# if ckpt_manager.latest_checkpoint:
#   ckpt.restore(ckpt_manager.latest_checkpoint)
#   print ('Latest checkpoint restored!!')

#transformer.load_weights('./checkpoints/windspeed')

transformer.fit(train_x,
                train_y,
                epochs=EPOCHS,
                optimizer=optimizer,
                loss=kr.losses.MeanSquaredError(),
                metrics={'mse': kr.metrics.mse, 'mae': kr.metrics.mae},
                callbacks=[ckpt_manager])

pred = transformer(train_x[0], False)[0]
print(np.round(pred.numpy(), 3))
print(train_y[0])
