import time
import os

import numpy as np
import tensorflow as tf
import tensorflow.keras as kr

import vanilla_transformer.transformer as vt
import visualization_tools.visualization as results


def prepare_dataset(input_length, prediction_time, batch_size,
                    dataset_path, test_size, valid_size, dataset_limit=None):
    dataset = np.load(dataset_path, allow_pickle=True)
    dataset = np.reshape(dataset, (dataset.shape[0], dataset.shape[1] * dataset.shape[2]))
    if dataset_limit is not None:
        dataset = dataset[:dataset_limit, ...]
    print(dataset.shape)

    num_examples = dataset.shape[0] - input_length - prediction_time
    input_sequences = []
    output_rows = []
    for i in range(num_examples):
        input_sequences.append(dataset[i:i + input_length])
        output_rows.append(dataset[i + input_length + prediction_time])

    batches_x = []
    batches_y = []
    for b in range(0, len(input_sequences) - batch_size, batch_size):
        batch_x = np.stack(input_sequences[b:b + batch_size], axis=0)
        batch_y = np.stack(output_rows[b:b + batch_size], axis=0)

        batches_x.append(batch_x)
        batches_y.append(batch_y)

    test_batches = test_size // batch_size
    valid_batches = valid_size // batch_size
    train_batches = len(batches_x) - valid_batches - test_batches
    train_x = np.stack(batches_x[:train_batches], axis=0).astype(np.float32)
    train_y = np.stack(batches_y[:train_batches], axis=0).astype(np.float32)
    valid_x = np.stack(batches_x[train_batches:train_batches + valid_batches], axis=0).astype(np.float32)
    valid_y = np.stack(batches_y[train_batches:train_batches + valid_batches], axis=0).astype(np.float32)
    test_x = np.stack(batches_x[-test_batches:], axis=0).astype(np.float32)
    test_y = np.stack(batches_y[-test_batches:], axis=0).astype(np.float32)
    print(f'train x: {train_x.shape}')
    print(f'train y: {train_y.shape}')
    print(f'valid x: {valid_x.shape}')
    print(f'valid y: {valid_y.shape}')
    print(f'test x: {test_x.shape}')
    print(f'test y: {test_y.shape}')
    return train_x, train_y, valid_x, valid_y, test_x, test_y


def run_full_measurements_experiment(input_length, prediction_time,
                                     num_layers, d_model, dff, num_heads, dropout_rate,
                                     epochs, batch_size,
                                     dataset_path, test_size, valid_size, dataset_limit=None,
                                     save_checkpoints=True):
    train_x, train_y, valid_x, valid_y, test_x, test_y = prepare_dataset(input_length, prediction_time, batch_size,
                                                                         dataset_path, test_size, valid_size,
                                                                         dataset_limit)

    train_x = train_x[:100, ...]
    train_y = train_y[:100, ...]
    input_size = train_x.shape[-1]
    output_size = train_x.shape[-1]

    transformer = vt.Transformer(input_size, num_layers, d_model, num_heads, dff, input_length, output_size,
                                 rate=dropout_rate)
    transformer.compile()

    learning_rate = vt.CustomSchedule(d_model)
    optimizer = kr.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    early_stopping = kr.callbacks.EarlyStopping(min_delta=.01, patience=10, restore_best_weights=True, verbose=1)

    transformer.fit(train_x, train_y,
                    validation_data=(valid_x, valid_y),
                    epochs=epochs,
                    optimizer=optimizer,
                    loss=kr.losses.MeanSquaredError(),
                    metrics={'mse': kr.metrics.mse, 'mae': kr.metrics.mae},
                    callbacks=[early_stopping])

    transformer.summary()

    pred = transformer(test_x[0], False)[0]
    mse = np.mean(kr.metrics.mse(test_y[0], pred))
    mae = np.mean(kr.metrics.mae(test_y[0], pred))
    print(f'mse: {mse}, mae: {mae}')

    return transformer, (mse, mae)


if __name__ == '__main__':
    transformer, (mse, mae) = run_full_measurements_experiment(input_length=10, prediction_time=1,
                                                               num_layers=4, d_model=64, dff=64, num_heads=8,
                                                               dropout_rate=.1,
                                                               epochs=20, batch_size=32,
                                                               dataset_path='../../processed_dataset/dataset_tensor.npy',
                                                               test_size=24 * 365, valid_size=24 * 365,
                                                               save_checkpoints=False)
