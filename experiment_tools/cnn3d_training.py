import numpy as np
import tensorflow as tf
import tensorflow.keras as kr

import model_3d_cnn.CNN3d as cnn3d

def initialize_tpu():
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Running on TPU ', tpu.cluster_spec().as_dict()['worker'])
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.TPUStrategy(tpu)  # tf.distribute.experimental.TPUStrategy(tpu)
    print("REPLICAS: ", strategy.num_replicas_in_sync)
    return strategy

def transform_dataset(Xtr, Xvalid, Xtest):
    transpose_axes = [0, 2, 1, 3]
    Xtr_t = np.transpose(Xtr, axes=transpose_axes)
    Xvalid_t = np.transpose(Xvalid, axes=transpose_axes)
    Xtest_t = np.transpose(Xtest, axes=transpose_axes)
    return Xtr_t, Xvalid_t, Xtest_t


def train_model(dataset, epoch=300, patience=20,
                filters=10, kernel_size=2,
                batch_size=128, learning_rate=0.0001, loss='mse', use_tpu=False):
    if use_tpu:
        strategy = initialize_tpu()

    Xtr, Ytr, Xvalid, Yvalid, Xtest, Ytest = dataset

    input_length = Xtr.shape[-3]
    stations = Xtr.shape[-2]
    features = Xtr.shape[-1]

    Xtr_t, Xvalid_t, Xtest_t = transform_dataset(Xtr, Xvalid, Xtest)

    print(f'Xtr_t: {Xtr_t.shape}')
    print(f'Ytr: {Ytr.shape}')
    print(f'Xvalid_t: {Xvalid_t.shape}')
    print(f'Yvalid: {Yvalid.shape}')
    print(f'Xtest_t: {Xtest_t.shape}')
    print(f'Ytest: {Ytest.shape}')

    optimizer = kr.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    if use_tpu:
        with strategy.scope():
            model = cnn3d.ThreeDimCNN_parallel_output(stations, input_length, features, filters, kernel_size)
            model.compile(optimizer=optimizer, loss=loss, metrics=['mse', 'mae'])
    else:
        model = cnn3d.ThreeDimCNN_parallel_output(stations, input_length, features, filters, kernel_size)
        model.compile(optimizer=optimizer, loss=loss, metrics=['mse', 'mae'])

    early_stopping = kr.callbacks.EarlyStopping(monitor='val_loss',
                                                patience=patience,
                                                restore_best_weights=True,
                                                verbose=1)

    model.summary()
    history = model.fit(Xtr_t, Ytr,
                        validation_data=(Xvalid_t, Yvalid),
                        epochs=epoch,
                        callbacks=[early_stopping])

    params = [
        ('epoch', epoch),
        ('patience', patience),
        ('stopped_epoch', early_stopping.stopped_epoch),
        ('filters', filters),
        ('kernel_size', kernel_size),
        ('batch_size', batch_size),
        ('learning_rate', learning_rate),
        ('loss', loss),
    ]

    return model, params, history
