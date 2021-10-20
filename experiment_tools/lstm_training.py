import tensorflow.keras as kr
import tensorflow as tf
import numpy as np

import experiment_tools.load_dataset as experiment_dataset
import lstm.lstm as lstm
# from keras.callbacks import History

def initialize_tpu():
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Running on TPU ', tpu.cluster_spec().as_dict()['worker'])
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.TPUStrategy(tpu)  # tf.distribute.experimental.TPUStrategy(tpu)
    print("REPLICAS: ", strategy.num_replicas_in_sync)
    return strategy

def train_lstm(dataset, epoch=300, patience=20,
                num_layers=2, hidden_units=128, dropout_rate=0.1,
                learning_rate=1e-4,
                batch_size=128, loss=kr.losses.mean_squared_error, use_tpu=False):

    if use_tpu:
        strategy = initialize_tpu()

    Xtr, Ytr, Xvalid, Yvalid, Xtest, Ytest = dataset
    Xtr_flat, Xtest_flat, Xvalid_flat = experiment_dataset.to_flatten_dataset(Xtr, Xtest, Xvalid)

    input_length = Xtr_flat.shape[1]
    input_size = Xtr_flat.shape[2]
    output_size = Ytr.shape[-1]

    # Xtr_flat, Ytr = experiment_dataset.reshape_to_batches(Xtr_flat, Ytr, batch_size)
    # Xvalid_flat, Yvalid = experiment_dataset.reshape_to_batches(Xvalid_flat, Yvalid, batch_size)

    print(f'Xtr_flat: {Xtr_flat.shape}')
    print(f'Ytr: {Ytr.shape}')
    print(f'Xvalid_flat: {Xvalid_flat.shape}')
    print(f'Yvalid: {Yvalid.shape}')
    print(f'Xtest: {Xtest_flat.shape}')
    print(f'Ytest: {Ytest.shape}')

    optimizer = kr.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    if use_tpu:
        with strategy.scope():
            model = lstm.create_lstm(input_length, input_size, output_size, num_layers, hidden_units, dropout_rate)
            model.compile(optimizer,
                          loss=loss,
                          metrics=['mse', 'mae'],
                          )
    else:
        model = lstm.create_lstm(input_length, input_size, output_size, num_layers, hidden_units, dropout_rate)
        model.compile(optimizer,
                      loss=loss,
                      metrics=['mse', 'mae'],
                      )
    # history = History()

    early_stopping = kr.callbacks.EarlyStopping(monitor='val_loss',
                                                patience=patience,
                                                restore_best_weights=True,
                                                verbose=1)

    model.fit(Xtr_flat, Ytr,
              validation_data=(Xvalid_flat, Yvalid),
              epochs=epoch,
              batch_size=batch_size,
              callbacks=[early_stopping,]
              )

    model.summary()

    params = [
        ('epoch', epoch),
        ('patience', patience),
        ('stopped_epoch', early_stopping.stopped_epoch),
        ('num_layers', num_layers),
        ('hidden_units', hidden_units),
        ('batch_size', batch_size),
        ('dropout_rate', dropout_rate),
        ('loss', loss),
    ]

    return model, params


def transform_dataset_for_conv_lstm(Xtr, Xvalid, Xtest):
    print(Xtr.shape, Xvalid.shape, Xtest.shape)
    Xtr_expanded = np.expand_dims(Xtr, axis=2)
    Xvalid_expanded = np.expand_dims(Xvalid, axis=2)
    Xtest_expanded = np.expand_dims(Xtest, axis=2)
    return Xtr_expanded, Xvalid_expanded, Xtest_expanded


def train_conv_lstm(dataset, epoch=300, patience=20,
                    num_layers=2,
                    filters=8, kernel_size=3,
                    dropout_rate=0.1, padding='same',
                    learning_rate=1e-4,
                    batch_size=128, loss=kr.losses.mean_squared_error, use_tpu=False):

    if use_tpu:
        strategy = initialize_tpu()

    Xtr, Ytr, Xvalid, Yvalid, Xtest, Ytest = dataset
    Xtr, Xtest, Xvalid = transform_dataset_for_conv_lstm(Xtr, Xtest, Xvalid)

    input_length = Xtr.shape[1]
    input_size = Xtr.shape[2:]
    output_size = Ytr.shape[-1]

    # Xtr_flat, Ytr = experiment_dataset.reshape_to_batches(Xtr_flat, Ytr, batch_size)
    # Xvalid_flat, Yvalid = experiment_dataset.reshape_to_batches(Xvalid_flat, Yvalid, batch_size)

    print(f'Xtr: {Xtr.shape}')
    print(f'Ytr: {Ytr.shape}')
    print(f'Xvalid: {Xvalid.shape}')
    print(f'Yvalid: {Yvalid.shape}')
    print(f'Xtest: {Xtest.shape}')
    print(f'Ytest: {Ytest.shape}')

    optimizer = kr.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    if use_tpu:
        with strategy.scope():
            model = lstm.create_conv_lstm(input_length, input_size, output_size, num_layers,
                                          filters, kernel_size,
                                          dropout_rate, padding)
            model.compile(optimizer,
                          loss=loss,
                          metrics=['mse', 'mae'],
                          )
    else:
        model = lstm.create_conv_lstm(input_length, input_size, output_size, num_layers,
                                      filters, kernel_size,
                                      dropout_rate, padding)
        model.compile(optimizer,
                      loss=loss,
                      metrics=['mse', 'mae'],
                      )
    # history = History()

    early_stopping = kr.callbacks.EarlyStopping(monitor='val_loss',
                                                patience=patience,
                                                restore_best_weights=True,
                                                verbose=1)

    model.fit(Xtr, Ytr,
              validation_data=(Xvalid, Yvalid),
              epochs=epoch,
              batch_size=batch_size,
              callbacks=[early_stopping,]
              )

    model.summary()

    params = [
        ('epoch', epoch),
        ('patience', patience),
        ('stopped_epoch', early_stopping.stopped_epoch),
        ('num_layers', num_layers),
        ('filters', filters),
        ('kernel_size', kernel_size),
        ('padding', padding),
        ('batch_size', batch_size),
        ('dropout_rate', dropout_rate),
        ('loss', loss),
    ]

    return model, params

if __name__ == '__main__':
    import load_dataset
    import common.paths

    dataset, dataset_params = load_dataset.get_usa_dataset(data_path=common.paths.PROCESSED_DATASET_DIR,
                                                           input_length=16, prediction_time=4,
                                                           y_feature=4, y_city=0,
                                                           start_city=0, end_city=30,
                                                           remove_last_from_test=800,
                                                           valid_split=1024, split_random=None)
    Xtr, Ytr, Xvalid, Yvalid, Xtest, Ytest = dataset
    print(Xtr.shape, Ytr.shape, Xtest.shape, Ytest.shape, Xvalid.shape, Yvalid.shape)
    model, model_params = train_conv_lstm(dataset, epoch=1)
    params = dataset_params + model_params
