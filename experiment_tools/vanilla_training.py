import tensorflow.keras as kr
import tensorflow as tf

import experiment_tools.load_dataset as experiment_dataset
import vanilla_transformer.transformer as vt
from keras.callbacks import History

def initialize_tpu():
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Running on TPU ', tpu.cluster_spec().as_dict()['worker'])
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.TPUStrategy(tpu)  # tf.distribute.experimental.TPUStrategy(tpu)
    print("REPLICAS: ", strategy.num_replicas_in_sync)
    return strategy

def train_model(dataset, epoch=300, patience=20,
                num_layers=3, head_num=32, d_model=512, dense_units=512,
                batch_size=128, dropout_rate=0.01, loss=kr.losses.mean_squared_error, use_tpu=False):

    if use_tpu:
        strategy = initialize_tpu()
    
    Xtr, Ytr, Xvalid, Yvalid, Xtest, Ytest = dataset
    Xtr_flat, Xtest_flat, Xvalid_flat = experiment_dataset.to_flatten_dataset(Xtr, Xtest, Xvalid)

    input_length = Xtr_flat.shape[1]
    input_size = Xtr_flat.shape[2]
    output_size = Ytr.shape[-1]

    Xtr_flat, Ytr = experiment_dataset.reshape_to_batches(Xtr_flat, Ytr, batch_size)
    Xvalid_flat, Yvalid = experiment_dataset.reshape_to_batches(Xvalid_flat, Yvalid, batch_size)

    print(f'Xtr_flat: {Xtr_flat.shape}')
    print(f'Ytr: {Ytr.shape}')
    print(f'Xvalid_flat: {Xvalid_flat.shape}')
    print(f'Yvalid: {Yvalid.shape}')
    print(f'Xtest: {Xtest_flat.shape}')
    print(f'Ytest: {Ytest.shape}')

    learning_rate = vt.CustomSchedule(d_model)
    optimizer = kr.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    if use_tpu:
        with strategy.scope():
            model = vt.Transformer(input_size, num_layers, d_model, head_num, dense_units, input_length, output_size,
                           rate=dropout_rate)
            model.compile()
    else:
        model = vt.Transformer(input_size, num_layers, d_model, head_num, dense_units, input_length, output_size,
                               rate=dropout_rate)
        model.compile()

    early_stopping = kr.callbacks.EarlyStopping(monitor='val_loss',
                                                patience=patience,
                                                restore_best_weights=True,
                                                verbose=1)
    history = History()

    model.fit(Xtr_flat, Ytr,
              validation_data=(Xvalid_flat, Yvalid),
              epochs=epoch,
              optimizer=optimizer,
              loss=loss,
              metrics={'mse': kr.metrics.mse, 'mae': kr.metrics.mae},
              callbacks=[early_stopping,
                         history])

    model.summary()

    params = [
        ('epoch', epoch),
        ('patience', patience),
        ('stopped_epoch', early_stopping.stopped_epoch),
        ('num_layers', num_layers),
        ('head_num', head_num),
        ('d_model', d_model),
        ('dense_units', dense_units),
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
    model, model_params = train_model(dataset, epoch=1)
    params = dataset_params + model_params
