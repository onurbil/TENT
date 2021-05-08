import tensorflow as tf
import tensorflow.keras as kr

import common.paths
import model.tensorized_transformer as tt
from experiment_tools import load_dataset


def initialize_tpu():
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Running on TPU ', tpu.cluster_spec().as_dict()['worker'])
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.TPUStrategy(tpu)  # tf.distribute.experimental.TPUStrategy(tpu)
    print("REPLICAS: ", strategy.num_replicas_in_sync)
    return strategy


def create_model(input_shape, output_shape,
                 input_length, num_layers,
                 d_model, head_num, dense_units,
                 initializer, softmax_type, batch_size, save_aw):
    layers = [kr.Input(shape=input_shape),
              tt.PositionalEncoding()]

    for _ in range(num_layers):
        layers.append(
            tt.EncoderLayer(input_length, d_model, head_num, dense_units, initializer, softmax_type, batch_size, save_attention=save_aw))

    layers.extend([
        kr.layers.Flatten(),
        kr.layers.Dense(tf.reduce_prod(output_shape), activation='linear'),
        kr.layers.Reshape(output_shape),
    ])

    model = kr.Sequential(layers)
    return model


def train_model(dataset, softmax_type=3, epoch=300, patience=20,
                num_layers=3, head_num=32, d_model=256, dense_units=128,
                batch_size=16, loss=kr.losses.mse, use_tpu=True, save_aw = False):

    if use_tpu:
        strategy = initialize_tpu()

    Xtr, Ytr, Xvalid, Yvalid, Xtest, Ytest = dataset

    warmup_steps = 50
    factor1 = -0.6
    factor2 = -1.5
    initializer = 'RandomNormal'

    input_length = Xtr.shape[1]
    output_size = Ytr.shape[-1]

    print(f'Xtr: {Xtr.shape}')
    print(f'Ytr: {Ytr.shape}')
    print(f'Xvalid: {Xvalid.shape}')
    print(f'Yvalid: {Yvalid.shape}')
    print(f'Xtest: {Xtest.shape}')
    print(f'Ytest: {Ytest.shape}')

    num_examples = (Xtr.shape[0] // (batch_size * 8)) * (batch_size * 8)
    num_valid_examples = (Xvalid.shape[0] // (batch_size * 8)) * (batch_size * 8)

    Xtr = Xtr[:num_examples, ...]
    Ytr = Ytr[:num_examples, ...]
    Xvalid = Xvalid[:num_valid_examples, ...]
    Yvalid = Yvalid[:num_valid_examples, ...]

    input_shape = (input_length, Xtr.shape[-2], Xtr.shape[-1])
    output_shape = (1, 1)

    learning_rate = tt.CustomSchedule(d_model, warmup_steps=warmup_steps, factor1=factor1, factor2=factor2)
    optimizer = tf.keras.optimizers.Adam(learning_rate,
                                         beta_1=0.9,
                                         beta_2=0.98,
                                         epsilon=1e-9
                                         )
    lr_metric = tt.get_lr_metric(optimizer)

    if use_tpu:
        with strategy.scope():
            model = create_model(input_shape, output_shape, input_length,
                                 num_layers, d_model, head_num, dense_units,
                                 initializer, softmax_type, batch_size, save_aw)
            model.compile(optimizer=optimizer, loss=loss, metrics=['mse', 'mae', lr_metric])
    else:
        model = create_model(input_shape, output_shape, input_length,
                             num_layers, d_model, head_num, dense_units,
                             initializer, softmax_type, batch_size, save_aw)
        model.compile(optimizer=optimizer, loss=loss, metrics=['mse', 'mae', lr_metric])

    # Callbacks
    print_attention_weights = kr.callbacks.LambdaCallback(
        on_train_end=lambda batch: print(model.layers[1].attention_weights))
    early_stopping = kr.callbacks.EarlyStopping(monitor='val_loss',
                                                patience=patience,
                                                restore_best_weights=True,
                                                verbose=1)
    model.summary()
    history = model.fit(
        Xtr, Ytr,
        epochs=epoch,
        batch_size=batch_size * 8,
        validation_data=(Xvalid, Yvalid),
        callbacks=[early_stopping]
    )

    params = [
        ('softmax_type', softmax_type),
        ('epoch', epoch),
        ('patience', patience),
        ('stopped_epoch', early_stopping.stopped_epoch),
        ('num_layers', num_layers),
        ('head_num', head_num),
        ('d_model', d_model),
        ('dense_units', dense_units),
        ('warmup_steps', warmup_steps),
        ('factor1', factor1),
        ('factor2', factor2),
        ('initializer', initializer),
        ('batch_size', batch_size),
    ]
    return model, params, history


if __name__ == '__main__':
    dataset, dataset_params = load_dataset.get_usa_dataset(data_path=common.paths.PROCESSED_DATASET_DIR,
                                                           input_length=16, prediction_time=4,
                                                           y_feature=4, y_city=0,
                                                           start_city=0, end_city=30,
                                                           remove_last_from_test=800,
                                                           valid_split=1024, split_random=None)
    Xtr, Ytr, Xvalid, Yvalid, Xtest, Ytest = dataset
    print(Xtr.shape, Ytr.shape, Xtest.shape, Ytest.shape, Xvalid.shape, Yvalid.shape)
    model, model_params, history = train_model(dataset, use_tpu=False)
    params = dataset_params + model_params
