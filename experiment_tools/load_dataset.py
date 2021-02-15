import os.path

import numpy as np
import sklearn.model_selection
import scipy.io

import common.paths
import dataset_tools.split


def get_tensor_dataset(data_path, file_name,
                       input_length, prediction_time, y_feature, y_city,
                       start_city=0, end_city=None,
                       remove_last_from_test=0, valid_split=None, split_random=None):
    filename = os.path.join(data_path, file_name)
    dataset = np.load(filename, allow_pickle=True)

    print(dataset.shape)
    city_feature_shape = (dataset.shape[1], dataset.shape[2])

    train, test = dataset_tools.split.split_train_test(dataset)
    x_train, y_train = dataset_tools.split.get_xy(train, input_length=input_length, pred_time=prediction_time)
    x_test, y_test = dataset_tools.split.get_xy(test, input_length=input_length, pred_time=prediction_time)

    if end_city is None:
        end_city = x_train.shape[-2]

    print(x_train.shape, y_train.shape)
    if valid_split is not None:
        x_train, x_valid, y_train, y_valid = sklearn.model_selection.train_test_split(x_train, y_train,
                                                                                      test_size=valid_split,
                                                                                      random_state=split_random)
        x_valid, y_valid = prepare_usa_xy(x_valid, y_valid, city_feature_shape,
                                          start_city, end_city, y_city, y_feature)

    x_train, y_train = prepare_usa_xy(x_train, y_train, city_feature_shape, start_city, end_city, y_city, y_feature)
    x_test, y_test = prepare_usa_xy(x_test, y_test, city_feature_shape, start_city, end_city, y_city, y_feature)

    x_test = x_test[:-remove_last_from_test, ...]
    y_test = y_test[:-remove_last_from_test, ...]

    print(f'FULL_x_train.shape: {x_train.shape}')

    params = [
        ('input_length', input_length),
        ('prediction_time', prediction_time),
        ('y_feature', y_feature),
        ('y_city', y_city),
        ('start_city', start_city),
        ('end_city', end_city),
        ('train_size', x_train.shape[0]),
        ('test_size', x_test.shape[0]),
    ]

    if valid_split is None:
        return (x_train, y_train, x_test, y_test), params
    else:
        params.append(('valid_size', x_valid.shape[0]))
        return (x_train, y_train, x_valid, y_valid, x_test, y_test), params


def get_usa_dataset(data_path, input_length, prediction_time, y_feature, y_city, start_city=0, end_city=None,
                    remove_last_from_test=0, valid_split=None, split_random=None):
    return get_tensor_dataset(data_path, 'dataset_tensor.npy',
                              input_length, prediction_time,
                              y_feature, y_city, start_city, end_city,
                              remove_last_from_test, valid_split, split_random)


def get_eu_dataset(data_path, input_length, prediction_time, y_feature, y_city, start_city=0, end_city=None,
                   remove_last_from_test=0, valid_split=None, split_random=None):
    return get_tensor_dataset(data_path, 'eu_dataset_tensor.npy',
                              input_length, prediction_time,
                              y_feature, y_city, start_city, end_city,
                              remove_last_from_test, valid_split, split_random)


def prepare_usa_xy(x, y, city_feature_shape, start_city, end_city, y_city, y_feature):
    x = np.reshape(x, (x.shape[0], x.shape[1], city_feature_shape[0], city_feature_shape[1]))
    y = np.reshape(y, (y.shape[0], city_feature_shape[0], city_feature_shape[1]))
    x = x[:, :, start_city:end_city, :]
    y = y[:, start_city:end_city, :]
    y = np.expand_dims(y[..., y_city, y_feature], axis=-1)
    return x, y


def get_denmark_dataset(data_path, step, feature, y_city=None, valid_split=None, split_random=None):
    filename = os.path.join(data_path, 'Denmark')
    filename = os.path.join(filename, f'{feature}')
    filename = os.path.join(filename, f'step{step}.mat')
    mat = scipy.io.loadmat(filename)
    Xtr = mat['Xtr'].swapaxes(1, 2)
    Ytr = mat['Ytr']
    Xtest = mat['Xtest'].swapaxes(1, 2)
    Ytest = mat['Ytest']
    if y_city is not None:
        Ytr = Ytr[:, y_city:y_city + 1]
        Ytest = Ytest[:, y_city:y_city + 1]

    if valid_split is None:
        return Xtr, Ytr, Xtest, Ytest
    else:
        Xtr, Xvalid, Ytr, Yvalid = sklearn.model_selection.train_test_split(Xtr, Ytr, test_size=valid_split,
                                                                            random_state=split_random)
        return Xtr, Ytr, Xvalid, Yvalid, Xtest, Ytest


def to_flatten_dataset(Xtr, Xtest, Xvalid=None):
    t = Xtr.shape[1]
    f = Xtr.shape[2] * Xtr.shape[3]

    Xtr = Xtr.reshape((Xtr.shape[0], t, f))
    Xtest = Xtest.reshape((Xtest.shape[0], t, f))
    if Xvalid is None:
        return Xtr, Xtest
    else:
        print(Xvalid.shape)
        Xvalid = Xvalid.reshape((Xvalid.shape[0], t, f))
        return Xtr, Xtest, Xvalid


def reshape_to_batches(Xs, Ys, batch_size):
    num_batches = Xs.shape[0] // batch_size
    batched_len = num_batches * batch_size
    print('Ys.shape', Ys.shape, 'batch_size', batch_size)
    print('batched_len', batched_len)
    Xs = Xs[:batched_len, ...].reshape((num_batches, batch_size) + Xs.shape[1:])
    Ys = Ys[:batched_len, ...].reshape((num_batches, batch_size) + Ys.shape[1:])

    return Xs, Ys


if __name__ == '__main__':
    (Xtr, Ytr, Xvalid, Yvalid, Xtest, Ytest), params = get_usa_dataset(data_path=common.paths.PROCESSED_DATASET_DIR,
                                                                       input_length=16, prediction_time=4, y_feature=4,
                                                                       y_city=0,
                                                                       start_city=0, end_city=30,
                                                                       remove_last_from_test=800,
                                                                       valid_split=1024, split_random=None)

    print(Xtr.shape, Ytr.shape, Xtest.shape, Ytest.shape, Xvalid.shape, Yvalid.shape)
    Xtr, Xtest, Xvalid = to_flatten_dataset(Xtr, Xtest, Xvalid)
    Xtr, Ytr = reshape_to_batches(Xtr, Ytr, 128)
    print(Xtr.shape, Ytr.shape, Xtest.shape, Ytest.shape, Xvalid.shape, Yvalid.shape)

    import matplotlib.pyplot as plt

    plt.figure(figsize=(40, 8))
    plt.plot(range(Ytest.size), Ytest.flatten())
    plt.show()
