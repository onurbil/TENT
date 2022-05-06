import os.path

import numpy as np
import sklearn.model_selection

import common.paths
import dataset_tools.split


def get_tensor_dataset(data_path, file_name, test_size,
                       input_length, prediction_time, y_feature, y_city,
                       start_city=0, end_city=None,
                       remove_last_from_test=None,
                       valid_split=None, split_random=None):
    filename = os.path.join(data_path, file_name)
    dataset = np.load(filename, allow_pickle=True)

    print('dataset.shape', dataset.shape)
    city_feature_shape = (dataset.shape[1], dataset.shape[2])

    train, test = dataset_tools.split.split_train_test_based_on_test(dataset, test_size)
    x_train, y_train = dataset_tools.split.get_xy(train, input_length=input_length, pred_time=prediction_time)
    x_test, y_test = dataset_tools.split.get_xy(test, input_length=input_length, pred_time=prediction_time)

    if end_city is None:
        end_city = city_feature_shape[0]

    if valid_split is not None:
        x_train, x_valid, y_train, y_valid = sklearn.model_selection.train_test_split(x_train, y_train,
                                                                                      test_size=valid_split,
                                                                                      random_state=split_random)
        x_valid, y_valid = prepare_xy(x_valid, y_valid, city_feature_shape,
                                      start_city, end_city, y_city, y_feature)

    x_train, y_train = prepare_xy(x_train, y_train, city_feature_shape, start_city, end_city, y_city, y_feature)
    x_test, y_test = prepare_xy(x_test, y_test, city_feature_shape, start_city, end_city, y_city, y_feature)

    if remove_last_from_test is not None and remove_last_from_test > 0:
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


def get_dataset_normalization(data_path, file_name, y_feature):
    filename = os.path.join(data_path, file_name)
    scale = np.load(filename, allow_pickle=True)
    return scale[y_feature][1].astype(np.float), scale[y_feature][2].astype(np.float)


def get_usa_dataset(data_path, input_length, prediction_time,
                    y_feature, y_city, start_city=0, end_city=None,
                    remove_last_from_test=None, valid_split=None, split_random=None):
    test_size = 69 * 128
    return get_tensor_dataset(data_path, 'usa-canada_dataset_tensor.npy', test_size,
                              input_length, prediction_time,
                              y_feature, y_city, start_city, end_city,
                              remove_last_from_test, valid_split, split_random)


def get_usa_normalization(data_path, y_feature):
    return get_dataset_normalization(data_path, file_name='usa-canada_scale.npy', y_feature=y_feature)


def get_eu_dataset(data_path, test_size, input_length, prediction_time,
                   y_feature, y_city, start_city=0, end_city=None,
                   remove_last_from_test=None, valid_split=None, split_random=None):
    return get_tensor_dataset(data_path, 'europe_dataset_tensor.npy', test_size,
                              input_length, prediction_time,
                              y_feature, y_city, start_city, end_city,
                              remove_last_from_test, valid_split, split_random)


def get_eu_averaged_dataset(data_path, test_size, input_length, prediction_time,
                            y_feature, y_city, start_city=0, end_city=None,
                            remove_last_from_test=None, valid_split=None, split_random=None):
    return get_tensor_dataset(data_path, 'europe_averaged_dataset_tensor.npy', test_size,
                              input_length, prediction_time,
                              y_feature, y_city, start_city, end_city,
                              remove_last_from_test, valid_split, split_random)


def get_eu_upsampled_dataset(data_path, test_size, input_length, prediction_time,
                             y_feature, y_city, start_city=0, end_city=None,
                             remove_last_from_test=None, valid_split=None, split_random=None):
    return get_tensor_dataset(data_path, 'europe_upsampled_dataset_tensor.npy', test_size,
                              input_length, prediction_time,
                              y_feature, y_city, start_city, end_city,
                              remove_last_from_test, valid_split, split_random)


def get_eu_normalization(data_path, y_feature):
    return get_dataset_normalization(data_path, file_name='europe_scale.npy', y_feature=y_feature)


def prepare_xy(x, y, city_feature_shape, start_city, end_city, y_city, y_feature):
    x = np.reshape(x, (x.shape[0], x.shape[1], city_feature_shape[0], city_feature_shape[1]))
    y = np.reshape(y, (y.shape[0], city_feature_shape[0], city_feature_shape[1]))
    x = x[:, :, start_city:end_city, :]
    y = y[:, start_city:end_city, :]
    y = np.expand_dims(y[..., y_city, y_feature], axis=-1)
    return x, y


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
    input_length = 8
    prediction_time = 2
    y_feature = 4
    y_city = 1
    valid_size = 512
    test_size = 1095  # 3 years of measurements

    dataset, dataset_params = get_eu_dataset(common.paths.PROCESSED_DATASET_DIR, test_size,
                                             input_length, prediction_time,
                                             y_feature, y_city,
                                             valid_split=valid_size, split_random=1337)

    Xtr, Ytr, Xvalid, Yvalid, Xtest, Ytest = dataset
    print('Xtr.shape', Xtr.shape)
    print('Ytr.shape', Ytr.shape)
    print('Xvalid.shape', Xvalid.shape)
    print('Yvalid.shape', Yvalid.shape)
    print('Xtest.shape', Xtest.shape)
    print('Ytest.shape', Ytest.shape)
