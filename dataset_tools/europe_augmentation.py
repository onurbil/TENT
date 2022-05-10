import os.path

import numpy as np
import sklearn.model_selection

import dataset_tools.split


def _piecewise_mean(data):
    reshaped_data = np.reshape(data, (data.shape[0] // 2, 2) + data.shape[1:])
    mean_data = np.mean(reshaped_data, axis=1)
    return mean_data


def reduce_average_dataset(data_path, file_name, output_file_name):
    filename = os.path.join(data_path, file_name)
    dataset = np.load(filename, allow_pickle=True)

    mean_dataset = _piecewise_mean(dataset)

    new_dataset_filepath = os.path.join(data_path, output_file_name)
    np.save(new_dataset_filepath, mean_dataset)
    return mean_dataset


def linear_upsample_dataset(data_path, file_name, output_file_name):
    filename = os.path.join(data_path, file_name)
    dataset = np.load(filename, allow_pickle=True)
    first_mean_dataset = _piecewise_mean(dataset)
    second_mean_dataset = _piecewise_mean(dataset[1:-1])

    upsampled_dataset = np.empty((dataset.shape[0] * 2 - 1,) + dataset.shape[1:], dtype=dataset.dtype)
    upsampled_dataset[0::2] = dataset
    upsampled_dataset[1::4] = first_mean_dataset
    upsampled_dataset[3::4] = second_mean_dataset

    new_dataset_filepath = os.path.join(data_path, output_file_name)
    np.save(new_dataset_filepath, upsampled_dataset)
    return upsampled_dataset


if __name__ == '__main__':
    # dataset = reduce_average_dataset('data', 'europe_dataset_tensor.npy', 'europe_averaged_dataset_tensor.npy')
    dataset = linear_upsample_dataset('data', 'europe_dataset_tensor.npy', 'europe_upsampled_dataset_tensor.npy')
