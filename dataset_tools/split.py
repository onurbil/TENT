import numpy as np

"""
First split the dataset in train and test array with 'split_train_test'. 
Then split train and test into x_train, y_train, x_test, y_test with 'get_xy'.
"""


def split_train_test(dataset, tr_batch_count=284, te_batch_count=69,
                     batch_size=128):
    """
    Returns train and test arrays. Flattens last dimesion.
    Test is last te_batch_count*batch_size rows of the dataset.
    Train is tr_batch_count*batch_size rows before test.
    The first rows of the dataset is not used.
    Inputs:
    dataset: dataset with shape (x,y,z)
    tr_batch_count: Batch count for train data.
    te_batch_count: Batch count for test data.
    batch_size: Size of each batch.     
    """
    dataset = dataset.reshape(dataset.shape[0], -1)
    train_range = tr_batch_count * batch_size
    test_range = te_batch_count * batch_size
    train = dataset[-(train_range + test_range):-test_range]
    test = dataset[-test_range:]

    return train, test


def get_xy(array, input_length=24, lag=1):
    """
    Make make_x and y arrays from given array:
    x:
    x_1, x_2, x_3, ..., x_n
    x_2, x_3, x_4, ..., x_n+1
    x_3, x_4, x_5, ..., x_n+2
    with n=input_length.
    y:
    y_(n+lag)
    y_(n+1+lag)
    y_(n+2+lag)
    """
    shape0 = array.shape[0]
    shape1 = array.shape[1]
    recurrent = np.zeros((shape0, input_length + lag, shape1))

    for i in range(input_length + lag):
        rec = array[i:]
        zeros_arr = np.zeros((shape0, shape1))
        zeros_arr[:shape0 - i, :] = rec
        recurrent[:, i, :] = zeros_arr

    recurrent = recurrent[:-(input_length + lag - 1), :, :]
    x = recurrent[:, :input_length]
    y = recurrent[:, -1, :]

    return x, y
