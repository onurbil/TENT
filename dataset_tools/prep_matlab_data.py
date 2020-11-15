import os

from scipy.io import loadmat
import numpy as np

import common.paths


def mat_to_npy(in_path, out_path, name):
    """
    Matlab file is a dict with 6 entries
     Xp, Yp = full data set
     Xtrain,Ytrain, Xtest, Ytest = xp/yp splitted into training and test set
     Size of dataset := N
     Number of cities := C
     Number of timesteps := T
     Number of features := F

     the x input entries have a shape N x C(5) x T(4) x F(4)
     the y output are have a shape N x 3, I didn t find where the '3'(what the output means) is mentioned in the paper

    For simplicity ill just save x/y train/test into 4 differnt .npy files

    """

    annotations = loadmat(in_path, struct_as_record=False)

    # xp = annotations['Xp']
    # yp = annotations['Yp']
    x_test = annotations['Xtest']
    y_test = annotations['Ytest']
    x_train = annotations['Xtr']
    y_train = annotations['Ytr']

    if not os.path.exists(common.paths.PROCESSED_DATASET_DIR + '/siamak/'):
        os.makedirs(common.paths.PROCESSED_DATASET_DIR + '/siamak/')

    with open(common.paths.PROCESSED_DATASET_DIR + '/siamak/train_' + name + '_x.npy', 'wb') as f:
        np.save(f, x_train)
        f.close()

    with open(common.paths.PROCESSED_DATASET_DIR + '/siamak/train_' + name + '_y.npy', 'wb') as f:
        np.save(f, y_train)
        f.close()

    with open(common.paths.PROCESSED_DATASET_DIR + '/siamak/test_' + name + '_x.npy', 'wb') as f:
        np.save(f, x_test)
        f.close()

    with open(common.paths.PROCESSED_DATASET_DIR + '/siamak/test_' + name + '_y.npy', 'wb') as f:
        np.save(f, y_test)
        f.close()


mat_to_npy("C:/Users/Yannick/Downloads/datasetmrp/dataset/step1.mat", "./", "step1")
mat_to_npy("C:/Users/Yannick/Downloads/datasetmrp/dataset/step2.mat", "./", "step2")



