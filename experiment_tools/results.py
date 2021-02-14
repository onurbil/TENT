import os
import datetime

import tensorflow.keras as kr
import numpy as np
import matplotlib.pyplot as plt

import dataset_tools.denormalization as denorm


def plot_predictions(Ys, pred, folder, name, ending):
    fileName = folder + name
    pred = pred.flatten()
    Ys = Ys.flatten()
    mae = kr.metrics.mae(Ys, pred)
    mse = kr.metrics.mse(Ys, pred)
    f = open(fileName + ".txt", "a")
    print(f'Figure mae{ending}: {np.mean(mae)}', file=f)
    print(f'Figure mse{ending}: {np.mean(mse)}', file=f)
    print('\n\n', file=f)
    f.close()
    print(f'Figure mae: {np.mean(mae)}')
    print(f'Figure mse: {np.mean(mse)}')

    plot_width = 20 if Ys.size < 1000 else 100
    plt.figure(figsize=(plot_width, 8))
    plt.plot(range(pred.size), pred, label='pred')
    plt.plot(range(len(Ys)), Ys, label='true')
    plt.legend()
    plt.savefig(fileName + ending)
    plt.show()


def save_results(parameters, model, folder, name):
    fileName = os.path.join(folder, name)
    print(f'saving to folder: {folder}')
    os.makedirs(os.path.dirname(folder), exist_ok=True)
    with open(fileName + ".txt", "a") as fh:
        model.summary(print_fn=lambda x: fh.write(x + '\n'))
    f = open(fileName + ".txt", "a")
    print("\n\n######################## Model description ################################", file=f)
    for name, results in parameters:
        print(str(name) + " = " + str(results), file=f)
    print("######################### Results ##########################################", file=f)
    f.close()


def save_results_with_datetime(model, base_name, path, parameters):
    date = datetime.datetime.now().strftime("%Y_%m_%d")
    time = datetime.datetime.now().strftime("%H_%M_%S")
    name = f'{base_name}_{time}'
    folder = os.path.join(path, 'Tests/')
    folder = os.path.join(folder,  f'{date}/')
    save_results(parameters, model, folder, name)


def plot_valid_test_predictions(model, Xvalid, Yvalid, Xtest, Ytest, y_feature, folder, base_name,
                                pred_valid=None, pred_test=None,
                                model_returns_activations=False):
    if pred_valid is not None:
        pred = pred_valid
    else:
        pred = model.predict(Xvalid)
        if model_returns_activations:
            pred = pred[0]
    plot_predictions(Yvalid, pred, folder, base_name, '_valid.png')

    if pred_test is not None:
        pred = pred_test
    else:
        pred = model.predict(Xtest)
        if model_returns_activations:
            pred = pred[0]
    plot_predictions(Ytest, pred, folder, base_name, '_test.png')

    pred_denorm = denorm.denormalize_feature(pred, y_feature)
    Ytest_denorm = denorm.denormalize_feature(Ytest, y_feature)
    plot_predictions(Ytest_denorm, pred_denorm, folder, base_name, '_1.png')


def print_params(params):
    for param in params:
        print(f'{param[0]} = {param[1]}')
