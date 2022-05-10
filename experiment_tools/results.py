import os
import datetime
import tensorflow.keras as kr
import numpy as np
import matplotlib.pyplot as plt
import dataset_tools.denormalization as denorm
import pickle


def plot_predictions(Ys, pred, folder, name, ending):
    fileName = folder + name
    pred = pred.flatten()
    Ys = Ys.flatten()
    mae = kr.metrics.mae(Ys, pred)
    mse = kr.metrics.mse(Ys, pred)
    f = open(fileName + ".txt", "a")
    print(f'Figure mae{ending}: {np.mean(mae)}', file=f)
    print(f'Figure mse{ending}: {np.mean(mse)}', file=f)
    print('\n', file=f)
    f.close()
    print(f'Figure mae: {np.mean(mae)}')
    print(f'Figure mse: {np.mean(mse)}')

    plot_width = 80 if Ys.size < 1000 else 100
    plt.figure(figsize=(plot_width, 8))
    # NEW!
    plt.style.use('seaborn-darkgrid')
    my_dpi = 96
    plt.figure(figsize=(1200 / my_dpi, 480 / my_dpi), dpi=my_dpi)

    plt.plot(range(pred.size), pred, label='pred')
    plt.plot(range(len(Ys)), Ys, label='true')
    plt.legend()
    plt.savefig(fileName + ending)
    plt.show()
    return mae, mse


def save_results(parameters, model, folder, name):
    fileName = os.path.join(folder, name)
    print(f'saving to folder: {folder}')
    os.makedirs(os.path.dirname(folder), exist_ok=True)
    if name.startswith('MultiConv'):
        f = open(fileName + ".txt", "a")
        print(model.load_state_dict, file=f)
    else:
        with open(fileName + ".txt", "a") as fh:
            model.summary(print_fn=lambda x: fh.write(x + '\n'))
        f = open(fileName + ".txt", "a")
        print("\n######################### Model Run ##########################################", file=f)
        history = model.history.history
        keys = model.history.history.keys()
        try:
          size = len(history['loss'])
        except:
          size = 0
        print(f'Number of epochs: {size}', file=f)
        for ii in range(size):
            for i in keys:
                print(f'Epoch_{ii}_{i} : {history[i][ii]}', file=f)
            print('\n', file=f)
    print("######################## Model description ################################", file=f)
    for name, results in parameters:
        print(str(name) + " = " + str(results), file=f)
    print("\n######################### Results ##########################################", file=f)
    f.close()


def save_results_with_datetime(model, base_name, path, parameters):
    date = datetime.datetime.now().strftime("%Y_%m_%d")
    time = datetime.datetime.now().strftime("%H_%M_%S")
    name = f'{base_name}_{time}'
    folder = os.path.join(path, 'Tests/')
    folder = os.path.join(folder, f'{date}/')
    save_results(parameters, model, folder, name)
    return folder, name


def plot_valid_test_predictions(model, Xvalid, Yvalid, Xtest, Ytest, folder, base_name,
                                y_feature=None, denorm_min=None, denorm_max=None,
                                pred_valid=None, pred_test=None,
                                model_returns_activations=False):
    if pred_valid is not None:
        pred = pred_valid
    else:
        pred = model.predict(Xvalid)
        if model_returns_activations:
            pred = pred[0]
    valid_mae, valid_mse = plot_predictions(Yvalid, pred, folder, base_name, '_valid_improved.png')

    if pred_test is not None:
        pred = pred_test
    else:
        pred = model.predict(Xtest)
        if model_returns_activations:
            pred = pred[0]
    test_mae, test_mse = plot_predictions(Ytest, pred, folder, base_name, '_test_improved.png')

    pred_denorm = denorm.denormalize_feature(pred, y_feature, denorm_min, denorm_max)
    Ytest_denorm = denorm.denormalize_feature(Ytest, y_feature, denorm_min, denorm_max)
    test_denorm_mae, test_denorm_mse = plot_predictions(Ytest_denorm, pred_denorm, folder, base_name, '_denormalized_improved.png')

    fileName = os.path.join(folder, base_name + '_values.txt')
    print(fileName)
    saving_values = []
    saving_values.append(pred_denorm)
    saving_values.append(Ytest_denorm)
    saving_values.append(folder)
    saving_values.append(base_name)
    saving_values.append('_denormalized_improved.png')
    saving_values.append(test_denorm_mae)
    saving_values.append(test_denorm_mse)

    with open(fileName, 'wb') as fp:
      pickle.dump(saving_values, fp)

    return valid_mae, valid_mse, test_mae, test_mse, test_denorm_mae, test_denorm_mse


def print_params(params):
    for param in params:
        print(f'{param[0]} = {param[1]}')
