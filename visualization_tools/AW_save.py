import seaborn as sns
import os
import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def plot_save(plot_data, folder_name, x_labels, y_labels, x_axis, y_axis, plot_title, save_name, show_plot = False):
  # Checking that the folder exists, if not, create it.
  if not os.path.exists(folder_name):
    os.makedirs(folder_name)
  save_path = folder_name + '/' + str(save_name)

  fig, ax = plt.subplots(figsize=(40,20))
  ax = sns.heatmap(plot_data, annot=True, cmap='Blues')
  plt.xlabel(x_axis, fontsize=16)
  plt.ylabel(y_axis, fontsize=16)
  plt.title(plot_title, fontsize=16)
  bottom, top = ax.get_ylim()
  ax.set_ylim(bottom + 0.2, top - 0.2)
  ax.set_xticklabels(x_labels, rotation=90)
  ax.set_yticklabels(y_labels, rotation=0)
  # Save figure
  fig.savefig(save_path + '.png')
  if show_plot:
    plt.show()
  plt.close()

  x_np = plot_data.numpy()
  x_df = pd.DataFrame(x_np, columns = x_labels, index = y_labels)
  # Save Data
  x_df.to_csv(save_path + '.csv')

def save_weights(model, city_labels, layer=1, folder_name = '/content/drive/MyDrive/Colab Notebooks/Tensorized Transformers/AW'):
  AW = model.layers[layer].attention_weights
  # Saving data + plots for 2 reduction
  for Head in range(AW.shape[0]):
    arr1 = AW[:,0,...][Head]
    plot_data = tf.math.reduce_sum(arr1, axis=-2)
    y_labels = np.arange(plot_data.shape[-2])
    x_labels = city_labels[:AW.shape[-1]]

    plot_save(plot_data,
              folder_name = folder_name + '/Heads',
              x_labels = x_labels ,
              y_labels = y_labels,
              x_axis = 'City',
              y_axis='Input Time',
              plot_title='Attention weights for head {}'.format(Head),
              save_name = 'HeatMapHead{}'.format(Head))

# Saving data + plot for 3 reduction
  arr1 = AW[:,0,...]
  arr2 = tf.math.reduce_sum(arr1, axis=-2)
  plot_data = tf.math.reduce_sum(arr2, axis=-2)
  print(f'Layer: {layer}, AW.shape: {AW.shape}')
  print(f'1st reduction shape: {arr1.shape}')
  print(f'2nd reduction shape: {arr2.shape}')
  print(f'Plot data shape: {plot_data.shape}')
  y_labels = np.arange(plot_data.shape[-2])
  x_labels = city_labels[:AW.shape[-1]]

  plot_save(plot_data,
            folder_name = folder_name,
            x_labels=x_labels,
            y_labels=y_labels,
            x_axis = 'City',
            y_axis='Heads',
            plot_title='Attention weights for Cities x Heads',
            save_name = 'HeatMapTotal',
            show_plot = True)