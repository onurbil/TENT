import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import datetime
from tensorflow.keras.callbacks import TensorBoard
import seaborn as sns
from matplotlib.patches import Rectangle
from matplotlib import cm
from debugging_tools import *

def visualize_pos_encoding(array):
    """
    Plot heatmap of positional encoding for the array.
    """
    plt.pcolormesh(array, cmap='viridis')
    plt.xlabel('Embedding Dimensions')
    plt.xlim((0, array.shape[1]))
    plt.ylabel('Token Position')
    plt.ylim((array.shape[0],0))
    plt.colorbar()
    plt.show()


def attention_plotter(attention_weights, y_labels, save=False):
    """
    Visualization for attention weights for each input:
    Inputs:
    attention_weights: Numpy array with shape (batch,number of attention heads)
    y_labels: Labels of y-axis (here: daily hours) in a list with batch length.
    """
    assert(attention_weights.shape[0]==len(y_labels)
           ), 'Attention weight has size {} and labels has size {}!'.format(
           attention_weights.shape[0], len(y_labels))
    # Get for each column the cell with maximal attention:
    max_attention = np.argmax(attention_weights, axis=1)
        
    ax = sns.heatmap(attention_weights, annot=True, cmap='Blues')
    ax.set(xlabel='Time steps', ylabel='Time steps',
           title='Visualization of Attention Weights')
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    ax.set_yticklabels(y_labels, rotation=0)
    # Draw rectangle around the cell with maximum attention:
    for row, variable in enumerate(max_attention):
        ax.add_patch(Rectangle((variable,row),1,1,
                     fill=False, edgecolor='red', lw=3))
    plt.show()
    if save:
        fig = ax.get_figure()
        fig.savefig("Attention Visualization.png")


def attention_3d_plotter(array, x_labels, save=False):

    assert array.shape[0] == array.shape[1]
    time_length = array.shape[0]
    cities = array.shape[2]
    
    array = array.reshape((cities,time_length,time_length)) 

    x_axis = np.arange(1,cities+1).reshape(cities,1)
    y_axis = np.arange(1,time_length+1)
    z_axis = np.arange(1,time_length+1)

    grid_x = x_axis * np.ones((1,time_length))
    grid_y = y_axis * np.ones((cities,1))

    scam = plt.cm.ScalarMappable(norm=cm.colors.Normalize(0,1),cmap='jet')
    fig = plt.figure()
    ax  = fig.gca(projection='3d')
    ax.set_xlim(1,5, auto=True)
    ax.set_xticklabels(city_labels, rotation=0)

    for i in range(time_length):
        
        grid_z = np.ones((cities,time_length)) * (i+1)
        scam.set_array([])  
        surf = ax.plot_surface(grid_x, grid_y, grid_z,
            facecolors  = scam.to_rgba(array[:,i,:]), 
            antialiased = True,
            rstride=1, cstride=1, alpha=None)
            
    fig.colorbar(scam, shrink=0.5, aspect=5)
    plt.show()
    if save:
        fig = ax.get_figure()
        fig.savefig("Attention Visualization 3D.png")



# test_array = np.random.rand(8,8,4)
# city_labels = ['a','b','c','d']
# attention_3d_plotter(test_array, city_labels)
    

# To call tensorboard:
#tensorboard --logdir=/notebooks/tensorized_transformers/vanilla_transformer/tb_logs --bind_all
#change logdir to the dir that is needed, it will give a websit to folow, usually with port 6006.

class callbacks():
    def __init__(self, transformer, optimizer, tensorBoard_path,save_checkpoints):
        self.tensorBoard_path = tensorBoard_path + datetime.now().strftime("%Y%m%d-%H%M%S")
        self.save_checkpoints = save_checkpoints
        self.ckpt = tf.train.Checkpoint(transformer=transformer,
                                        optimizer=optimizer)
        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt,
                                                      tensorBoard_path + 'checkpoints/train',
                                                      max_to_keep=5)
        self.calbacks = []
        self.calbacks.append(['train', tf.summary.create_file_writer(self.tensorBoard_path + '/train')])
        self.calbacks.append(['test', tf.summary.create_file_writer(self.tensorBoard_path + '/test')])
    def store_data(self, measure, result, step, type):
        for name, writer in self.calbacks:
            if name == type:
                with writer.as_default():
                    tf.summary.scalar(measure, result, step=step)
    def save_checkpoint(self,epoch):
        if self.save_checkpoints and (epoch + 1) % 2 == 0:
            ckpt_save_path = self.ckpt_manager.save()
            print('Saving checkpoint for epoch {} at {}'.format(epoch + 1, ckpt_save_path))



