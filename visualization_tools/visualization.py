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
from common.variables import city_labels
from debugging_tools import *

# from mpl_toolkits.mplot3d import Axes3D


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


def attention_3d_plotter(array, x_labels):

    assert array.shape[0] == array.shape[1]
    assert array.shape[2] == len(x_labels)
    time_length = array.shape[0]
    cities = array.shape[2]
    
    array = array.reshape((cities,time_length,time_length)) 

    x_axis = np.arange(0,cities+1).reshape(cities+1,1)
    y_axis = np.arange(0,time_length+1)
    z_axis = np.arange(0,time_length+1)

    grid_x = x_axis * np.ones((1,time_length+1))
    grid_y = y_axis * np.ones((cities+1,1))

    # scam = plt.cm.ScalarMappable(norm=cm.colors.Normalize(0,1),cmap='Blues')
    scam = plt.cm.ScalarMappable(norm=cm.colors.Normalize(array.min(),array.max()),cmap='Blues')

    fig = plt.figure(figsize=(5,10))
    ax  = fig.gca(projection='3d')

    plt.xticks(np.arange(cities)+0.5)
    plt.yticks(np.arange(time_length)+0.5)
        
    ax.set_zticks(np.arange(time_length+1))
    ax.set_xticklabels(city_labels, rotation=90)
    ax.set_yticklabels(np.arange(1,time_length+1))
    ax.set_zticklabels(np.arange(1,time_length+1))
    
    for label in ax.get_yticklabels()[::2]:
        label.set_visible(False)
    for label in ax.get_zticklabels()[::2]:
        label.set_visible(False)
    
    for i in range(time_length):
        grid_z = np.ones((cities+1,time_length+1)) * i
        
        scam.set_array([])  
        surf = ax.plot_surface(grid_x, grid_y, grid_z,
            facecolors  = scam.to_rgba(array[:,i,:]), 
            antialiased = True, cmap='Blues',
            rstride=1, cstride=1, alpha=None)
            
    fig.colorbar(scam, shrink=0.5, aspect=5)
    
    # """                                                               
    # Scaling is done from here...                                                                                                                           
    # """
    # x_scale=2
    # y_scale=3
    # z_scale=5
    # scale=np.diag([x_scale, y_scale, z_scale, 1.0])
    # scale=scale*(1.0/scale.max())
    # print(scale)
    # scale[3,3]=1
    # 
    # def short_proj():
    #   return np.dot(Axes3D.get_proj(ax), scale)
    # 
    # ax.get_proj=short_proj
    
    plt.show()


# city_labels = ['a','b','c','d']
# test_array = np.random.rand(24,24,36)
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



