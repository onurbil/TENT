import matplotlib.pyplot as plt
import tensorflow as tf
import datetime
from tensorflow.keras.callbacks import TensorBoard

def visualize_pos_encoding(array):
    """
    Plot heatmap of positional encoding. 
    """
    plt.pcolormesh(array, cmap='viridis')
    plt.xlabel('Embedding Dimensions')
    plt.xlim((0, array.shape[1]))
    plt.ylabel('Token Position')
    plt.ylim((array.shape[0],0))
    plt.colorbar()
    plt.show()


# To call tensorboard:
#tensorboard --logdir=/notebooks/tensorized_transformers/vanilla_transformer/tb_logs --bind_all
#change logdir to the dir that is needed, it will give a websit to folow, usually with port 6006.

class callbacks():
    def __init__(self, transformer, optimizer, tensorBoard_path,save_checkpoints):
        self.tensorBoard_path = tensorBoard_path + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
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



