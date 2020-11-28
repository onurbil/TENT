import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from keras import backend as K 
from keras.layers import Layer
from keras.models import Sequential
from visualization_tools.visualization import visualize_pos_encoding


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates

class PositionalEncoding(Layer): 
    
    def __init__(self, model_shape, broadcast=True,**kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.model_shape = model_shape
        self.broadcast = broadcast


    def build(self, input_shape): 
        super(PositionalEncoding, self).build(input_shape)

        
    def call(self, input_data):

        position = self.model_shape[0]
        
        angle_dim = self.model_shape[0]
        if not self.broadcast:
            angle_dim *= self.model_shape[1]

        angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                                np.arange(angle_dim)[np.newaxis, :],
                                angle_dim)
        # apply sin to even indices in the array; 2i
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        # apply cos to odd indices in the array; 2i+1
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        if self.broadcast:
            angle_rads = np.broadcast_to(np.expand_dims(angle_rads, -1), angle_rads.shape + (self.model_shape[1],))
        else:
            new_shape = angle_rads.shape[:-1] + self.model_shape
            angle_rads = np.reshape(angle_rads, new_shape)

        pos_encoding = angle_rads[np.newaxis, ...]

        return tf.cast(pos_encoding, dtype=tf.float32)
        

        
# Test the PositionalEncoding layer:
# test = np.zeros((64,24,216))
test = np.zeros((1,24,36,6))

model = Sequential()
model.add(PositionalEncoding(input_shape = (test.shape[1],test.shape[2],test.shape[3]), model_shape=test.shape, broadcast=True)) 
model.summary()

pos_encoded = model.predict(test)
# print(test)
# print(pos_encoded)


# test = np.zeros((64,24,216))
# bb = positional_encoding(test.shape[0],test.shape, broadcast=True)
bb = pos_encoded[0]

# bb = bb.numpy()
print(bb.shape)
bb = bb[0].reshape((bb.shape[1],-1))
print(bb.shape)
visualize_pos_encoding(bb)