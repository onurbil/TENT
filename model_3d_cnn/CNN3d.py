from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Flatten, Lambda, Conv3D
import tensorflow as tf

"""
Based on: S. Mehrkanoon, 
'Deep shared representation learning for weather elements forecasting' 
Knowledge-Based Systems, vol. 179, pp. 120–128, 2019.
"""


def ThreeDimCNN_parallel_output(stations, lags, features, filters, kernSize):
    input1 = Input(shape=(stations, lags, features))
    input2 = Lambda(lambda X: tf.expand_dims(X, axis=4))(input1)
    block1 = Conv3D(filters, (kernSize, kernSize, kernSize), padding='same', activation='relu')(input2)
    block1 = Flatten(name='flatten')(block1)
    block2 = Dense(100, activation='relu')(block1)  # number from the original paper
    output1 = Dense(1, activation='linear')(block2)
    return Model(inputs=input1, outputs=output1)

# parameters from the original paper
# stations = 5
# features = 4
# lags = 4
# filters = 10
# kernel_size = 2
# model = ThreeDimCNN_parallel_output(stations, lags, features,  filters, kernel_size)
# model.summary()
