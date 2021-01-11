import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense

# Create dummy data
weather_class_count = 56
cities = 29
x_train = np.random.randint(weather_class_count, size=(32,29,1))
y_train = np.random.randint(weather_class_count, size=(32))


model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(56, 3, input_length=cities))
# The layers below are not needed in our model:
model.add(Flatten())
model.add(Dense(weather_class_count, activation='softmax'))

print(model.summary())

model.compile(optimizer='Adam', loss='mae')
model.fit(x_train, y_train, epochs=10, verbose=1)

output_array = model.predict(x_train)
print(output_array.shape)