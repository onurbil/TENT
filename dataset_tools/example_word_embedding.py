from numpy import array
from numpy import asarray
from numpy import zeros
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding


"""
Example:

Download glove.6B.zip file from:
https://nlp.stanford.edu/projects/glove/
"""


docs = ['shower drizzle', 'freezing rain', 'volcanic ash', 'proximity shower rain', 'fog', 'shower snow', 'tornado', 'drizzle', 'heavy shower snow', 'few clouds', 'proximity sand/dust whirls', 'mist', 'light rain', 'light shower sleet', 'rain and snow', 'proximity thunderstorm with rain', 'thunderstorm with heavy drizzle', 'overcast clouds', 'sky is clear', 'light rain and snow', 'proximity moderate rain', 'light intensity drizzle rain', 'heavy thunderstorm', 'thunderstorm with rain', 'scattered clouds', 'sand/dust whirls', 'moderate rain', 'broken clouds', 'shower rain', 'smoke', 'haze', 'heavy intensity shower rain', 'sleet', 'squalls', 'heavy snow', 'sand', 'ragged shower rain', 'thunderstorm with heavy rain', 'ragged thunderstorm', 'thunderstorm with light rain', 'thunderstorm with light drizzle', 'light intensity shower rain', 'snow', 'heavy intensity rain', 'light shower snow', 'thunderstorm with drizzle', 'heavy intensity drizzle', 'thunderstorm', 'light snow', 'proximity thunderstorm', 'light intensity drizzle', 'dust', 'proximity thunderstorm with drizzle', 'very heavy rain']

class_count = len(docs)
labels = np.arange(class_count)
        
# prepare tokenizer
t = Tokenizer()
t.fit_on_texts(docs)
vocab_size = len(t.word_index) + 1

# integer encode the documents
encoded_docs = t.texts_to_sequences(docs)

# pad documents to a max length of 4 words
max_length = 4
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
# load the whole embedding into memory
embeddings_index = dict()
f = open('glove.6B.100d.txt')
for line in f:
	values = line.split()
	word = values[0]
	coefs = asarray(values[1:], dtype='float32')
	embeddings_index[word] = coefs
f.close()
print('Loaded %s word vectors.' % len(embeddings_index))
# create a weight matrix for words in training docs
embedding_matrix = zeros((vocab_size, 100))
for word, i in t.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector


model = Sequential()
# e = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=4, trainable=False)
e = Embedding(vocab_size, 3, input_length=4, trainable=True)
model.add(e)
model.add(Flatten())
model.add(Dense(class_count, activation='sigmoid'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print(model.summary())

labels = keras.utils.to_categorical(labels, len(docs))
model.fit(padded_docs, labels, epochs=2000, verbose=1)
# evaluate the model
loss, accuracy = model.evaluate(padded_docs, labels, verbose=0)
print('Accuracy: %f' % (accuracy*100))

from keras.models import Model

layer_name = 'embedding'
intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer(layer_name).output)
intermediate_output = intermediate_layer_model.predict(padded_docs)
print(intermediate_output)
print(intermediate_output.shape)