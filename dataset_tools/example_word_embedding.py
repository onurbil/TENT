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


docs = ['overcast clouds', 'sky is clear', 'broken clouds', 'fog', 'mist',
 'scattered clouds', 'few clouds', 'light rain', 'light intensity drizzle',
 'moderate rain', 'light intensity shower rain', 'haze', 'heavy shower snow',
 'heavy snow', 'shower snow', 'proximity shower rain', 'snow', 'freezing rain',
 'light rain and snow', 'light snow', 'light shower sleet',
 'light intensity drizzle rain', 'proximity thunderstorm',
 'thunderstorm with light rain', 'heavy intensity rain',
 'thunderstorm with rain', 'very heavy rain', 'smoke', 'thunderstorm', 'dust',
 'light shower snow', 'shower rain', 'shower drizzle', 'sand',
 'thunderstorm with heavy rain', 'heavy intensity shower rain', 'drizzle']

class_count = len(docs)
labels = np.arange(class_count)
        
# prepare tokenizer
t = Tokenizer()
t.fit_on_texts(docs)
vocab_size = len(t.word_index) + 1
# integer encode the documents
encoded_docs = t.texts_to_sequences(docs)
print(encoded_docs)
# pad documents to a max length of 4 words
max_length = 4
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
print(padded_docs)
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
# define model
model = Sequential()
e = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=4, trainable=False)
model.add(e)
model.add(Flatten())
model.add(Dense(class_count, activation='sigmoid'))
# compile the model
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# summarize the model
print(model.summary())
# fit the model
# debug(padded_docs.reshape((-1,1)))
labels = keras.utils.to_categorical(labels, len(docs))
model.fit(padded_docs, labels, epochs=100, verbose=1)
# evaluate the model
loss, accuracy = model.evaluate(padded_docs, labels, verbose=0)
print('Accuracy: %f' % (accuracy*100))