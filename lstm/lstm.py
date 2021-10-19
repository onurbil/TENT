import tensorflow as tf
import tensorflow.keras as kr


def create_lstm(sequence_length, input_size, output_size, num_layers, hidden_units, dropout_rate):
    model = kr.Sequential()
    model.add(kr.Input(shape=(sequence_length, input_size)))

    for i in range(num_layers):
        if i == num_layers - 1:
            model.add(kr.layers.LSTM(units=hidden_units, dropout=dropout_rate))
        else:
            model.add(kr.layers.LSTM(units=hidden_units, return_sequences=True, dropout=dropout_rate))

    model.add(kr.layers.Dense(output_size, activation=kr.activations.linear))
    model.summary()
    return model


if __name__ == '__main__':
    lstm = create_lstm(16, 330, 1, 2, 32, 0.1, 64)
