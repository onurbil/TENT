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


def create_conv_lstm(sequence_length, input_size, output_size, num_layers, filters, kernel_size, dropout_rate,
                     padding='same', data_format='channels_first'):
    strides = (1, 1)

    model = kr.Sequential()
    model.add(kr.Input(shape=(sequence_length,) + input_size))

    for i in range(num_layers):
        if i == num_layers - 1:
            model.add(kr.layers.ConvLSTM2D(filters=filters, kernel_size=kernel_size,
                                           strides=strides, padding=padding, dropout=dropout_rate,
                                           data_format=data_format))
        else:
            model.add(kr.layers.ConvLSTM2D(filters=filters, kernel_size=kernel_size,
                                           strides=strides, padding=padding, dropout=dropout_rate,
                                           data_format=data_format, return_sequences=True))

    model.add(kr.layers.Flatten())
    model.add(kr.layers.Dense(output_size, activation=kr.activations.linear))
    model.summary()
    return model


if __name__ == '__main__':
    # lstm = create_lstm(sequence_length=16, input_size=330, output_size=1, num_layers=1, hidden_units=32, dropout_rate=0)
    lstm = create_conv_lstm(sequence_length=16, input_size=(1, 30, 11), output_size=1,
                            num_layers=2, filters=8, kernel_size=3, dropout_rate=0)
