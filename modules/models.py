from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Conv1D, Dense, Flatten, MaxPooling1D
from keras.models import Sequential
from keras.layers import BatchNormalization


def create_model_vanilla(hyper_params, features):
    model = Sequential()
    model.add(LSTM(hyper_params['lstm'], input_shape=(hyper_params['win_size'], features)))
    model.add(Dense(hyper_params['out_size']))

    return model


def create_model_vanilla_dropout_batchnormalization(hyper_params, features):
    model = Sequential()
    model.add(LSTM(hyper_params['lstm'], input_shape=(hyper_params['win_size'], features)))
    model.add(BatchNormalization())
    model.add(Dropout(hyper_params['dropout']))
    model.add(Dense(hyper_params['out_size']))

    return model


def create_model_stacked(hyper_params, features):
    model = Sequential()
    model.add(LSTM(hyper_params['lstm_0'], input_shape=(hyper_params['win_size'], features), return_sequences=True))
    model.add(LSTM(hyper_params['lstm_1']))
    model.add(Dense(hyper_params['out_size']))

    return model


def create_model_stacked_dropout_batchnormalization(hyper_params, features):
    model = Sequential()
    model.add(LSTM(hyper_params['lstm_0'], input_shape=(hyper_params['win_size'], features), return_sequences=True))
    model.add(BatchNormalization())
    model.add(Dropout(hyper_params['dropout_0']))
    model.add(LSTM(hyper_params['lstm_1']))
    model.add(BatchNormalization())
    model.add(Dropout(hyper_params['dropout_1']))
    model.add(Dense(hyper_params['out_size']))

    return model


def create_model_1d_cnn(hyper_params, features):
    model = Sequential()
    model.add(Conv1D(filters=hyper_params['filter_0'], kernel_size=3, activation='relu', input_shape=(hyper_params['win_size'], features)))
    model.add(Conv1D(filters=hyper_params['filter_1'], kernel_size=3, activation='relu'))
    model.add(Dropout(hyper_params['dropout']))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(hyper_params['dense'], activation='relu'))
    model.add(Dense(hyper_params['out_size']))

    return model
