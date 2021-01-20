import os
import sys
from config import model_config, data_config, input_output_size


baseline_config = \
[
    {
        "name": "create_model_vanilla;load_data_beijing_multi",
        "lstm": 100,
        "batch_size": 32,
        "learning_rate": 0.001,
        "optimizer": "Adam"
    },
    {
        "name": "create_model_vanilla;load_data_appliances_multi",
        "lstm": 100,
        "batch_size": 32,
        "learning_rate": 0.001,
        "optimizer": "Adam"
    },
    {
        "name": "create_model_vanilla;load_data_solar_multi",
        "lstm": 100,
        "batch_size": 32,
        "learning_rate": 0.001,
        "optimizer": "Adam"
    },
    {
        "name": "create_model_vanilla;load_data_load_consumption_multi",
        "lstm": 100,
        "batch_size": 32,
        "learning_rate": 0.001,
        "optimizer": "Adam"
    },


    {
        "name": "create_model_vanilla_dropout_batchnormalization;load_data_beijing_multi",
        "lstm": 100,
        "dropout": 0.3,
        "batch_size": 32,
        "learning_rate": 0.001,
        "optimizer": "Adam"
    },
    {
        "name": "create_model_vanilla_dropout_batchnormalization;load_data_appliances_multi",
        "lstm": 100,
        "dropout": 0.3,
        "batch_size": 32,
        "learning_rate": 0.001,
        "optimizer": "Adam"
    },
    {
        "name": "create_model_vanilla_dropout_batchnormalization;load_data_solar_multi",
        "lstm": 100,
        "dropout": 0.3,
        "batch_size": 32,
        "learning_rate": 0.001,
        "optimizer": "Adam"
    },
    {
        "name": "create_model_vanilla_dropout_batchnormalization;load_data_load_consumption_multi",
        "lstm": 100,
        "dropout": 0.3,
        "batch_size": 32,
        "learning_rate": 0.001,
        "optimizer": "Adam"
    },


    {
        "name": "create_model_stacked;load_data_beijing_multi",
        "lstm_0": 100,
        "lstm_1": 100,
        "batch_size": 32,
        "learning_rate": 0.001,
        "optimizer": "Adam"
    },
    {
        "name": "create_model_stacked;load_data_appliances_multi",
        "lstm_0": 100,
        "lstm_1": 100,
        "batch_size": 32,
        "learning_rate": 0.001,
        "optimizer": "Adam"
    },
    {
        "name": "create_model_stacked;load_data_solar_multi",
        "lstm_0": 100,
        "lstm_1": 100,
        "batch_size": 32,
        "learning_rate": 0.001,
        "optimizer": "Adam"
    },
    {
        "name": "create_model_stacked;load_data_load_consumption_multi",
        "lstm_0": 100,
        "lstm_1": 100,
        "batch_size": 32,
        "learning_rate": 0.001,
        "optimizer": "Adam"
    },


    {
        "name": "create_model_stacked_dropout_batchnormalization;load_data_beijing_multi",
        "lstm_0": 100,
        "lstm_1": 100,
        "dropout_0": 0.3,
        "dropout_1": 0.4,
        "batch_size": 32,
        "learning_rate": 0.001,
        "optimizer": "Adam"
    },
    {
        "name": "create_model_stacked_dropout_batchnormalization;load_data_appliances_multi",
        "lstm_0": 100,
        "lstm_1": 100,
        "dropout_0": 0.3,
        "dropout_1": 0.4,
        "batch_size": 32,
        "learning_rate": 0.001,
        "optimizer": "Adam"
    },
    {
        "name": "create_model_stacked_dropout_batchnormalization;load_data_solar_multi",
        "lstm_0": 100,
        "lstm_1": 100,
        "dropout_0": 0.3,
        "dropout_1": 0.4,
        "batch_size": 32,
        "learning_rate": 0.001,
        "optimizer": "Adam"
    },
    {
        "name": "create_model_stacked_dropout_batchnormalization;load_data_load_consumption_multi",
        "lstm_0": 100,
        "lstm_1": 100,
        "dropout_0": 0.3,
        "dropout_1": 0.4,
        "batch_size": 32,
        "learning_rate": 0.001,
        "optimizer": "Adam"
    },


    {
        "name": "create_model_1d_cnn;load_data_beijing_multi",
        "filter_0": 32,
        "filter_1": 64,
        "dense": 64,
        "dropout": 0.3,
        "batch_size": 32,
        "learning_rate": 0.001,
        "optimizer": "Adam"
    },
    {
        "name": "create_model_1d_cnn;load_data_appliances_multi",
        "filter_0": 32,
        "filter_1": 64,
        "dense": 64,
        "dropout": 0.3,
        "batch_size": 32,
        "learning_rate": 0.001,
        "optimizer": "Adam"
    },
    {
        "name": "create_model_1d_cnn;load_data_solar_multi",
        "filter_0": 32,
        "filter_1": 64,
        "dense": 64,
        "dropout": 0.3,
        "batch_size": 32,
        "learning_rate": 0.001,
        "optimizer": "Adam"
    },
    {
        "name": "create_model_1d_cnn;load_data_load_consumption_multi",
        "filter_0": 32,
        "filter_1": 64,
        "dense": 64,
        "dropout": 0.3,
        "batch_size": 32,
        "learning_rate": 0.001,
        "optimizer": "Adam"
    }
]


def get_neural_network(model_, data_, params, epochs_=32, validation_split_=0.2, gpu_memory_fraction_=0.05):
    nn = \
f'''\
import matplotlib.pyplot as plt
import keras
from keras.callbacks import EarlyStopping
import tensorflow as tf
from tensorflow.compat.v1.keras.backend import set_session
import modules.models as models
import modules.load_data as load_data

config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = {gpu_memory_fraction_}
set_session(tf.compat.v1.Session(config=config))

params = {params}

x_train, y_train, x_test, y_test = load_data.{data_}(params)
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

model = models.{model_}(params, x_train.shape[2])

if params['optimizer'] == 'Adam':
    opt = keras.optimizers.Adam(lr=params['learning_rate'])
elif params['optimizer'] == 'RMSprop':
    opt = keras.optimizers.RMSprop(lr=params['learning_rate'])

es = EarlyStopping(monitor='val_loss', mode='min', patience=5)
model.compile(optimizer=opt, loss='mse', metrics=['mse'])
history_model = model.fit(x_train, y_train, validation_split={validation_split_}, epochs={epochs_}, batch_size=params['batch_size'], verbose=1, shuffle=False, callbacks=[es])

loss = history_model.history['loss']
val_loss = history_model.history['val_loss']
plt.plot(loss, 'y', label='Training loss')
plt.plot(val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

ret = model.evaluate(x_test, y_test, verbose=0)
print(ret)
with open(params['name'] + '.txt', 'w') as f:
    f.write(str(ret[0]))
'''
    return nn


def generate_baseline(path):
    idx = 0
    for i in range(0, len(model_config)):
        for j in range(0, len(data_config)):

            name_model = model_config[i]
            name_data = data_config[j]
            config = {**input_output_size[idx%4], **baseline_config[idx]}
            nn = get_neural_network(name_model, name_data, config)
            
            path_baseline = path + str(name_model) + '_' +  str(name_data) + '.py'
            with open(path_baseline, 'w') as f:
                f.write(nn)
            idx += 1


if __name__ == "__main__":

    path = 'baselines/'
    if len(baseline_config) != len(model_config) * len(data_config):
        sys.exit('invalid parameters. check config')

    print('creating baseline...')
    print('num of models: ', len(baseline_config))

    if not os.path.isdir(path):
        os.mkdir(path)

    generate_baseline(path)
    print('done')