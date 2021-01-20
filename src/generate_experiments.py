import os
import csv
import json
import ruamel.yaml as yaml
from config import model_config, data_config, input_output_size


# ==== NNI SETTINGS ====
duration = 10 #hours
max_trial = 1000000
trial_concurrency = 3

nn_file_name_py = 'test.py'
command = f'python3 {nn_file_name_py}'
search_space_json = 'search_space.json'
config_yml = 'config.yml'

use_active_gpu = 'true'
gpu_num = 0
# ======================

tuner_config_global = \
f'''
authorName: default
experimentName: automl
maxExecDuration: {duration}h
maxTrialNum: {max_trial}
trialConcurrency: {trial_concurrency}
localConfig:
    useActiveGpu: {use_active_gpu}
    maxTrialNumPerGpu: 5
trainingServicePlatform: local
searchSpacePath: {search_space_json}
useAnnotation: false
trial:
    command: {command}
    codeDir: .
    gpuNum: {gpu_num}
'''

tuner_config = [
    f'''\
    tuner:
        builtinTunerName: Random
    ''',

    f'''\
    tuner:
        builtinTunerName: Evolution
        classArgs:
            optimize_mode: minimize
            population_size: 10
    ''',

    f'''\
    tuner:
        builtinTunerName: Anneal
        classArgs:
            optimize_mode: minimize
    ''',

    f'''\
    advisor:
        builtinAdvisorName: Hyperband
        classArgs:
            R: 32
            eta: 2
            optimize_mode: minimize
    ''',
        
    f'''\
    advisor:
        builtinAdvisorName: BOHB
        classArgs:
            min_budget: 4
            max_budget: 32
            eta: 2
            optimize_mode: minimize
    ''',

    f'''\
    tuner:
        builtinTunerName: PBTTuner
        classArgs:
            optimize_mode: minimize
            population_size: 10
    '''
]

search_space_config_global = \
{
    "optimizer":{"_type":"choice","_value":["Adam", "RMSprop"]},
    "learning_rate":{"_type":"uniform","_value":[0.0001, 0.1]},
    "batch_size": {"_type":"randint", "_value": [8, 128]}
}

search_space_config = \
[
    {
        "lstm":{"_type":"randint","_value":[5, 200]}
    },
    {
        "lstm":{"_type":"randint","_value":[5, 200]},
        "dropout":{"_type":"uniform","_value":[0.01, 0.5]}
    },
    {
        "lstm_0":{"_type":"randint","_value":[5, 200]},
        "lstm_1":{"_type":"randint","_value":[5, 200]}
    },
    {
        "lstm_0":{"_type":"randint","_value":[5, 200]},
        "lstm_1":{"_type":"randint","_value":[5, 200]},
        "dropout_0":{"_type":"uniform","_value":[0.01, 0.5]},
        "dropout_1":{"_type":"uniform","_value":[0.01, 0.5]}
    },
    {
        "filter_0":{"_type":"randint","_value":[5, 200]},
        "filter_1":{"_type":"randint","_value":[5, 200]},
        "dense":{"_type":"randint","_value":[5, 200]},
        "dropout":{"_type":"uniform","_value":[0.01, 0.5]}
    }
]


def get_neural_network(model_, data_, input_output_, epochs_=32, validation_split_=0.2, gpu_memory_fraction_=0.05):
    nn = \
f'''\
import logging
import keras
import nni
import tensorflow as tf
from tensorflow.compat.v1.keras.backend import set_session
from keras.callbacks import EarlyStopping
import modules.models as models
import modules.load_data as load_data

config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = {gpu_memory_fraction_}
set_session(tf.compat.v1.Session(config=config))

trial_path = os.environ['NNI_OUTPUT_DIR']
LOG = logging.getLogger('automl')

class SendMetrics(keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs={{}}):
        nni.report_intermediate_result(logs["val_loss"])

def train(params):
    input_output_size = {input_output_}

    x_train, y_train, x_test, y_test = load_data.{data_}(input_output_size)
    model = models.{model_}(params, input_output_size['win_size'], input_output_size['out_size'], x_train.shape[2])

    if params['optimizer'] == 'Adam':
        optimizer = keras.optimizers.Adam(lr=params['learning_rate'])
    elif params['optimizer'] == 'RMSprop':
        optimizer = keras.optimizers.RMSprop(lr=params['learning_rate'])

    es = EarlyStopping(monitor='val_loss', mode='min', patience=5)
    model.compile(loss=keras.losses.mse, optimizer=optimizer, metrics=['mse'])
    model.fit(x_train, y_train, batch_size=params['batch_size'], epochs={epochs_}, verbose=1,
        validation_split={validation_split_}, callbacks=[SendMetrics(), es])
    model.save_model(trial_path + '/model')

    _, mse = model.evaluate(x_test, y_test, verbose=0)
    nni.report_final_result(mse)

if __name__ == '__main__':

    try:
        RECEIVED_PARAMS = nni.get_next_parameter()
        LOG.debug(RECEIVED_PARAMS)
        train(RECEIVED_PARAMS)
    except Exception as e:
        LOG.exception(e)
        raise
'''
    return nn


def extract_tuner_name(pos):
    text = tuner_config[pos]
    matched_lines = [line for line in text.split('\n') if ('builtinTunerName' in line or 'builtinAdvisorName' in line)]
    text = matched_lines[0].split(',')
    return text[0].split(':')[-1].strip()


def write_to_file(path, data, format, mode='w'):
    with open(path, mode) as f:
        if format == 'yaml':
            yaml.dump(data, f, yaml.RoundTripDumper)
        elif format == 'json':
            json.dump(data, f, indent=4)
        elif format == 'csv':
            csv_writer = csv.writer(f)
            csv_writer.writerow(data)
        elif format == 'text':
            f.write(str(data))
        else:
            print('unknown format')


def generate_experiments(path):
    with open(path + 'log_info.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['num', 'tuner', 'data', 'model'])

    idx = 0
    for i in range(0, len(tuner_config)):
        for j in range(0, len(data_config)):
            for k in range(0, len(model_config)):
                if not os.path.isdir(path + str(idx)):
                    os.mkdir(path + str(idx))

                tuner_yml = yaml.load(tuner_config_global + tuner_config[i], Loader=yaml.RoundTripLoader)
                write_to_file(path + str(idx) + f'/{config_yml}', tuner_yml, 'yaml')

                search_space = {**search_space_config_global, **search_space_config[k]}
                write_to_file(path + str(idx) + f'/{search_space_json}', search_space, 'json')

                name_data = data_config[j]
                name_model = model_config[k]
                epochs = "params['TRIAL_BUDGET']" if ('Hyperband' in tuner_config[i] or 'BOHB' in tuner_config[i]) else 32
                nn = get_neural_network(name_model, name_data, input_output_size[j], epochs)
                write_to_file(path + str(idx) + f'/{nn_file_name_py}', nn, 'text')

                log_data = [str(idx), extract_tuner_name(i), data_config[j], model_config[k]]
                write_to_file(path + 'log_info.csv', log_data, 'csv', 'a')
                idx += 1


if __name__ == "__main__":

    path = 'experiments/'
    print('creating experiments...')
    count_of_experiments = len(tuner_config) * len(model_config) * len(data_config)
    print('num of experiments: ', count_of_experiments)

    if not os.path.isdir(path):
        os.mkdir('experiments')

    generate_experiments(path)
    print('done')