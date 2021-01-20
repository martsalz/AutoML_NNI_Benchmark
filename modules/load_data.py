import pandas as pd
import numpy as np
import modules.split_sequence as split_sequence
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def load_data_beijing_multi(hyper_params):
    data = pd.read_csv("../data/beijing.csv")
    del data['No']
    del data['year']
    del data['month']
    del data['day']
    del data['hour']

    data.dropna(subset=['pm2.5'], inplace=True)
    data.rename(columns={'pm2.5': 'pollution', 'DEWP': 'dew', 'TEMP': 'temp', 'PRES': 'press',
                         'cbwd': 'wnd_dir', 'Iws': 'wnd_spd', 'Is': 'snow', 'Ir': 'rain'}, inplace=True)
    data.index = np.arange(0, len(data))
    dataMulti = data.values[:, :]

    encoder = LabelEncoder()
    dataMulti[:, 4] = encoder.fit_transform(dataMulti[:, 4])
    dataMulti = dataMulti.astype('float32')

    mn = list()
    mx = list()

    for i in range(0, dataMulti.shape[1]):
        mn.append(dataMulti[:, i].min())
        mx.append(dataMulti[:, i].max())

    scaledMulti = dataMulti.copy()
    for i in range(0, dataMulti.shape[1]):
        scaledMulti[:, i] = (scaledMulti[:, i] - mn[i]) / (mx[i] - mn[i])

    y_train, y_test = train_test_split(scaledMulti, test_size=0.2, random_state=42, shuffle=False)
    X_tr_multi, y_tr_multi = split_sequence.multi_split_sequence(y_train, 0, hyper_params['win_size'], hyper_params['out_size'])
    X_ts_multi, y_ts_multi = split_sequence.multi_split_sequence(y_test, 0, hyper_params['win_size'], hyper_params['out_size'])

    return X_tr_multi, y_tr_multi, X_ts_multi, y_ts_multi


def load_data_appliances_multi(hyper_params):
    data = pd.read_csv("../data/energy.csv")
    data.date = pd.to_datetime(data.date)
    data = data.set_index(data.date)
    data = data.drop(columns=["date"])
    fullData = data.astype('float32')
    dataMulti = np.array(fullData.iloc[:, :])

    mn = list()
    mx = list()

    for i in range(0, dataMulti.shape[1]):
        mn.append(dataMulti[:, i].min())
        mx.append(dataMulti[:, i].max())

    scaledMulti = dataMulti.copy()
    for i in range(0, dataMulti.shape[1]):
        scaledMulti[:, i] = (scaledMulti[:, i] - mn[i]) / (mx[i] - mn[i])

    y_train, y_test = train_test_split(scaledMulti, test_size=0.2, random_state=42, shuffle=False)
    X_tr_multi, y_tr_multi = split_sequence.multi_split_sequence(y_train, 0, hyper_params['win_size'], hyper_params['out_size'])
    X_ts_multi, y_ts_multi = split_sequence.multi_split_sequence(y_test, 0, hyper_params['win_size'], hyper_params['out_size'])

    return X_tr_multi, y_tr_multi, X_ts_multi, y_ts_multi


def load_data_solar_multi(hyper_params):
    ranges = list(range(263001, 324360))
    weather = pd.read_csv("../data/weather.csv").iloc[ranges]
    ranges = list(range(43834, 105194))
    load = pd.read_csv("../data/solar.csv").iloc[ranges]
    data = load.merge(weather, how="outer")
    data = data.fillna(method='ffill')
    data.utc_timestamp = pd.to_datetime(data.utc_timestamp)
    data = data.set_index(data.utc_timestamp)
    data = data.drop(columns=["utc_timestamp"])
    data = data.drop(columns=["cet_cest_timestamp"])
    fullData = data.astype('float32')
    dataMulti = np.array(fullData.iloc[:, :])

    mn = list()
    mx = list()

    for i in range(0, dataMulti.shape[1]):
        mn.append(dataMulti[:, i].min())
        mx.append(dataMulti[:, i].max())

    scaledMulti = dataMulti.copy()
    for i in range(0, dataMulti.shape[1]):
        scaledMulti[:, i] = (scaledMulti[:, i] - mn[i]) / (mx[i] - mn[i])

    y_train, y_test = train_test_split(scaledMulti, test_size=0.2, random_state=42, shuffle=False)
    X_tr_multi, y_tr_multi = split_sequence.multi_split_sequence(y_train, 0, hyper_params['win_size'], hyper_params['out_size'])
    X_ts_multi, y_ts_multi = split_sequence.multi_split_sequence(y_test, 0, hyper_params['win_size'], hyper_params['out_size'])

    return X_tr_multi, y_tr_multi, X_ts_multi, y_ts_multi


def load_data_load_consumption_multi(hyper_params):
    ranges = list(range(306818, 324360))
    weather = pd.read_csv("../data/weather.csv").iloc[ranges]
    ranges = list(range(78891, 96433))
    load = pd.read_csv("../data/load.csv").iloc[ranges]

    data = load.merge(weather, how="outer")
    data = data.fillna(method='ffill')
    data.utc_timestamp = pd.to_datetime(data.utc_timestamp)
    data = data.set_index(data.utc_timestamp)
    data = data.drop(columns=["utc_timestamp"])
    fullData = data.astype('float32')
    dataMulti = np.array(fullData.iloc[:, :])

    mn = list()
    mx = list()

    for i in range(0, dataMulti.shape[1]):
        mn.append(dataMulti[:, i].min())
        mx.append(dataMulti[:, i].max())

    scaledMulti = dataMulti.copy()
    for i in range(0, dataMulti.shape[1]):
        scaledMulti[:, i] = (scaledMulti[:, i] - mn[i]) / (mx[i] - mn[i])

    y_train, y_test = train_test_split(scaledMulti, test_size=0.2, random_state=42, shuffle=False)
    X_tr_multi, y_tr_multi = split_sequence.multi_split_sequence(y_train, 0, hyper_params['win_size'], hyper_params['out_size'])
    X_ts_multi, y_ts_multi = split_sequence.multi_split_sequence(y_test, 0, hyper_params['win_size'], hyper_params['out_size'])

    return X_tr_multi, y_tr_multi, X_ts_multi, y_ts_multi
