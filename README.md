# AutoML Benchmark

This repository was created as part of my master thesis "Hyperparameter tuning with AutoML for time series forecasts in the energy sector" at [Fraunhofer IEE](https://www.iee.fraunhofer.de). The purpose of this repo is to provide code to automatically generate and run a large number of experiments including baselines using [NNI](https://github.com/microsoft/nni). The following algorithms are considered:

  - Random Search
  - Evolution
  - Simulated Annealing
  - Hyperband
  - BOHB
  - Population Based Training

  The following data sets are used for the experiments:
  
    data set                | url
    ----------------------- | -------------
    Solar Production        | https://open-power-system-data.org
    Beijing PM2.5           | https://aqicn.org/city/beijing/
    Appliances energy       | https://archive.ics.uci.edu/ml/datasets/Appliances+energy+prediction
    Load forecast Germany   | https://open-power-system-data.org

  The following models are used to perform the experiments:

  - Vanilla LSTM
  - Vanilla LSTM with dropout and batchnormalization layer
  - Stacked LSTM
  - Stacked LSTM with dropout and batchnormalization layer
  - 1D CNN

The generate_experiments.py is responsible for generating all files for the experiments. Afterwards all necessary files for NNI (neural_network, config.yml, search_space.json) are generated automatically. Feel free to add more algorithms.

### Installation
It is recommended to install all packages in a virtual environment:
```sh
$ python3 -m venv venv_nni
$ source venv_nni/bin/activate
```

Install the dependencies and generate all files for the experiments and baselines:
```sh
$ ./install.sh
```

Alternatively, you can create the files for the experiments and baselines with:
```sh
$ python3 src/generate_experiments.py
$ python3 src/generate_baselines.py
```

Start the experiments:
```sh
$ python3 run.py -t 10
```

The environment variable NNI_OUTPUT_DIR specifies the path to all log files for each trial.

### Additional
Check out my article about AutoML and how you can optimize neural networks automatically.

https://medium.com/@martsalz/automl-hyperparameter-tuning-with-nni-and-keras-ffbef61206cf