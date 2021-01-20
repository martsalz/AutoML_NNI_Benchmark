pip install tensorflow-gpu
pip install keras
pip install ruamel.yaml
pip install nni

git clone https://github.com/microsoft/nni.git
mkdir nni/automl
python3 src/generate_experiments.py
python3 src/generate_baselines.py

cp -r modules nni/automl/
cp -r data nni/automl/
cp -r experiments nni/automl