import os
import sys
import time
import operator
import argparse
import subprocess


def prepare_experiment(cntr, nni_path='nni'):
    os.system('mkdir ' + nni_path + '/automl/log/' + str(cntr))
    os.system('rm -r ' + nni_path + '/automl/tmp/*')
    os.system('cp -r ' + nni_path + '/automl/modules ' + nni_path + '/automl/tmp')
    os.system('cp -r ' + nni_path + '/automl/data ' + nni_path + '/automl/tmp')
    os.system('cp -r ' + nni_path + '/automl/experiments/' + str(cntr) + '/* ' + nni_path + '/automl/tmp')


def start_experiment(nni_path='nni'):
    os.system('nnictl create --config ' + nni_path + '/automl/tmp/config.yml --port=8081')


def save_experiment_files(cntr, nni_path='nni'):
    os.system('cp -r -a ' + get_newest_experiment() + '/* ' + nni_path + '/automl/log/' + str(cntr))


def get_newest_experiment():
    alist = {}
    directory = os.path.join('/root/nni-experiments')
    os.chdir(directory)
    for file in os.listdir('.'):
        if os.path.isdir(file):
            timestamp = os.path.getmtime(file)
            alist[os.path.join(os.getcwd(), file)] = timestamp

    for i in sorted(alist.items(), key=operator.itemgetter(1)):
        latest = '%s' % (i[0])

    return latest


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Routine for automatic execution of NNI AutoML experiments')
    parser.add_argument('-n', '--start', default=0, type=int, help='start number of experiments')
    parser.add_argument('-t', '--time', type=int, required=True, help='time for each experiments in [h]')
    parser.add_argument('-p', '--nni_path', default='nni', type=str, required=False, help='path to nni directory')
    args = parser.parse_args()

    os.system('mkdir ' + args.nni_path + '/automl/log/')
    os.system('mkdir ' + args.nni_path + '/automl/tmp')

    count_of_experiments = subprocess.check_output('ls ' + args.nni_path + '/automl/experiments -lR | grep ^d | wc -l',
                                                   shell=True)

    for i in range(args.start, (int(count_of_experiments) + args.start)):
        prepare_experiment(i, args.nni_path)
        start_experiment()
        time.sleep(3600 * args.time)
        save_experiment_files(i)
        os.system('nnictl stop')
        time.sleep(10)
