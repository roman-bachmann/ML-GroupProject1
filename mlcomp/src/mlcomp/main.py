import os

from mlcomp import config
from mlcomp.data import load_csv_data

if __name__ == '__main__':
    print('Hello! This is the ml project \n')

    print('Reading Data...')
    y_train, x_train, ids_train = load_csv_data(os.path.join(config.DATA_PATH, 'train.csv'), sub_sample=False)
    y_test, x_test, ids_test = load_csv_data(os.path.join(config.DATA_PATH, 'test.csv'), sub_sample=False)

    print('Train shape: {train} \nTest shape: {test}'.format(train=y_train.shape, test=y_test.shape))
