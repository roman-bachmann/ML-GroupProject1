import os
import pandas as pd

from mlcomp import config

if __name__ == '__main__':
    print('Hello! This is the ml project \n')

    print('Reading Data...')
    train = pd.read_csv(os.path.join(config.DATA_PATH, 'train.csv'))
    test = pd.read_csv(os.path.join(config.DATA_PATH, 'test.csv'))

    print('Train shape: {train} \nTest shape: {test}'.format(train=train.shape, test=test.shape))
