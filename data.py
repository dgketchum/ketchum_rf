import os
import numpy as np
from pandas import read_csv


def get_data(csv, mode='binary', train_fraction=None, seed=None):

    assert mode in ['binary', 'multiclass']

    if seed:
        np.random.seed(seed)

    df = read_csv(csv, engine='python')
    labels = df['POINT_TYPE'].values
    df.drop(columns=['system:index', 'YEAR', 'POINT_TYPE', '.geo'], inplace=True)

    if mode == 'binary':
        labels[labels > 1] = 1

    data = df.values
    names = df.columns

    x = data
    y = labels.reshape((labels.shape[0],))
    if not train_fraction:
        return x, y
    else:
        idx = list(np.random.permutation(range(x.shape[0])))
        train_chunk = int(np.floor(len(idx) * train_fraction))
        x_train_idx, x_test_idx = idx[:train_chunk], idx[train_chunk:]
        x_train, x_test = df.iloc[x_train_idx], df.iloc[x_test_idx]
        y_train, y_test = y[x_train_idx], y[x_test_idx]
        return x_train, x_test, y_train, y_test


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
