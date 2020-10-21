import os
from pandas import read_csv


def get_data(csv, train_fraction=None):
    df = read_csv(csv, engine='python')
    labels = df['POINT_TYPE'].values
    df.drop(columns=['YEAR', 'POINT_TYPE'], inplace=True)
    data = df.values
    names = df.columns

    x = data
    y = labels.reshape((labels.shape[0],))
    if not train_fraction:
        return x, y
    else:
        # TODO: split data
        return x, y


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
