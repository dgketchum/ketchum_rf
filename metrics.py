from pandas import read_csv
from data import get_data


def consumer(arr):
    c = [(arr[x, x] / sum(arr[x, :])) for x in range(0, arr.shape[1])]
    print('consumer accuracy: {}'.format(c))


def producer(arr):
    c = [(arr[x, x] / sum(arr[:, x])) for x in range(0, arr.shape[0])]
    print('producer accuracy: {}'.format(c))


# def pca(csv):
#     x, x_test, y, y_test = train_test_split(x, y, test_size=0.33,
#                                             random_state=None)
#     pca = PCA()
#     _ = pca.fit_transform(x)
#     x_centered = x - mean(x, axis=0)
#     cov_matrix = dot(x_centered.T, x_centered) / len(names)
#     eigenvalues = pca.explained_variance_
#     for eigenvalue, eigenvector in zip(eigenvalues, pca.components_):
#         print(dot(eigenvector.T, dot(cov_matrix, eigenvector)))
#         print(eigenvalue)


# def find_rf_variable_importance(csv):
#     first = True
#     master = {}
#
#
#     for x in range(10):
#         print('model iteration {}'.format(x))
#         rf = RandomForestClassifier(n_estimators=100,
#                                     min_samples_split=11,
#                                     n_jobs=-1,
#                                     bootstrap=False)
#
#         rf.fit(data, labels)
#         _list = [(f, v) for f, v in zip(names, rf.feature_importances_)]
#         imp = sorted(_list, key=lambda x: x[1], reverse=True)
#
#         if first:
#             for (k, v) in imp:
#                 master[k] = v
#             first = False
#         else:
#             for (k, v) in imp:
#                 master[k] += v
#
#     master = list(master.items())
#     master = sorted(master, key=lambda x: x[1], reverse=True)
#     pprint(master)



# def random_forest_k_fold(csv):
#     df = read_csv(csv, engine='python')
#     labels = df['POINT_TYPE'].values
#     df.drop(columns=['YEAR', 'POINT_TYPE'], inplace=True)
#     df.dropna(axis=1, inplace=True)
#     data = df.values
#     names = df.columns
#     labels = labels.reshape((labels.shape[0],))
#     kf = KFold(n_splits=2, shuffle=True)
#
#     for train_idx, test_idx in kf.split(data[:-1, :], y=labels[:-1]):
#         x, x_test = data[train_idx], data[test_idx]
#         y, y_test = labels[train_idx], labels[test_idx]
#
#         rf = RandomForestClassifier(n_estimators=100,
#                                     n_jobs=-1,
#                                     bootstrap=False)
#
#         rf.fit(x, y)
#         _list = [(f, v) for f, v in zip(names, rf.feature_importances_)]
#         important = sorted(_list, key=lambda x: x[1], reverse=True)
#         pprint(rf.score(x_test, y_test))
#         y_pred = rf.predict(x_test)
#         cf = confusion_matrix(y_test, y_pred)
#         pprint(cf)
#         producer(cf)
#         consumer(cf)
#
#     return important


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
