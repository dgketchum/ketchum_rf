import os
import numpy as np
from data import get_data


class Node():
    def __init__(self, depth=0, min_samples_leaf=2, min_samples_split=4):
        self.depth = depth
        self.f_idx = None
        self.value = None
        self.leaf = False
        self.right_child, self.left_child = None, None
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split

    def recursive_nodes(self, x, y, max_depth=8):

        self.feat_idxs = np.array(list(range(0, x.shape[1])))

        if self.depth < max_depth and x.shape[0] > self.min_samples_split:

            self.f_idx, self.split_value, group_1, group_2 = \
                self._get_split(x, y)

            if self.f_idx is not np.NaN:
                self.left_child = Node(self.depth + 1)
                self.right_child = Node(self.depth + 1)
                self.left_child.recursive_nodes(*group_1)
                self.right_child.recursive_nodes(*group_2)
            else:
                self.leaf = True
                _classes, counts = np.unique(y, return_counts=True)
                self.label = _classes[np.argmax(counts)]
        else:
            self.leaf = True
            _classes, counts = np.unique(y, return_counts=True)
            self.label = _classes[np.argmax(counts)]

    def _get_split(self, x, y):
        gini_scores = []
        split_values = []
        splits = []
        for feature_idx in self.feat_idxs:
            g, s_value, s = self._split(x, y, feature_idx)
            gini_scores.append(g)
            split_values.append(s_value)
            splits.append(s)

        arg_min = np.nanargmin(gini_scores)
        group_1, group_2 = splits[arg_min]
        return self.feat_idxs[arg_min], split_values[arg_min], group_1, group_2

    def _split(self, x, y, feature_idx):

        gini = []
        splits = []
        split_values = []
        series = x[:, feature_idx]

        for split_value in series:

            bool_mask = x[:, feature_idx] < split_value
            group_1 = (x[bool_mask], y[bool_mask])
            group_2 = (x[bool_mask == 0], y[bool_mask == 0])

            def needs_split(groups):
                for g in groups:
                    if g[0].shape[0] < self.min_samples_leaf:
                        return False
                return True

            if needs_split((group_1, group_2)):
                s = [group_1, group_2]
                gini.append(self._gini_impurity(*s))
                splits.append((group_1, group_2))
                split_values.append(split_value)

        if len(gini) == 0:
            return np.nan, np.nan, None

        arg_min = np.argmin(gini)
        return gini[arg_min], split_values[arg_min], splits[arg_min]

    def _gini_impurity(self, *groups):
        m = np.sum([group[0].shape[0] for group in groups])
        gini = 0.0
        for group in groups:
            y = group[1]
            group_size = y.shape[0]
            _, class_count = np.unique(y, return_counts=True)
            proportions = class_count / group_size
            weight = group_size / m
            gini += (1 - np.sum(proportions ** 2)) * weight
        return gini

    def predict(self, x):
        if self.leaf:
            return self.label
        else:
            if x[self.f_idx] < self.split_value:
                return self.left_child.predict(x)
            else:
                return self.right_child.predict(x)


class DecisionTree():
    def __init__(self, x, y, n_features, f_idxs, idxs, depth=10, min_leaf=5):
        self.x, self.y, self.idxs, self.min_leaf, self.f_idxs = x, y, idxs, min_leaf, f_idxs
        self.depth = depth
        self.n_features = n_features
        self.n, self.c = len(idxs), x.shape[1]
        self.root = Node(depth=0)
        self.root.recursive_nodes(self.x, self.y, max_depth=8)

    def predict(self, x):
        return self.root.predict(x)


class RF_Classifier():
    def __init__(self, x, y, sample_size, n_trees, n_features='sqrt', depth=10, min_leaf=5):

        if n_features == 'sqrt':
            self.n_features = int(np.sqrt(x.shape[1]))
        else:
            self.n_features = n_features

        self.x, self.y, self.sample_sz, self.depth, self.min_leaf = x, y, sample_size, depth, min_leaf
        self.trees = [self.create_tree() for i in range(n_trees)]

    def create_tree(self):
        idxs = np.random.permutation(len(self.y))[:self.sample_sz]
        f_idxs = np.random.permutation(self.x.shape[1])[:self.n_features]

        return DecisionTree(self.x[idxs],
                            self.y[idxs],
                            self.n_features,
                            f_idxs,
                            idxs=np.array(range(self.sample_sz)),
                            depth=self.depth,
                            min_leaf=self.min_leaf)

    def predict(self, x):
        predictions = [t.predict(x) for t in self.trees]
        (_, idx, counts) = np.unique(np.array(predictions), return_index=True, return_counts=True)
        index = idx[np.argmax(counts)]
        mode = predictions[index]
        return mode


if __name__ == '__main__':
    _csv = 'irrmapper_training_sample.csv'
    x_tr, x_te, y_tr, y_te = get_data(_csv, train_fraction=0.6, mode='multiclass')
    rf = RF_Classifier(x_tr, y_tr, sample_size=100, n_trees=3, depth=3)
    for i in range(x_te.shape[0]):
        x_, y_ = x_te[i, :], y_te[i]
        if rf.predict(x_) != y_:
            print(rf.predict(x_), y_)
# ========================= EOF ====================================================================
