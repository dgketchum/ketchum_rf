import os
import numpy as np
from data import get_data
from sklearn.datasets import make_moons

"""
David Ketchum
November 5, 2020
Implementation of Random Forest, after Breiman, L., 2001. Random forests. Machine learning, 45(1), pp.5-32.

This implementation uses three classes: RF_Classifier, DecisionTree, and Node.

Class doc-strings explain in detail what each class is doing to build a succcessful classifier.

The recursive nodes approach is credited to github.com/sachaMorin.
"""


class Node():
    """
    The Node object represents the point at the base of two branches, where a vector of features must be split.

    The DecisionTree starts with a parent Node at depth=0, and then grows the tree by spliting data into branches.
    A 'split' means we're dividing the data in two, based on a specific value of a specific feature in the
    data. The way to find what value of what feature on which to base the split is to test the purity of the split,
    i.e., how mixed the resulting labels are on each side of the split. Here, we use the Gini impurity measure,
    (explained in its method doc-string). See splitting examples below.

    Before a split is made, Nodes checks to see if the depth of the tree has exceeded max_depth, and that there
    are more than min_samples_split samples in the node. If these tests pass, the node splits the samples into
    groups 1 and 2, groups 1 and 2 undergo feature and feature split value search (again using Gini), and are split
    again. This goes on, recursively, until the depth or the min_samples_split tests fail, when the node is
    marked as a leaf and assigned the label of the majority of samples within it. This implemenation is 'greedy',
    meaning that it checks every possible split (where each side has more than min_samples_leaf) for every
    feature.

    In prediction, the nodes are traveresed again, starting from the parent node, until the sample reaches a
    leaf, which has has a label, and the prediction is made that the sample is indeed that label.

    Example of perfect split: classes [cars, motorcycles] has feature num_wheels. Using Gini, we find impurity of
    zero when we use the feature num_wheels, and base the split on 'less than 4'. We put all samples with
    num_wheels = 2 on one side of the split, and all samples with num_wheels >= 4 on the other side of the split.
    When we test the impurity, we see that the population on one side is all motorcycles, and the population on
    the other side is all cars. Since 'windshield', 'passengers', 'headlights' all have values that overlap
    somewhat between cars and motorcycles, we find larger (worse) impurity values when we test their possible splits,
    so we choose num_wheels < 4 and move on.

    Example of bad split: classes [Rick, Roger] has feature height. Using Gini, we look through all the data we have
    about our population of men named Rick and Roger. When we look at 'height', and pick some value (e.g. 175 cm),
    we find that when we split all the men into two groups depending on whether they are greater than, or less than
    or equal to this measure, the resulting populations on each side of the split are both mixed with Ricks and
    Rogers. Because men's names aren't really correlated with height, this is a poor feature (and value) for a
    split. This will give a value that approaches 0.5 in the binary case.

    :param depth: depth of the current node
    :param min_samples_leaf: minimum samples needed to be considered a leaf
    :param min_samples_split: minimum samples needed to contiue building with a new split
    """

    def __init__(self, depth=0, min_samples_leaf=2, min_samples_split=4):

        self.depth = depth
        self.f_idx = None
        self.value = None
        self.leaf = False
        self.right_child, self.left_child = None, None
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split

    def recursive_nodes(self, x, y, max_depth=8):
        """Build new nodes, split, and create child nodes recursively."""
        self.feat_idxs = np.array(list(range(0, x.shape[1])))

        if self.depth < max_depth and x.shape[0] > self.min_samples_split:

            self.f_idx, self.split_value, group_1, group_2 = self._get_split(x, y)

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
        """Iterate over each feature and each possible split to find a good split for the data"""
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
        """Iterate over each possible split of the selected feature, find lowest Gini score"""
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
        """ Gini Impurity Score
        Gini Impurity calculates the 'mixing' of classes given a split. A good split will perfectly split classes
        and the Gini score will be zero. For a binary problem, a poorly split class will have near equal number
        of each class in each group, and approach 0.5. A perfect split gives Gini score = 0.
        """
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
    """
    The Decision Tree constructor.

    Here we construct the decision tree. This a stucture that consists of a recursive structure of nodes,
    each of which, once trained, holds a feature and the split in the feature, as explained in Node docs.
    :param x:
    :param y:
    :param n_features:
    :param feature_idxs:
    :param min_leaf:
    """

    def __init__(self, x, y, n_features, feature_idxs, min_leaf=5):

        self.x, self.y, self.min_leaf, self.f_idxs = x, y, min_leaf, feature_idxs
        self.n_features = n_features
        self.c = x.shape[1]
        self.root = Node(depth=0)
        self.root.recursive_nodes(self.x, self.y, max_depth=8)

    def predict(self, x):
        return self.root.predict(x)


class RF_Classifier():
    """A Random Forest Classifier"""

    def __init__(self, x, y, n_trees=10, sample_size=None, n_features='sqrt', min_leaf=5, max_depth=8):
        """
        Create Random Forest Classifier

        Random Forests reduce variance by bootstrap aggregation (bagging), i.e., a sample data set (of sample_size)
        is randomly selected from the data, with replacement, meaning there can be duplicate data samples. As the
        bootstrapped sample approaches the number of total training samples, the bootstrap sample is expected to
        have about 63% of the unique samples in the original data. 'Bagging' is bootstrap aggregating, i.e., generating
        a collection of bootstrap datasets (size n_trees) and using each to train a DecisionTree, then using the
        collection of DecisionTree instances as an ensemble. In classification a prediction is made by simply
        predicting the label of a data point on each DecisionTree, and choosing the mode of all predictions.

        Breiman found that to reduce generalization error (i.e., improve prediction on new data), withstand noise and
        outliers, run in parallel, and provide internal estimates of error and feature importance, it was helpful to
        randomly select a subset of features. It's since been found that a good option sub-selection of features
        is the square root of total features, default here (i.e., n_features).

        Random Forest performs well because it overcomes the high variance of potentially overfit models (DecisionTrees)
        by constructing many of them (n_trees) and using an ensemble approach to the prediction. Further, the algorithm
        gives each DecisionTree independence, as the features selected build the tree, and the data samples used
        to train the tree are random.

        Note: This implementation does not support out-of-bag (OOB) error estimates, which is one of the great features
        of Random Forests. To make OOB error estimate, we would track each data sample not selected in the bootstrap
        sample for use in the construction of a given DecisionTree. After construction, we'd predict each data sample
        to the population of trees that were trained without that sample. It's expected that as the data set size
        and number of trees increase, the OOB error estimate will converge with that of a held-out test data set.
        See https://www.stat.berkeley.edu/~breiman/OOBestimation.pdf

        :param x: the data. ndarray of shape (data instances, data_features)
        :param y: the labels. ndarray of shape (data instances,)
        :param sample_size: number of samples to draw from data
        :param n_trees: number of decision trees to construct
        :param n_features: number of features to use in each tree
        :param min_leaf: minimum number of samples in each of left and right branches to be considered a
        leaf (terminal) node
        :param max_depth: maximum tree depth

        """
        if n_features == 'sqrt':
            self.n_features = int(np.sqrt(x.shape[1]))
        else:
            self.n_features = n_features

        self.x, self.y, self.sample_sz, = x, y, sample_size,
        self.min_leaf, self.max_depth = min_leaf, max_depth
        self.trees = [self.create_tree() for i in range(n_trees)]

    def create_tree(self):
        """Bootstrap sample creation and DecisionTree construction"""
        #  randomly select data with replacement to get bootstap data set
        idxs = np.arange(self.y.shape[0])

        if not self.sample_sz:
            sample_idxs = np.random.choice(idxs, replace=True)
        else:
            sample_idxs = np.random.choice(idxs, self.sample_sz, replace=True)

        #  random selection of features
        f_idxs = np.random.permutation(self.x.shape[1])[:self.n_features]

        return DecisionTree(self.x[idxs], self.y[idxs], n_features=self.n_features, feature_idxs=f_idxs,
                            min_leaf=self.min_leaf)

    def predict(self, x):
        predictions = [t.predict(x) for t in self.trees]
        (_, idx, counts) = np.unique(np.array(predictions), return_index=True, return_counts=True)
        index = idx[np.argmax(counts)]
        mode = predictions[index]
        return mode


def get_metrics(y, pred, n_class=2):
    """Construct a confusion matrix"""
    batch_conf = np.zeros((n_class, n_class))
    for i in range(len(y)):
        batch_conf[y[i]][pred[i]] += 1
    return batch_conf


def classify_moons():
    """Classify overlapping moons"""
    x_tr, y_tr = make_moons(n_samples=1000, noise=0.3)
    x_te, y_te = make_moons(n_samples=1000, noise=0.3)
    rf = RF_Classifier(x_tr, y_tr, n_trees=4, n_features='sqrt', max_depth=3)
    preds, labels = [rf.predict(x_te[i, :]) for i in range(x_te.shape[0])], [y_te[i] for i in range(x_te.shape[0])]
    conf = get_metrics(labels, preds)
    pass


def classify_irrmapper(mode='multiclass'):
    """Classify irrigation with the IrrMapper data set"""
    _csv = 'irrmapper_training_data.csv'
    x_tr, x_te, y_tr, y_te = get_data(_csv, train_fraction=0.6, mode=mode, head=1000)
    rf = RF_Classifier(x_tr, y_tr, n_trees=10, n_features='sqrt', max_depth=10)
    preds, labels = [rf.predict(x_te[i, :]) for i in range(x_te.shape[0])], [y_te[i] for i in range(x_te.shape[0])]
    n_classes = np.unique(np.array(labels)).shape[0]
    conf = get_metrics(labels, preds, n_class=n_classes)
    print(conf)


if __name__ == '__main__':
    classify_irrmapper('binary')
# ========================= EOF ====================================================================
