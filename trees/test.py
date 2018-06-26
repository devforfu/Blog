from os.path import join, dirname

import pytest
import graphviz
import numpy as np

from plots import create_graph
from ensemble import RandomForestClassifier
from decision_tree import learn_tree, predict_tree
from utils import read_csv, encode_labels, train_test_split


PALETTE = [
    [167, 98, 228],
    [74, 144, 226],
    [244, 166, 35]
]


def test_creating_a_single_decision_tree(request, wine_dataset):
    feature_names, X, y = wine_dataset
    y, encoder, class_names = encode_labels(y.astype(int))
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    tree = learn_tree(X_train, y_train, max_depth=5)
    acc = compute_accuracy(tree, X_test, y_test)
    print(f'Test set accuracy: {acc:2.2%}')

    dot_data = create_graph(tree, feature_names, class_names, palette=PALETTE)
    graph = graphviz.Source(dot_data, format='png')
    graph.render(request.node.name)


@pytest.mark.parametrize('trial', [0, 1, 2])
def test_creating_set_of_trees(trial, wine_dataset):
    np.random.seed(trial)

    feature_names, X, y = wine_dataset
    y, encoder, class_names = encode_labels(y.astype(int))
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    tree = learn_tree(X_train, y_train, max_depth=2)
    acc = compute_accuracy(tree, X_test, y_test)
    dot_data = create_graph(
        tree, feature_names, class_names,
        palette=PALETTE, title=f'Tree accuracy: {acc:2.2%}')
    graph = graphviz.Source(dot_data, format='png')
    graph.render('tree_%d' % trial)


def test_creating_an_ensemble_of_trees(wine_dataset):
    feature_names, X, y = wine_dataset
    y, encoder, class_names = encode_labels(y.astype(int))
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    random_forest = RandomForestClassifier(
        tree_funcs=(learn_tree, predict_tree),
        n_trees=10, max_depth=1, min_leaf_size=5,
        min_split_size=10, feature_subset_size='sqrt')
    preds = random_forest.fit(X_train, y_train).predict(X_test)
    acc = np.mean(y_test == preds)

    print(f'Test set accuracy: {acc:2.2%}')


def compute_accuracy(tree, X_test, y_test):
    preds = predict_tree(tree, X_test)
    acc = np.mean(y_test == preds)
    return acc


@pytest.fixture
def wine_dataset():
    data_path = join(dirname(__file__), 'datasets', 'wine.csv')
    feature_names, X, y = read_csv(data_path)
    return feature_names, X, y
