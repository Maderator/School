#!/usr/bin/env python3
import argparse

import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection

parser = argparse.ArgumentParser()
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.1, type=lambda x:int(x) if x.isdigit() else float(x), help="Test set size")

def main(args):
    # Load Boston housing dataset
    dataset = sklearn.datasets.load_boston()

    # The input data are in dataset.data, targets are in dataset.target.

    # If you want to learn about the dataset, uncomment the following line.
    #print(dataset.DESCR)
    #print(dataset.data.shape)
    #print(dataset.target.shape)

    # TODO: Append a new feature to all input data, with value "1"
    one_feature = np.ones((dataset.data.shape[0],1))
    dataset.data = np.concatenate((dataset.data, one_feature), axis=1)

    # TODO: Split the dataset into a train set and a test set.
    # Use `sklearn.model_selection.train_test_split` method call, passing
    # arguments `test_size=args.test_size, random_state=args.seed`.
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(dataset.data, dataset.target, test_size=args.test_size, random_state=args.seed)

    # TODO: Solve the linear regression using the algorithm from the lecture,
    # explicitly computing the matrix inverse (using `np.linalg.inv`).
    # (X^tX)^-1X^tt) kde t je true value.
    weights = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train

    # TODO: Predict target values on the test set
    # f(x,w) = X*w
    y_pred = X_test @ weights

    # TODO: Compute root mean square error on the test set predictions
    #sklearn.metrics.mean_squared_error() #měla by mít argument aby se sama odmocnila

    rmse = sklearn.metrics.mean_squared_error(y_test, y_pred, squared=False)

    return rmse

if __name__ == "__main__":
    args = parser.parse_args()
    rmse = main(args)
    print("{:.2f}".format(rmse))