#!/usr/bin/env python3
import argparse
import lzma
import pickle
import os
import urllib.request
import sys
import timeit

from sklearn.neural_network import MLPClassifier

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class Dataset:
    """MNIST Dataset.

    The train set contains 60000 images of handwritten digits. The data
    contain 28*28=784 values in range 0-255, the targets are numbers 0-9.
    """
    def __init__(self,
                 name="mnist.train.npz",
                 url="https://ufal.mff.cuni.cz/~straka/courses/npfl129/2021/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve(url + name, filename=name)

        # Load the dataset, i.e., `data` and optionally `target`.
        dataset = np.load(name)
        for key, value in dataset.items():
            setattr(self, key, value)
        self.data = self.data.reshape([-1, 28*28]).astype(np.float)


parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Run prediction on given data")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="mnist_competition_adam2.model", type=str, help="Model path")

def main(args):
    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset()
        data = train.data
        target = train.target

        # TODO: Train a model on the given dataset and store it in `model`.

        clf = MLPClassifier(solver='adam', tol=1e-7, alpha=0.0001, epsilon=1e-08, max_iter=2000, n_iter_no_change=100, hidden_layer_sizes=(392,196,98,49,24,), random_state=1, verbose=True)
 
        #with lzma.open(args.model_path, "rb") as model_file:
        #    model = pickle.load(model_file)
        
        #print(model.get_params())
        #model.set_params(warm_start=True)
        #print(model.get_params())
        
        start = timeit.default_timer()

        model = clf.fit(train.data/255.0, train.target)

        stop = timeit.default_timer()

        print('Time: ', stop - start)   

        # If you trained one or more MLPs, you can use the following code
        # to compress it significantly (approximately 12 times). The snippet
        # assumes the trained MLPClassifier is in `mlp` variable.
        # mlp._optimizer = None
        # for i in range(len(mlp.coefs_)): mlp.coefs_[i] = mlp.coefs_[i].astype(np.float16)
        # for i in range(len(mlp.intercepts_)): mlp.intercepts_[i] = mlp.intercepts_[i].astype(np.float16)

        # Serialize the model.
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(model, model_file)

    else:
        # Use the model and return test set predictions, as either a Python list or a NumPy array.
        test = Dataset(args.predict)

        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)

        # TODO: Generate `predictions` with the test set predictions.
        predictions = model.predict(test.data)

        return predictions


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
