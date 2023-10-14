#!/usr/bin/env python3
import argparse
import os
import sys
import urllib.request

import math
import numpy as np
import sklearn.base
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
import sklearn.pipeline
import sklearn.kernel_approximation
import sklearn.linear_model
import sklearn.decomposition
import sklearn.svm
import sklearn.neural_network

class MNIST:
    """MNIST Dataset.

    The train set contains 60000 images of handwritten digits. The data
    contain 28*28=784 values in range 0-255, the targets are numbers 0-9.
    """
    def __init__(self,
                 name="mnist.train.npz",
                 data_size=None,
                 url="https://ufal.mff.cuni.cz/~straka/courses/npfl129/2021/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve(url + name, filename=name)

        # Load the dataset, i.e., `data` and optionally `target`.
        dataset = np.load(name)
        for key, value in dataset.items():
            setattr(self, key, value[:data_size])
        self.data = self.data.reshape([-1, 28*28]).astype(np.float)

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--gamma", default=0.022, type=float, help="RBF gamma")
parser.add_argument("--max_iter", default=100, type=int, help="Maximum iterations for LR")
parser.add_argument("--nystroem", default=0, type=int, help="Use Nystroem approximation")
parser.add_argument("--original", default=False, action="store_true", help="Use original data")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--rff", default=0, type=int, help="Use RFFs")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--svm", default=False, action="store_true", help="Use SVM instead of LR")
parser.add_argument("--test_size", default=0.5, type=lambda x:int(x) if x.isdigit() else float(x), help="Test set size")
# If you add more arguments, ReCodEx will keep them with your default values.

class RFFsTransformer(sklearn.base.TransformerMixin):
    def __init__(self, n_components, gamma, seed):
        self._n_components = n_components
        self._gamma = gamma
        self._seed = seed

    def fit(self, X, y=None):
        generator = np.random.RandomState(self._seed)
        # TODO: Generate suitable `w` and `b`.
        # To obtain deterministic results, generate
        # - `w` first, using a single `generator.normal` call with
        #   output shape `(input_features, self._n_components)`
        # - `b` second, using a single `generator.uniform` call
        #   with output shape `(self._n_components,)`
        self.w = np.sqrt(2*self._gamma) * generator.normal(size=(X.shape[1], self._n_components))
        self.b = generator.uniform(0, 2*math.pi, size=self._n_components)

        return self

    def transform(self, X):
        # TODO: Transform the given `X` using precomputed `w` and `b`.
        return np.sqrt(2 / self._n_components) * np.cos(X @ self.w  + self.b)

class NystroemTransformer(sklearn.base.TransformerMixin):
    def __init__(self, n_components, gamma, seed):
        self._n_components = n_components
        self._gamma = gamma
        self._seed = seed

    def _rbf_kernel(self, X, Z):
        # TODO: Compute the RBF kernel with `args._gamma` for
        # given two sets of examples.
        #
        # A reasonably efficient implementation should probably compute the
        # kernel line-by-line, computing K(X_i, Z) using a singe `np.linalg.norm`
        # call, and then concatenate the results using `np.stack`.
        rows = []
        for i in range(X.shape[0]):
            dif = X[i,:]-Z
            norm = np.linalg.norm(dif, ord=2, axis=1)
            e = np.exp(-self._gamma *norm*norm)
            rows.append(e)

        kern = np.stack(rows)

        return kern

    def fit(self, X, y=None):
        generator = np.random.RandomState(self._seed)

        # TODO: Choose a random subset of examples, utilizing indices
        #   indices = generator.choice(X.shape[0], size=self._n_components, replace=False)
        #
        # Then, compute K as the RBF kernel of the chosen examples and
        # V as K^{-1/2} -- use `np.linalg.svd(K, hermitian=True)` to compute
        # the SVD (equal to eigenvalue decomposition for real symmetric matrices).
        # Add 1e-12 to the diagonal matrix returned by SVD before computing
        # the inverse of the square root.
        indices = generator.choice(X.shape[0], size=self._n_components, replace=False)
        self.chosen = X[indices]
        K = self._rbf_kernel(self.chosen, self.chosen)
        u,s,vh = np.linalg.svd(K, hermitian=True)
        d = np.diag((1/np.sqrt(s + 1e-12)))
        self.V = u  @ d @ u.T

        return self

    def transform(self, X):
        # TODO: Compute the RBF kernel of `X` and the chosen training examples
        # and then process it using the precomputed `V`.
        K = self._rbf_kernel(X,self.chosen)
        return K @ self.V

def main(args):
    # Use the digits dataset.
    dataset = MNIST(data_size=5000)

    # Split the dataset into a train set and a test set.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        dataset.data, dataset.target, test_size=args.test_size, random_state=args.seed)

    features = []
    if args.original:
        features.append(("original", sklearn.preprocessing.FunctionTransformer()))
    if args.rff:
        features.append(("rff", RFFsTransformer(args.rff, args.gamma, args.seed)))
    if args.nystroem:
        features.append(("nystroem", NystroemTransformer(args.nystroem, args.gamma, args.seed)))

    if args.svm:
        classifier = sklearn.svm.SVC()
    else:
        classifier = sklearn.linear_model.LogisticRegression(solver="saga", penalty="none", max_iter=args.max_iter, random_state=args.seed)

    pipeline = sklearn.pipeline.Pipeline([
        ("scaling", sklearn.preprocessing.MinMaxScaler()),
        ("features", sklearn.pipeline.FeatureUnion(features)),
        ("classifier", classifier),
    ])
    pipeline.fit(train_data, train_target)

    test_accuracy = sklearn.metrics.accuracy_score(test_target, pipeline.predict(test_data))
    return test_accuracy

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    accuracy = main(args)
    print("Test set accuracy: {:.2f}%".format(100 * accuracy))
