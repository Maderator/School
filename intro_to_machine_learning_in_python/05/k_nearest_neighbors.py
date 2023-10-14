#!/usr/bin/env python3
import argparse
import os
import sys
import urllib.request

import numpy as np
import sklearn.metrics
import sklearn.model_selection
import sklearn.preprocessing
#import timeit
import sklearn.utils

class MNIST:
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
parser.add_argument("--k", default=1, type=int, help="K nearest neighbors to consider")
parser.add_argument("--p", default=2, type=int, help="Use L_p as distance metric")
parser.add_argument("--plot", default=False, const=True, nargs="?", type=str, help="Plot the predictions")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=1000, type=lambda x:int(x) if x.isdigit() else float(x), help="Test set size")
parser.add_argument("--train_size", default=1000, type=lambda x:int(x) if x.isdigit() else float(x), help="Test set size")
parser.add_argument("--weights", default="uniform", type=str, help="Weighting to use (uniform/inverse/softmax)")
# If you add more arguments, ReCodEx will keep them with your default values.

def main(args):
    # Load MNIST data, scale it to [0, 1] and split it to train and test
    mnist = MNIST()
    mnist.data = sklearn.preprocessing.MinMaxScaler().fit_transform(mnist.data)
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        mnist.data, mnist.target, stratify=mnist.target, train_size=args.train_size, test_size=args.test_size, random_state=args.seed)

    # TODO: Generate `test_predictions` with classes predicted for `test_data`.
    #
    # Find `args.k` nearest neighbors, choosing the ones with smallest train_data
    # indices in case of ties. Use the most frequent class (optionally weighted
    # by a given scheme described below) as prediction, again using the one with
    # smaller index when there are multiple classes with the same frequency.
    #
    # Use L_p norm for a given p (1, 2, 3) to measure distances.
    #
    # The weighting can be:
    # - "uniform": all nearest neighbors have the same weight
    # - "inverse": `1/distances` is used as weights
    # - "softmax": `softmax(-distances)` is uses as weights
    #
    # If you want to plot misclassified examples, you need to also fill `test_neighbors`
    # with indices of nearest neighbors; but it is not needed for passing in ReCodEx.


    def softmax(z):
        e_x = np.exp(z) # - np.max(z)) # avoiding the overflow errors with max so that the values in exponent are negative
        return e_x / np.array([np.sum(e_x, axis=1),]*e_x.shape[1]).T

    def compute_distance(data, x):
        return np.linalg.norm(x-data, ord=args.p, axis=1)
        #dif = x-data
        #if(args.p == 1):
        #    norm = np.sum(np.absolute(dif), axis=1)
        #elif(args.p == 2):
        #    norm = np.sqrt(np.sum(dif*dif, axis=1))
        #else:
        #    norm = np.cbrt(np.sum(np.absolute(dif*dif*dif),axis=1))
        #return norm

    def compute_distances(data, xs):
    # Compute distance of xs from data 
        dists = np.zeros((xs.shape[0], data.shape[0]))
        for i in range(xs.shape[0]):
            dists[i] = compute_distance(data, xs[i])
        return dists

    def find_k_nearest(data, classes, xs, k):
        dists =compute_distances(data, xs)
        k_nearest = np.zeros((dists.shape[0], k))
        kn_classes = np.zeros((dists.shape[0], k), dtype=np.int)
        dists_sorted = np.zeros((dists.shape[0], k), dtype=np.int)
        for i in range(dists.shape[0]):
            ds = np.argsort(dists[i])
            dists_sorted[i] = ds[:k]
            k_nearest[i] = dists[i][dists_sorted[i]]
            kn_classes[i] = classes[dists_sorted[i]]

        return [k_nearest, kn_classes, dists_sorted]

    #start = timeit.default_timer()

    [k_nearest, classes, test_neighbors] = find_k_nearest(train_data, train_target, test_data, args.k)

    test_predictions = []
    if(args.weights == "uniform"):
        for i in range(classes.shape[0]):
            test_predictions.append(np.argmax(np.bincount(classes[i])))
    elif (args.weights == "inverse"):
        w = 1/k_nearest
        classes_weighted_counts = sklearn.utils.extmath.weighted_mode(classes, w, axis=1)
        test_predictions = classes_weighted_counts[0].astype(int)
    elif (args.weights == "softmax"):
        w = softmax(-k_nearest)
        classes_weighted_counts = sklearn.utils.extmath.weighted_mode(classes, w, axis=1)
        test_predictions = classes_weighted_counts[0].astype(int)

    #stop = timeit.default_timer()

    #print('Time: ', stop - start)   
 
    accuracy = sklearn.metrics.accuracy_score(test_target, test_predictions)

    if args.plot:
        import matplotlib.pyplot as plt
        examples = [[] for _ in range(10)]
        for i in range(len(test_predictions)):
            if test_predictions[i] != test_target[i] and not examples[test_target[i]]:
                examples[test_target[i]] = [test_data[i], *train_data[test_neighbors[i]]]
        examples = [[img.reshape(28, 28) for img in example] for example in examples if example]
        examples = [[example[0]] + [np.zeros_like(example[0])] + example[1:] for example in examples]
        plt.imshow(np.concatenate([np.concatenate(example, axis=1) for example in examples], axis=0), cmap="gray")
        plt.gca().get_xaxis().set_visible(False)
        plt.gca().get_yaxis().set_visible(False)
        if args.plot is True: plt.show()
        else: plt.savefig(args.plot, transparent=True, bbox_inches="tight")

    return accuracy

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    accuracy = main(args)
    print("K-nn accuracy for {} nearest neighbors, L_{} metric, {} weights: {:.2f}%".format(
        args.k, args.p, args.weights, 100 * accuracy))
