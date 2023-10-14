#!/usr/bin/env python3
import argparse
import sys

import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
from scipy.special import expit
import math

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=20, type=int, help="Batch size")
parser.add_argument("--classes", default=10, type=int, help="Number of classes to use")
parser.add_argument("--iterations", default=10, type=int, help="Number of iterations over the data")
parser.add_argument("--learning_rate", default=0.005, type=float, help="Learning rate")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=797, type=lambda x:int(x) if x.isdigit() else float(x), help="Test set size")
# If you add more arguments, ReCodEx will keep them with your default values.

def softmax(z):
    m = np.max(z)
    e_x = np.exp(z - m) # avoiding the overflow errors with max so that the values in exponent are negative
    e_x_sum = np.sum(e_x)
    div = e_x / e_x_sum
    return div

def second_softmax(weights, x, i):
    m = np.max((weights.T @ x))
    return np.exp((weights.T @ x)[i]-m) / np.sum(np.exp(weights.T @ x - m))

def main(args):
    # Create a random generator with a given seed
    generator = np.random.RandomState(args.seed)

    # Use the digits dataset
    data, target = sklearn.datasets.load_digits(n_class=args.classes, return_X_y=True)

    # Append a constant feature with value 1 to the end of every input data
    data = np.pad(data, ((0, 0), (0, 1)), constant_values=1)

    # Split the dataset into a train set and a test set.
    # Use `sklearn.model_selection.train_test_split` method call, passing
    # arguments `test_size=args.test_size, random_state=args.seed`.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, stratify=target, test_size=args.test_size, random_state=args.seed)

    # Generate initial linear regression weights
    weights = generator.uniform(size=[train_data.shape[1], args.classes], low=-0.1, high=0.1)
    train_target_prob = np.zeros((train_target.shape[0], args.classes))
    train_target_prob[[i for i in range(train_target.shape[0])], train_target] = 1

    for iteration in range(args.iterations):
        permutation = generator.permutation(train_data.shape[0])

        # TODO: Process the data in the order of `permutation`.
        # For every `args.batch_size`, average their gradient, and update the weights.
        # You can assume that `args.batch_size` exactly divides `train_data.shape[0]`.
        #
        # Note that you need to be careful when computing softmax, because the exponentiation
        # in softmax can easily overflow. To avoid it, you can use the fact that
        # softmax(z) = softmax(z + any_constant) and compute softmax(z) = softmax(z - maximum_of_z).
        # That way we only exponentiate values which are non-positive, and overflow does not occur.

        gradient, gradient_components = 0, 0
        for i in permutation:
            cond_prob = softmax(weights.T @ train_data[i])
            dif = (cond_prob - train_target_prob[i])
            gradient += np.outer(dif, train_data[i])
            #gradient += np.array([dif,]*train_data.shape[1]).T * train_data[i]
            gradient_components += 1
            if gradient_components == args.batch_size:
                weights -= args.learning_rate * gradient.T / gradient_components
                gradient, gradient_components = 0, 0
        assert gradient_components == 0
        
        # TODO: After the SGD iteration, measure the average loss and accuracy for both the
        # train test and the test set. The loss is the average MLE loss (i.e., the
        # negative log likelihood, or crossentropy loss, or KL loss) per example.
        train_d_c_probs = train_data @ weights # train data class probabilities
        train_highest_prob = np.argmax(train_d_c_probs, axis=1) # class with highest probability
        train_accuracy = np.sum(train_highest_prob == train_target) / len(train_target)
        train_loss = 0
        for i in range(train_target.shape[0]):
            train_loss += train_target_prob[i] * np.log(softmax(train_d_c_probs[i]))
        train_loss = - sum(train_loss) / train_target.shape[0]

        test_d_c_probs = test_data @ weights
        test_highest_prob = np.argmax(test_d_c_probs, axis=1)
        test_accuracy = sum(test_highest_prob == test_target) / len(test_target)
        test_loss = 0
        test_target_prob = np.zeros((test_target.shape[0], args.classes))
        test_target_prob[[i for i in range(test_target.shape[0])], test_target] = 1
        for i in range(test_target.shape[0]):
            test_loss += test_target_prob[i] * np.log(softmax(test_d_c_probs[i]))
        test_loss = - sum(test_loss) / test_target.shape[0]

        print("After iteration {}: train loss {:.4f} acc {:.1f}%, test loss {:.4f} acc {:.1f}%".format(
            iteration + 1, train_loss, 100 * train_accuracy, test_loss, 100 * test_accuracy))

    return weights

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    weights = main(args)
    print("Learned weights:", *(" ".join([" "] + ["{:.2f}".format(w) for w in row[:10]] + ["..."]) for row in weights.T), sep="\n")
