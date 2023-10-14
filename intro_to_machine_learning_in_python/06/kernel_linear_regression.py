#!/usr/bin/env python3
import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=1, type=int, help="Batch size")
parser.add_argument("--data_size", default=50, type=int, help="Data size")
parser.add_argument("--kernel", default="rbf", type=str, help="Kernel type [poly|rbf]")
parser.add_argument("--kernel_degree", default=3, type=int, help="Degree for poly kernel")
parser.add_argument("--kernel_gamma", default=1.0, type=float, help="Gamma for poly and rbf kernel")
parser.add_argument("--iterations", default=200, type=int, help="Number of training iterations")
parser.add_argument("--l2", default=0.0, type=float, help="L2 regularization weight")
parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate")
parser.add_argument("--plot", default=False, const=True, nargs="?", type=str, help="Plot the predictions")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# If you add more arguments, ReCodEx will keep them with your default values.

def main(args):
    # Create a random generator with a given seed
    generator = np.random.RandomState(args.seed)

    # Generate an artifical regression dataset
    train_data = np.linspace(-1, 1, args.data_size)
    train_targets = np.sin(5 * train_data) + generator.normal(scale=0.25, size=args.data_size) + 1

    test_data = np.linspace(-1.2, 1.2, 2 * args.data_size)
    test_targets = np.sin(5 * test_data) + 1

    betas = np.zeros(args.data_size)

    # TODO: Perform `args.iterations` of SGD-like updates, but in dual formulation
    # using `betas` as weights of individual training examples.
    #
    # We assume the primary formulation of our model is
    #   y = phi(x)^T w + bias
    # and the loss in the primary problem is batched MSE with L2 regularization:
    #   L = sum_{i \in B} 1/|B| * [1/2 * (target_i - phi(x_i)^T w - bias)^2] + 1/2 * args.l2 * w^2
    #                             |     TODO this is mean square error      |
    #
    # For `bias`, use explicitly the average of the training targets, and do
    # not update it futher during training.
    #
    # Instead of using feature map `phi` directly, we use a given kernel computing
    #   K(x, y) = phi(x)^T phi(y)
    # We consider the following `args.kernel`s:
    # - "poly": K(x, y; degree, gamma) = (gamma * x^T y + 1) ^ degree
    # - "rbf": K(x, y; gamma) = exp^{- gamma * ||x - y||^2}
    #
    # After each iteration, compute RMSE both on training and testing data.
    def compute_kernel_val(x, y):
        #if np.isscalar(x):
        #    x=np.array(x)
        #if np.isscalar(y):
        #    y=np.array(y)
        if args.kernel == "poly":
            return (args.kernel_gamma * x * y + 1) ** args.kernel_degree # TODO make multidimensional
        elif args.kernel == "rbf":
            dif = x-y
            #norm = np.sqrt(dif*dif)
            #norm = np.linalg.norm(x-y, ord=2)
            return np.exp(-args.kernel_gamma *(dif*dif))
    
    
    train_rmses, test_rmses = [], []
    bias = np.mean(train_targets)

    td_shape = train_data.shape[0]
    kernel = np.zeros((td_shape, td_shape))
    for i in range(td_shape):
        for j in range(td_shape):
            kernel[i,j] = compute_kernel_val(train_data[i], train_data[j])

    for iteration in range(args.iterations):
        permutation = generator.permutation(train_data.shape[0])

        # TODO: Process the data in the order of `permutation`, performing
        # batched updates to the `betas`. You can assume that `args.batch_size`
        # exactly divides `train_data.shape[0]`.
        gradient_components = 0
        idxs = []
        new_betas = []
        for i in permutation:
            idxs.append(i)
            new_beta = betas[i] + args.learning_rate/args.batch_size * (train_targets[i] - (betas@kernel[i] + bias))
            new_betas.append(new_beta)

            #grad += -1/args.batch_size * (train_targets[i] - betas@kernel[i] - bias)*kernel[i] - args.l2*betas
            #new_betas.append(betas[i] + ) 
            gradient_components += 1
            if gradient_components == args.batch_size:
                regularization = args.learning_rate * args.l2 * betas * (train_data-bias)
                for i,idx in enumerate(idxs):
                    betas[idx] = new_betas[i]
                betas = betas + regularization
                regularization = np.zeros(betas.shape[0])
                new_betas = []
                idxs = []
                gradient_components = 0
        assert gradient_components == 0
        # TODO: Append RMSE on training and testing data to `train_rmses` and
        # `test_rmses` after the iteration.

        train_preds = kernel @ betas + bias# + bias/len(betas)
        train_rms = sklearn.metrics.mean_squared_error(train_targets, train_preds, squared=False)
        train_rmses.append(train_rms)

        test_preds = []
        for d in test_data:
            test_preds.append(compute_kernel_val(train_data, d) @ betas + bias)
        test_rms = sklearn.metrics.mean_squared_error(test_targets, test_preds, squared=False)
        test_rmses.append(test_rms)

        if (iteration + 1) % 10 == 0:
            print("Iteration {}, train RMSE {:.2f}, test RMSE {:.2f}".format(
                iteration + 1, train_rmses[-1], test_rmses[-1]))

    if args.plot:
        # If you want the plotting to work (not required for ReCodEx), compute the `test_predictions`.
        test_predictions = test_preds

        plt.plot(train_data, train_targets, "bo", label="Train targets")
        plt.plot(test_data, test_targets, "ro", label="Test targets")
        plt.plot(test_data, test_predictions, "g-", label="Predictions")
        plt.legend()
        if args.plot is True: plt.show()
        else: plt.savefig(args.plot, transparent=True, bbox_inches="tight")

    return train_rmses, test_rmses

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
