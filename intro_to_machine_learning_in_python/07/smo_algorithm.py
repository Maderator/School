#!/usr/bin/env python3
import argparse

import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--C", default=1, type=float, help="Inverse regularization strength")
parser.add_argument("--data_size", default=200, type=int, help="Data size")
parser.add_argument("--kernel", default="poly", type=str, help="Kernel type [poly|rbf]")
parser.add_argument("--kernel_degree", default=1, type=int, help="Degree for poly kernel")
parser.add_argument("--kernel_gamma", default=1.0, type=float, help="Gamma for poly and rbf kernel")
parser.add_argument("--max_iterations", default=1000, type=int, help="Maximum number of iterations to perform")
parser.add_argument("--max_passes_without_as_changing", default=10, type=int, help="Number of passes without changes to stop after")
parser.add_argument("--plot", default=False, const=True, nargs="?", type=str, help="Plot the predictions")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.5, type=lambda x:int(x) if x.isdigit() else float(x), help="Test set size")
parser.add_argument("--tolerance", default=1e-4, type=float, help="Default tolerance for KKT conditions")
# If you add more arguments, ReCodEx will keep them with your default values.

def kernel(args, x, y):
    # TODO: As in `kernel_linear_regression`, We consider the following `args.kernel`s:
    # - "poly": K(x, y; degree, gamma) = (gamma * x^T y + 1) ^ degree
    # - "rbf": K(x, y; gamma) = exp^{- gamma * ||x - y||^2}
    if args.kernel == "poly":
        return (args.kernel_gamma * (x.T @ y) + 1) ** args.kernel_degree # TODO make multidimensional
    elif args.kernel == "rbf":
        dif = x-y
        norm = np.linalg.norm(dif, ord=2)
        return np.exp(-args.kernel_gamma *norm*norm)


def predict_classif(args, kernels, i_cur, a, t, bias, xs):
    preds = 0
    for i in range(len(xs)):
        #print("kernel ", kernel(args, x, xs[i]))
        preds += a[i]*t[i] *kernels[i_cur, i] #+ bias
    return preds + bias

def predict_classifs(args, kernels, xs, a, t, bias):
    preds = []
    for i in range(len(xs)):
        pred_w = predict_classif(args, kernels, i, a, t, bias, xs)
     #   print(pred_w, end=" ")
        if pred_w > 0:
            preds.append(1)
        else:
            preds.append(-1)
    #print()
    return preds

def pred(args, x, a, t, bias, xs):
    preds = 0
    for i in range(len(xs)):
        #print("kernel ", kernel(args, x, xs[i]))
        preds += a[i]*t[i] *kernel(args, xs[i], x) #+ bias # TODO bias to return 
    return preds + bias

def pred_no_target(args, x, a, bias, xs):
    preds = 0
    for i in range(len(xs)):
        #print("kernel ", kernel(args, x, xs[i]))
        preds += a[i] *kernel(args, xs[i], x) #+ bias # TODO bias to return 
    return preds + bias

def pred_classifs_no_kernel(args, data, a, t, bias, sup_vects):
    preds = []
    for i in range(len(data)):
        pred_w = pred(args, data[i], a, t, bias, sup_vects)
        if pred_w > 0:
            preds.append(1)
        else:
            preds.append(-1)
    return preds

def compute_E(args, kernels, i, a, t, bias, xs):
    yxi = predict_classif(args, kernels, i, a, t, bias, xs)
    #print(yxi)
    return yxi - t[i]

# We implement the SMO algorithm as a separate method, so we can use
# it in the svm_multiclass assignment too.
def smo(args, train_data, train_target, test_data, test_target):
    t = train_target
    x = train_data
    # Create initial weights
    a, b = np.zeros(len(x)), 0
    generator = np.random.RandomState(args.seed)

    td_shape = x.shape[0]
    kernels = np.zeros((td_shape, td_shape))
    for i in range(td_shape):
        for j in range(td_shape):
            kernels[i,j] = kernel(args, x[i], x[j])

    passes_without_as_changing = 0
    train_accs, test_accs = [], []
    for _ in range(args.max_iterations):
        as_changed = 0
        # Iterate through the data
        for i, j in enumerate(generator.randint(len(a) - 1, size=len(a))):
            # We want j != i, so we "skip" over the value of i
            j = j + (j >= i)

            # TODO: Check that a[i] fulfuls the KKT condition, using `args.tolerance` during comparisons.
            tol = args.tolerance
            c = args.C
            Ei =  compute_E(args, kernels, i, a, t, b, x)
            #print(Ei)
            cond1 = a[i] < c - tol and t[i]*Ei < -tol
            cond2 = a[i] > tol and t[i]*Ei > tol
            
            if cond1 or cond2:
            # If the conditions, do not hold, then
            # - compute the updated unclipped a_j^new.
                Ej =  compute_E(args, kernels, j, a, t, b, x)
                first_deriv = t[j] * (Ei - Ej)
                second_deriv = 2*kernels[i,j] - kernels[i,i] - kernels[j, j]

            #   If the second derivative of the loss with respect to a[j]
            #   is > -`args.tolerance`, do not update a[j] and continue
            #   with next i.
                if second_deriv > - tol:
                    continue
                aj_new = a[j] - first_deriv / second_deriv
            # - clip the a_j^new to suitable [L, H].
                if t[i] == t[j]:
                    low_clip = max(0, a[i] + a[j] - c)
                    high_clip = min(c, a[i] + a[j])
                else:
                    low_clip = max(0, a[j] - a[i])
                    high_clip = min(c, c + a[j] - a[i])
                aj_new = max(low_clip, min(aj_new, high_clip)) # clip the aj_new
            #   If the clipped updated a_j^new differs from the original a[j]
            #   by less than `args.tolerance`, do not update a[j] and continue
            #   with next i.
                if abs(aj_new - a[j]) < tol:
                    continue
            # - update a[j] to a_j^new, and compute the updated a[i] and b.
                ai_new = a[i] - t[i] * t[j] * (aj_new - a[j])
                bj_new = b - Ej - t[i]*(ai_new - a[i]) * kernels[i,j] - t[j] * (aj_new-a[j]) * kernels[j,j]
                bi_new = b - Ei - t[i]*(ai_new - a[i]) * kernels[i,i] - t[j] * (aj_new-a[j]) * kernels[j,i]
            #   During the update of b, compare the a[i] and a[j] to zero by
            #   `> args.tolerance` and to C using `< args.C - args.tolerance`.
                if ai_new < c - tol and ai_new > tol:
                    b = bi_new
                elif aj_new < c - tol and aj_new > tol:
                    b = bj_new
                else:
                    b = (bi_new + bj_new) / 2
                a[i] = ai_new 
                a[j] = aj_new
            # - increase `as_changed`
                as_changed += 1

        # TODO: After each iteration, measure the accuracy for both the
        # train test and the test set and append it to `train_accs` and `test_accs`.
        train_pred = predict_classifs(args, kernels, train_data, a, train_target, b)
        train_accs.append(sum(train_pred==train_target)/len(train_target))
        test_pred = pred_classifs_no_kernel(args, test_data, a, train_target, b, train_data)
        test_accs.append(sum(test_pred==test_target)/len(test_target))

        # Stop training if max_passes_without_as_changing passes were reached
        passes_without_as_changing = 0 if as_changed else passes_without_as_changing + 1
        if passes_without_as_changing >= args.max_passes_without_as_changing:
            break

        if len(train_accs) % 100 == 0 and len(train_accs) < args.max_iterations:
            print("Iteration {}, train acc {:.1f}%, test acc {:.1f}%".format(
                len(train_accs), 100 * train_accs[-1], 100 * test_accs[-1]))

    print("Training finished after iteration {}, train acc {:.1f}%, test acc {:.1f}%".format(
        len(train_accs), 100 * train_accs[-1], 100 * test_accs[-1]))

    # TODO: Create an array of support vectors (in the same order in which they appeared
    # in the training data; to avoid rounding errors, consider a training example
    # a support vector only if a_i > `args.tolerance`) and their weights (a_i * t_i).
    support_vectors = []
    support_vector_weights = []
    for i in range(len(a)):
        if a[i] > args.tolerance:
            support_vectors.append(train_data[i])
            support_vector_weights.append(a[i] * train_target[i])
    support_vectors = np.array(support_vectors)
    support_vector_weights = np.array(support_vector_weights)

    return support_vectors, support_vector_weights, b, train_accs, test_accs

def main(args):
    # Generate an artifical regression dataset, with +-1 as targets
    data, target = sklearn.datasets.make_classification(
        n_samples=args.data_size, n_features=2, n_informative=2, n_redundant=0, random_state=args.seed)
    target = 2 * target - 1

    # Split the dataset into a train set and a test set.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, test_size=args.test_size, random_state=args.seed)

    # Run the SMO algorithm
    support_vectors, support_vector_weights, bias, train_accs, test_accs = smo(
        args, train_data, train_target, test_data, test_target)

    if args.plot:
        import matplotlib.pyplot as plt
        def plot(predict, support_vectors):
            xs = np.linspace(np.min(data[:, 0]), np.max(data[:, 0]), 50)
            ys = np.linspace(np.min(data[:, 1]), np.max(data[:, 1]), 50)
            predictions = [[predict(np.array([x, y])) for x in xs] for y in ys]
            test_mismatch = np.sign([predict(x) for x in test_data]) != test_target
            plt.figure()
            plt.contourf(xs, ys, predictions, levels=0, cmap=plt.cm.RdBu)
            plt.contour(xs, ys, predictions, levels=[-1, 0, 1], colors="k", zorder=1)
            plt.scatter(train_data[:, 0], train_data[:, 1], c=train_target, marker="o", label="Train", cmap=plt.cm.RdBu, zorder=2)
            plt.scatter(support_vectors[:, 0], support_vectors[:, 1], marker="o", s=90, label="Support Vectors", c="#00dd00")
            plt.scatter(test_data[:, 0], test_data[:, 1], c=test_target, marker="*", label="Test", cmap=plt.cm.RdBu, zorder=2)
            plt.scatter(test_data[test_mismatch, 0], test_data[test_mismatch, 1], marker="*", s=130, label="Test Errors", c="#ffff00")
            plt.legend(loc="upper center", ncol=4)

        # If you want plotting to work (not required for ReCodEx), you need to
        # define `predict_function` computing SVM prediction `y(x)` for the given x.
        predict_function = lambda x: pred_no_target(args, x, support_vector_weights, bias, support_vectors)

        plot(predict_function, support_vectors)
        if args.plot is True: plt.show()
        else: plt.savefig(args.plot, transparent=True, bbox_inches="tight")

    return support_vectors, support_vector_weights, bias, train_accs, test_accs

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
