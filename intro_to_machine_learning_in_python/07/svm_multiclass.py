#!/usr/bin/env python3
import argparse

import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--C", default=1, type=float, help="Inverse regularization strength")
parser.add_argument("--classes", default=10, type=int, help="Number of classes")
parser.add_argument("--kernel", default="poly", type=str, help="Kernel type [poly|rbf]")
parser.add_argument("--kernel_degree", default=1, type=int, help="Degree for poly kernel")
parser.add_argument("--kernel_gamma", default=1.0, type=float, help="Gamma for poly and rbf kernel")
parser.add_argument("--max_iterations", default=1000, type=int, help="Maximum number of iterations to perform")
parser.add_argument("--max_passes_without_as_changing", default=10, type=int, help="Number of passes without changes to stop after")
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
        preds += a[i]*t[i] *kernels[i_cur, i] #+ bias
    return preds + bias

def predict_classifs(args, kernels, xs, a, t, bias):
    preds = []
    for i in range(len(xs)):
        pred_w = predict_classif(args, kernels, i, a, t, bias, xs)
        if pred_w > 0:
            preds.append(1)
        else:
            preds.append(-1)
    return preds

def pred(args, x, a, t, bias, xs):
    preds = 0
    for i in range(len(xs)):
        preds += a[i]*t[i] *kernel(args, xs[i], x) #+ bias # TODO bias to return 
    return preds + bias

def pred_no_target(args, x, a, bias, xs):
    preds = 0
    for i in range(len(xs)):
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

def pred_binary(args, x, supv, supv_w, b):
    pred_w = 0
    for i in range(len(supv)):
        pred_w += supv_w[i] *kernel(args, supv[i], x)
    pred_w += b
    
    if pred_w > 0:
        return 1
    else:
        return -1

def pred_multiclass_sample(args, x, supvs, supvs_weights, biasses):
    k = len(supvs)+1
    votes = np.zeros(k)
    for i in range(k-1):
        for j in range(i+1,k):
            shift = i+1
            winner = pred_binary(args, x, supvs[i][j-shift], supvs_weights[i][j-shift], biasses[i][j-shift])
            if winner == 1:
                votes[i] += 1
            else:
                votes[j] += 1
    return np.argmax(votes)

def multiclass_pred_classifs_no_kernel(args, supvs, supvs_weights, biasses, test_data):
    preds = []
    for d in range(len(test_data)):
        preds.append(pred_multiclass_sample(args, test_data[d], supvs, supvs_weights, biasses))
    return preds

def compute_E(args, kernels, i, a, t, bias, xs):
    yxi = predict_classif(args, kernels, i, a, t, bias, xs)
    return yxi - t[i]

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

def get_two_class_data_from_n_class_data(c1, c2, data, target):
    two_classes = (target==c1) | (target==c2)
    t = target[two_classes]
    ti = t == c1
    tj = t == c2
    t[ti] = 1
    t[tj] = -1
    d = data[two_classes]
    return d, t


def main(args):
    # Use the digits dataset.
    data, target = sklearn.datasets.load_digits(n_class=args.classes, return_X_y=True)
    data = sklearn.preprocessing.MinMaxScaler().fit_transform(data)

    # Split the dataset into a train set and a test set.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, test_size=args.test_size, random_state=args.seed)
    
    k = np.max(train_target)+1

    # TODO: Using One-vs-One scheme, train (K \binom 2) classifiers, one for every
    # pair of classes $i < j$, using the `smo` method.
    supvs = []
    supvs_weights = []
    biasses = []
    for i in range(k-1):
        supvs.append([])
        supvs_weights.append([])
        biasses.append([])
        for j in range(i+1,k):
            print("Training classes {} and {}".format(i,j))
            #print(supvs[0][0], supvs_weights[0][0], biasses[0][0])
            trd, trt = get_two_class_data_from_n_class_data(i,j,train_data, train_target)
            tstd, tstt = get_two_class_data_from_n_class_data(i,j, test_data, test_target)
            cur_supvs, cur_supvs_weights, cur_biasses, tr_acc, tst_acc = smo(args, trd,trt,tstd,tstt)
            supvs[i].append(cur_supvs)
            supvs_weights[i].append(cur_supvs_weights)
            biasses[i].append(cur_biasses)


    # When training a classifier for classes $i < j$:
    # - keep only the training data of these classes, in the same order
    #   as in the input dataset;
    # - use targets 1 for the class $i$ and -1 for the class $j$.

    # TODO: Classify the test set by majority voting of all the trained classifiers,
    # using the lowest class index in the case of ties.
    #
    # Finally, compute the test set prediction accuracy.

    test_pred = multiclass_pred_classifs_no_kernel(args, supvs, supvs_weights, biasses, test_data)
    test_accuracy = sum(test_pred==test_target)/len(test_target)

    return test_accuracy

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    accuracy = main(args)
    print("Test set accuracy: {:.2f}%".format(100 * accuracy))
