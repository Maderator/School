#!/usr/bin/env python3
import argparse

import numpy as np
import scipy.stats

import sklearn.datasets
import sklearn.model_selection

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--alpha", default=0.1, type=float, help="Smoothing parameter for Bernoulli and Multinomial NB")
parser.add_argument("--naive_bayes_type", default="bernoulli", type=str, help="NB type to use")
parser.add_argument("--classes", default=10, type=int, help="Number of classes")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.5, type=lambda x:int(x) if x.isdigit() else float(x), help="Test set size")
# If you add more arguments, ReCodEx will keep them with your default values.

def main(args):
    # Use the digits dataset.
    data, target = sklearn.datasets.load_digits(n_class=args.classes, return_X_y=True)

    # Split the dataset into a train set and a test set.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, test_size=args.test_size, random_state=args.seed)

    # TODO: Fit the naive Bayes classifier on the train data.
    #
    # The `args.naive_bayes_type` can be one of:
    # - "gaussian": Fit Gaussian NB, by estimating mean and variance of the input
    #   features. For variance estimation use
    #     1/N * \sum_x (x - mean)^2
    #   and additionally increase all estimated variances by `args.alpha`.
    #
    #   During prediction, compute probability density function of a Gaussian
    #   distribution using `scipy.stats.norm`, which offers `pdf` and `logpdf`
    #   methods, among others.
    #
    # - "multinomial": Multinomial NB with smoothing factor `args.alpha`.
    #
    # - "bernoulli": Bernoulli NB with smoothing factor `args.alpha`.
    #   Do not forget that Bernoulli NB works with binary data, so consider
    #   all non-zero features as ones during both estimation and prediction.
    test_accuracy = None

    classes = max(train_target)+1
    class_probs = []
    for c in range(classes):
        class_probs.append(len(train_data[train_target == c])/len(train_data))

    if args.naive_bayes_type == "gaussian":
        means = []
        variances = []
        for i in range(classes):
            data = train_data[train_target == i]
            N = data.shape[0]
            means.append(1/N * np.sum(data, axis=0))
            variances.append(1/N * np.sum((data-means[i])*(data-means[i]),axis=0) + args.alpha)

        # predicting test data class
        test_pred = []
        logpdf = []
        for c in range(classes):
            logpdf.append(scipy.stats.norm.logpdf(test_data, loc=means[c], scale=np.sqrt(variances[c]))) 
        for i,x in enumerate(test_data):
            prob_of_class = np.zeros(classes)
            for c in range(classes):
                prob_of_class[c] = np.sum(logpdf[c][i]) #scipy.stats.norm.logpdf(xi, loc=means[c][i], scale=np.sqrt(variances[c][i]))
                prob_of_class[c] += np.log(class_probs[c])
            test_pred.append(np.argmax(prob_of_class))
        test_accuracy = sum(test_pred==test_target)/len(test_target)
    elif args.naive_bayes_type == "multinomial":
        sum_features = []
        pk = []
        for i in range(classes):
            data = train_data[train_target == i]
            sum_features.append(np.sum(data, axis=0))
            pk.append((sum_features[i] + args.alpha) / (np.sum(sum_features[i]) + args.alpha * len(sum_features[i]))) 

        # predicting test data class
        test_pred = []
        for x in test_data:
            prob_of_class = np.zeros(classes)
            for i,xi in enumerate(x):
                for k in range(classes):
                    prob_of_class[k] += xi * np.log(pk[k][i])
            for k in range(classes):
                prob_of_class[k] += np.log(class_probs[k])
            test_pred.append(np.argmax(prob_of_class))
        test_accuracy = sum(test_pred==test_target)/len(test_target)

    elif args.naive_bayes_type == "bernoulli":
        non_zero_features = []
        pk = []
        for c in range(classes):
            data = train_data[train_target == c]
            N = data.shape[0]
            non_zero_features.append(np.sum(data!=0, axis=0))
            pk.append((non_zero_features[c] + args.alpha) / (N + 2*args.alpha))
        
        # predicting test data class
        test_pred = []
        for x in test_data:
            prob_of_class = np.zeros(classes)
            for i,xi in enumerate(x):
                xi = 1 if xi > 0 else 0
                for k in range(classes):
                    prob_of_class[k] += xi * np.log(pk[k][i]/(1-pk[k][i])) + np.log(1-pk[k][i])
            for k in range(classes):
                prob_of_class[k] += np.log(class_probs[k])
            #print(prob_of_class)
            test_pred.append(np.argmax(prob_of_class))
        print(np.array(test_pred))
        print(test_target)
        test_accuracy = sum(test_pred==test_target)/len(test_target)
    
    # TODO: Predict the test data classes and compute test accuracy.
    return test_accuracy

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    test_accuracy = main(args)

    print("Test accuracy {:.2f}%".format(100 * test_accuracy))
