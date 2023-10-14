#!/usr/bin/env python3
import argparse

import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
import math

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--bootstrapping", default=False, action="store_true", help="Perform data bootstrapping")
parser.add_argument("--feature_subsampling", default=0.5, type=float, help="What fraction of features to subsample")
parser.add_argument("--max_depth", default=2, type=int, help="Maximum decision tree depth")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=46, type=int, help="Random seed")
parser.add_argument("--test_size", default=42, type=lambda x:int(x) if x.isdigit() else float(x), help="Test set size")
parser.add_argument("--trees", default=3, type=int, help="Number of trees in the forest")
# If you add more arguments, ReCodEx will keep them with your default values.

class Node:
    def __init__(self, args, train_examples, target_classes, depth, generator):
        self.args = args
        self.train_examples = train_examples
        self.target_classes = target_classes
        self.most_frequent_class = self.get_most_frequent_class(target_classes)
        self.criterion = self.get_criterion(target_classes)
        self.depth = depth
        #self.generator = generator
        self.decision_feature = None
        self.decision_threshold = None
        self.criterion_difference = None
        self.l_indices = None
        self.r_indices = None
        self.left = None
        self.right = None

        depth_cond = args.max_depth == None or depth < args.max_depth
        if depth_cond and self.criterion:
            self.decision_feature, self.decision_threshold, self.criterion_difference, \
                self.l_indices, self.r_indices = self.get_best_split(train_examples, target_classes, generator)
            self.split_node(generator)

    def __str__(self):
        cur_node = " depth:" + str(self.depth) + " crit:" + str(self.criterion) + " instances:" + str(len(self.train_examples)) + ", thr:" + str(self.decision_threshold) + ", feat:" + str(self.decision_feature)
        left_node = ""
        right_node = ""
        if self.left != None:
            left_node = self.left.__str__()
            right_node = self.right.__str__()

        return cur_node + "\n\t left" + left_node + "\n\t right" + right_node

    def classify_sample(self, sample):
        if self.left == None:
            return self.most_frequent_class
        else:
            feature_val = sample[self.decision_feature]
            if feature_val < self.decision_threshold:
                return self.left.classify_sample(sample)
            else:
                return self.right.classify_sample(sample)

    def split_node(self, generator):
        self.left = Node(self.args, self.train_examples[self.l_indices], self.target_classes[self.l_indices], self.depth+1, generator)
        self.right = Node(self.args, self.train_examples[self.r_indices], self.target_classes[self.r_indices], self.depth+1, generator)

    def get_best_split(self, train_examples, target_classes, generator):
        best_crit_diff = 0
        best_cd_indice = None
        best_feature = None
        best_decision_threshold = None
        best_indices = None

        # - feature subsampling: when searching for the best split, try only
            #   a subset of features. When splitting a node, start by generating
            #   a feature mask using
            #     generator.uniform(size=number_of_features) <= feature_subsampling
            #   which gives a boolean value for every feature, with `True` meaning the
            #   feature is used during best split search, and `False` it is not.
            #   (When feature_subsampling == 1, all features are used.)
        features_subset = generator.uniform(size=train_examples.shape[1]) <= self.args.feature_subsampling
        print(features_subset)
        for f, is_used in enumerate(features_subset):
            if is_used:
                feature = train_examples[:,f]
                sorted_indices = np.argsort(feature)
                for i in range(len(sorted_indices)-1):
                    if feature[sorted_indices[i]] == feature[sorted_indices[i+1]]:
                        print(target_classes[sorted_indices[i:i+2]])
                        continue
                    left_classes = target_classes[sorted_indices[:i+1]]
                    right_classes = target_classes[sorted_indices[i+1:]]
                    l_crit = self.get_criterion(left_classes)
                    r_crit = self.get_criterion(right_classes)
                    crit_diff = l_crit + r_crit - self.criterion
                    if crit_diff < best_crit_diff:
                        #if f == 11 and i == 47:
                        #    print("feature 12: OD280... is better than Flavanoids")
                        #    l_crit = self.get_criterion(left_classes)
                        #    r_crit = self.get_criterion(right_classes)
                        best_crit_diff = crit_diff
                        best_cd_indice = i
                        best_indices = sorted_indices
                        best_feature = f
                        best_decision_threshold = (feature[sorted_indices[i]] + feature[sorted_indices[i+1]]) / 2
        l_indices = best_indices[:best_cd_indice+1]
        r_indices = best_indices[best_cd_indice+1:]
        return best_feature, best_decision_threshold, best_crit_diff, l_indices, r_indices

    def get_most_frequent_class(self, target_classes):
        n = len(target_classes)
        classes = np.max(target_classes)+1
        mf_class = 0
        mfc_frequency = 0
        for c in range(classes):
            c_count = np.sum(target_classes==c)
            if c_count > mfc_frequency:
                mfc_frequency = c_count
                mf_class = c
        return mf_class


    def get_criterion(self, target_classes):
        #if self.args.criterion == 'gini':
        #    return self.get_gini(target_classes)
        #else:
        return self.get_entropy(target_classes)

   #def get_gini(self, target_classes):
   #    n = len(target_classes)
   #    classes = np.max(target_classes)+1
   #    gini = 0
   #    for c in range(classes):
   #        class_prob = np.sum(target_classes==c) / n
   #        gini += class_prob * (1 - class_prob)
   #    gini *= n
   #    return gini

    def get_entropy(self, target_classes):
        n = len(target_classes)
        classes = np.max(target_classes)+1
        entropy = 0
        for c in range(classes):
            class_prob = np.sum(target_classes==c) / n
            if class_prob > 0:
                entropy += class_prob * math.log(class_prob)
        entropy *= -n
        return entropy


def main(args):
    # Use the wine dataset
    data, target = sklearn.datasets.load_wine(return_X_y=True)

    # Split the data randomly to train and test using `sklearn.model_selection.train_test_split`,
    # with `test_size=args.test_size` and `random_state=args.seed`.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, test_size=args.test_size, random_state=args.seed)

    # TODO: Create a random forest on the trainining data.
    #
    # For determinism, create a generator
    #   generator = np.random.RandomState(args.seed)
    # at the beginning and then use this instance for all random number generation.
    generator = np.random.RandomState(args.seed)

    # Use a simplified decision tree from the `decision_tree` assignment. The
    # tree needs to support only the `entropy` criterion and `max_depth` constraint.
    # When splitting nodes, proceed in the depth-first order, splitting all nodes
    # in left subtrees before nodes in right subtrees.
    #
    # Additionally, implement:
    # - feature subsampling: when searching for the best split, try only
    #   a subset of features. When splitting a node, start by generating
    #   a feature mask using
    #     generator.uniform(size=number_of_features) <= feature_subsampling
    #   which gives a boolean value for every feature, with `True` meaning the
    #   feature is used during best split search, and `False` it is not.
    #   (When feature_subsampling == 1, all features are used.)
    #
    # - train a random forest consisting of `args.trees` decision trees
    #
    # - if `args.bootstrapping` is set, right before training a decision tree,
    #   create a bootstrap sample of the training data using the following indices
    #     indices = generator.randint(len(train_data), size=len(train_data))
    #   and if `args.bootstrapping` is not set, use the original training data.

    trees = []
    for i in range(args.trees):
        print("tree", i)
        if args.bootstrapping:
            indices = generator.randint(len(train_data), size=len(train_data))
            data = train_data[indices]
            target = train_target[indices]
        else:
            data = train_data
            target = train_target
        trees.append(Node(args, data, target, 0, generator))    
        print(trees[-1])

    # During prediction, use voting to find the most frequent class for a given
    # input, choosing the one with smallest class index in case of a tie.

    # TODO: Finally, measure the training and testing accuracy.
    def measure_accuracy(data, trees, target):
        classifications = []
        for d in data:
            votes = []
            for t in range(len(trees)):
                votes.append(trees[t].classify_sample(d))
            classifications.append(np.bincount(votes).argmax())
        classifications = np.array(classifications)
        return np.sum(target == classifications) / len(target)
    
    train_accuracy = measure_accuracy(train_data, trees, train_target)
    test_accuracy = measure_accuracy(test_data, trees, test_target)

    return train_accuracy, test_accuracy

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    train_accuracy, test_accuracy = main(args)

    print("Train accuracy: {:.1f}%".format(100 * train_accuracy))
    print("Test accuracy: {:.1f}%".format(100 * test_accuracy))
