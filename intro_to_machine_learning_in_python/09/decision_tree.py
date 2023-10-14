#!/usr/bin/env python3
import argparse

import heapq
import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--criterion", default="gini", type=str, help="Criterion to use; either `gini` or `entropy`")
parser.add_argument("--max_depth", default=None, type=int, help="Maximum decision tree depth")
parser.add_argument("--max_leaves", default=None, type=int, help="Maximum number of leaf nodes")
parser.add_argument("--min_to_split", default=2, type=int, help="Minimum examples required to split")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=91, type=int, help="Random seed")
parser.add_argument("--test_size", default=42, type=lambda x:int(x) if x.isdigit() else float(x), help="Test set size")
# If you add more arguments, ReCodEx will keep them with your default values.

class Node:
    def __init__(self, args, train_examples, target_classes, depth):
        self.args = args
        self.train_examples = train_examples
        self.target_classes = target_classes
        self.most_frequent_class = self.get_most_frequent_class(target_classes)
        self.criterion = self.get_criterion(target_classes)
        self.depth = depth
        self.decision_feature = None
        self.decision_threshold = None
        self.criterion_difference = None
        self.l_indices = None
        self.r_indices = None
        self.left = None
        self.right = None

        depth_cond = args.max_depth == None or depth < args.max_depth
        if len(train_examples) >= args.min_to_split and depth_cond and self.criterion != 0:
            self.decision_feature, self.decision_threshold, self.criterion_difference, \
                self.l_indices, self.r_indices = self.get_best_split(train_examples, target_classes)
            if args.max_leaves == None:
                self.split_node()

    def __str__(self):
        cur_node = "depth:" + str(self.depth) + " crit:" + str(self.criterion) + " instances:" + str(len(self.train_examples))
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

    def split_node(self):
        self.left = Node(self.args, self.train_examples[self.l_indices], self.target_classes[self.l_indices], self.depth+1)
        self.right = Node(self.args, self.train_examples[self.r_indices], self.target_classes[self.r_indices], self.depth+1)

    def get_best_split(self, train_examples, target_classes):
        best_crit_diff = 0
        best_cd_indice = None
        best_feature = None
        best_decision_threshold = None
        best_indices = None
        for f in range(train_examples.shape[1]):
            feature = train_examples[:,f]
            sorted_indices = np.argsort(feature)
            for i in range(len(sorted_indices)-1):
                left_classes = target_classes[sorted_indices[:i+1]]
                right_classes = target_classes[sorted_indices[i+1:]]
                l_crit = self.get_criterion(left_classes)
                r_crit = self.get_criterion(right_classes)
                crit_diff = l_crit + r_crit - self.criterion
                if crit_diff < best_crit_diff:
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
        if self.args.criterion == 'gini':
            return self.get_gini(target_classes)
        else:
            return self.get_entropy(target_classes)

    def get_gini(self, target_classes):
        n = len(target_classes)
        classes = np.max(target_classes)+1
        gini = 0
        for c in range(classes):
            class_prob = np.sum(target_classes==c) / n
            gini += class_prob * (1 - class_prob)
        gini *= n
        return gini

    def get_entropy(self, target_classes):
        n = len(target_classes)
        classes = np.max(target_classes)+1
        entropy = 0
        for c in range(classes):
            class_prob = np.sum(target_classes==c) / n
            if class_prob > 0:
                entropy += class_prob * np.log(class_prob)
        entropy *= -n
        return entropy

def main(args):
    # Use the wine dataset
    data, target = sklearn.datasets.load_wine(return_X_y=True)

    # Split the data randomly to train and test using `sklearn.model_selection.train_test_split`,
    # with `test_size=args.test_size` and `random_state=args.seed`.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, test_size=args.test_size, random_state=args.seed)

    # TODO: Create a decision tree on the trainining data.
    #
    # - For each node, predict the most frequent class (and the one with
    #   smallest index if there are several such classes).
    
    root = Node(args, train_data, train_target, 0)

    if args.max_leaves != None:
        leaves = 1
        leafs_sorted = []
        if root.criterion_difference != None:   # the node meet conditions to be split
            heapq.heappush(leafs_sorted, (root.criterion_difference, root))
        while leaves < args.max_leaves and len(leafs_sorted) > 0:
            cur_node = heapq.heappop(leafs_sorted)[1]
            cur_node.split_node()
            leaves += 1
            l_child = cur_node.left
            r_child = cur_node.right
            if l_child.criterion_difference != None:
                heapq.heappush(leafs_sorted, (l_child.criterion_difference, l_child))
            if r_child.criterion_difference != None:
                heapq.heappush(leafs_sorted, (r_child.criterion_difference, r_child))

    # - When splitting a node, consider the features in sequential order, then
    #   for each feature consider all possible split points ordered in ascending
    #   value, and perform the first encountered split descreasing the criterion
    #   the most. Each split point is an average of two nearest feature values
    #   of the instances corresponding to the given node (i.e., for three instances
    #   with values 1, 7, 3, the split points are 2 and 5).
    #
    # - Allow splitting a node only if:
    #   - when `args.max_depth` is not None, its depth must be at most `args.max_depth`;
    #     depth of the root node is zero;
    #   - there are at least `args.min_to_split` corresponding instances;
    #   - the criterion value is not zero.
    #
    # - When `args.max_leaves` is None, use recursive (left descendants first, then
    #   right descendants) approach, splitting every node if the constraints are valid.
    #   Otherwise (when `args.max_leaves` is not None), always split a node where the
    #   constraints are valid and the overall criterion value (c_left + c_right - c_node)
    #   decreases the most. If there are several such nodes, choose the one
    #   which was created sooner (a left child is considered to be created
    #   before a right child).

    # TODO: Finally, measure the training and testing accuracy.
    classifications = []
    for d in train_data:
        classifications.append(root.classify_sample(d))
    classifications = np.array(classifications)
    train_accuracy = np.sum(train_target == classifications) / len(train_target)

    classifications = []
    for d in test_data:
        classifications.append(root.classify_sample(d))
    classifications = np.array(classifications)
    test_accuracy = np.sum(test_target == classifications) / len(test_target)

    return train_accuracy, test_accuracy

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    train_accuracy, test_accuracy = main(args)

    print("Train accuracy: {:.1f}%".format(100 * train_accuracy))
    print("Test accuracy: {:.1f}%".format(100 * test_accuracy))

