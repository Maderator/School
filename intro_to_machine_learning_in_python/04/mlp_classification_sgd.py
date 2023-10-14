#!/usr/bin/env python3
import argparse
import sys

import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=10, type=int, help="Batch size")
parser.add_argument("--classes", default=10, type=int, help="Number of classes to use")
parser.add_argument("--hidden_layer", default=50, type=int, help="Hidden layer size")
parser.add_argument("--iterations", default=10, type=int, help="Number of iterations over the data")
parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=797, type=lambda x:int(x) if x.isdigit() else float(x), help="Test set size")
# If you add more arguments, ReCodEx will keep them with your default values.

def main(args):
    # Create a random generator with a given seed
    generator = np.random.RandomState(args.seed)
    
    # Use the digits dataset
    data, target = sklearn.datasets.load_digits(n_class=args.classes, return_X_y=True)

    # Split the dataset into a train set and a test set.
    # Use `sklearn.model_selection.train_test_split` method call, passing
    # arguments `test_size=args.test_size, random_state=args.seed`.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, stratify=target, test_size=args.test_size, random_state=args.seed)

    # Generate initial model weights
    weights = [generator.uniform(size=[train_data.shape[1], args.hidden_layer], low=-0.1, high=0.1),
               generator.uniform(size=[args.hidden_layer, args.classes], low=-0.1, high=0.1)]
    biases = [np.zeros(args.hidden_layer), np.zeros(args.classes)]

    train_target_prob = np.zeros((train_target.shape[0], args.classes))
    train_target_prob[[i for i in range(train_target.shape[0])], train_target] = 1

    def softmax(z):
        m = np.max(z)
        e_x = np.exp(z - m) # avoiding the overflow errors with max so that the values in exponent are negative
        e_x_sum = np.sum(e_x)
        div = e_x / e_x_sum
        return div

    def cross_entropy(pred, y):
        return - np.sum(y * np.log(pred), axis=0)

    def softmax_cross_entropy(pred, y):
        return - np.sum(y * softmax(pred), axis=0)

    def ReLU(x):
        return np.maximum(x, 0)

    def forward(inputs):
        # TODO: Implement forward propagation, returning *both* the value of the hidden
        # layer and the value of the output layer.
        #
        # We assume a neural network with a single hidden layer of size `args.hidden_layer`
        # and ReLU activation, where ReLU(x) = max(x, 0), and an output layer with softmax
        # activation.
        #
        # The value of the hidden layer is computed as ReLU(inputs @ weights[0] + biases[0]).
        # The value of the output layer is computed as softmax(hidden_layer @ weights[1] + biases[1]).
        #
        # Note that you need to be careful when computing softmax, because the exponentiation
        # in softmax can easily overflow. To avoid it, you can use the fact that
        # softmax(z) = softmax(z + any_constant) and compute softmax(z) = softmax(z - maximum_of_z).
        # That way we only exponentiate values which are non-positive, and overflow does not occur.

        #print(inputs @ weights[0] + biases[0])
        val_hl = inputs @ weights[0] + biases[0]
        activation_val_hl = ReLU(val_hl)
        val_outl = activation_val_hl @ weights[1] + biases[1]
        activation_val_outl = softmax(val_outl) 
        return [activation_val_hl, activation_val_outl]

    for iteration in range(args.iterations):
        permutation = generator.permutation(train_data.shape[0])

        # TODO: Process the data in the order of `permutation`.
        # For every `args.batch_size`, average their gradient, and update the weights.
        # You can assume that `args.batch_size` exactly divides `train_data.shape[0]`.
        #
        # The gradient used in SGD has now four parts, gradient of weights[0] and weights[1]
        # and gradient of biases[0] and biases[1].
        #
        # You can either compute the gradient directly from the neural network formula,
        # i.e., as a gradient of -log P(target | data), or you can compute
        # it step by step using the chain rule of derivatives, in the following order:
        # - compute the derivative of the loss with respect to *inputs* of the
        #   softmax on the last layer
        # - compute the derivative with respect to weights[1] and biases[1]
        # - compute the derivative with respect to the hidden layer output
        # - compute the derivative with respect to the hidden layer input
        # - compute the derivative with respect to weights[0] and biases[0]

        gradient_w1, gradient_b1, gradient_w0, gradient_b0, gradient_components = 0, 0, 0, 0, 0
        for i in permutation:
            outs = forward(train_data[i])
            cost = (outs[1] - train_target_prob[i])

            gradient_w1 += np.outer(cost, outs[0])
            gradient_b1 += cost
            
            hid_layer_out = weights[1] @ cost
            
            relu_der = outs[0].copy() #hid_layer_out.copy()
            relu_der[relu_der > 0] = 1
            error0 = relu_der * hid_layer_out  #* (weights[0].T@train_data[i]) 
 
            gradient_w0 += np.outer(error0, train_data[i])
            gradient_b0 += error0


            gradient_components += 1
            if gradient_components == args.batch_size:
                weights[1] -= args.learning_rate * gradient_w1.T / gradient_components
                biases[1] -= args.learning_rate * gradient_b1 / gradient_components
                weights[0] -= args.learning_rate * gradient_w0.T / gradient_components
                biases[0] -= args.learning_rate * gradient_b0 / gradient_components
                #print(biases[0])
                gradient_w1, gradient_w0, gradient_b1, gradient_b0, gradient_components = 0, 0, 0, 0, 0
        assert gradient_components == 0

        #w20 = weights[1][0,:20]
        #print(["{:.2f}".format(w) for w in w20.ravel()])
        #print()

        # TODO: After the SGD iteration, measure the accuracy for both the
        # train test and the test set and print it in percentages.
        train_outs = []
        for ins in train_data:
            train_outs.append(forward(ins)[1])
        train_accuracy = np.sum(np.argmax(train_outs, axis=1) == train_target)/train_target.shape[0]
        
        test_outs = []
        for ins in test_data:
            test_outs.append(forward(ins)[1])
        test_accuracy = np.sum(np.argmax(test_outs, axis=1) == test_target)/test_target.shape[0]

        print("After iteration {}: train acc {:.1f}%, test acc {:.1f}%".format(
            iteration + 1, 100 * train_accuracy, 100 * test_accuracy))

    return tuple(weights + biases)

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    parameters = main(args)
    print("Learned parameters:", *(" ".join([" "] + ["{:.2f}".format(w) for w in ws.ravel()[:20]] + ["..."]) for ws in parameters), sep="\n")
