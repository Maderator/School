#!/usr/bin/env python3

# edbe2dad-018e-11eb-9574-ea7484399335
# 44752d3d-fdd8-11ea-9574-ea7484399335

import argparse
import datetime
import os
import re

from tensorflow.python.ops.math_ops import reduce_sum
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2") # Report only TF errors by default

import numpy as np
import tensorflow as tf

from mnist import MNIST

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
parser.add_argument("--cnn", default="5-3-2,10-3-2", type=str, help="CNN architecture.")
parser.add_argument("--epochs", default=5, type=int, help="Number of epochs.")
parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# If you add more arguments, ReCodEx will keep them with your default values.

class Convolution:
    def __init__(self, channels, kernel_size, stride, input_shape):
        # Create convolutional layer with the given arguments
        # and given input shape (e.g., [28, 28, 1]).
        self._channels = channels
        self._kernel_size = kernel_size
        self._stride = stride

        # Here the kernel and bias variables are created
        self._kernel = tf.Variable(
            tf.initializers.GlorotUniform(seed=42)([self._kernel_size, self._kernel_size, input_shape[2], self._channels]),
            trainable=True)
        self._bias = tf.Variable(tf.initializers.Zeros()([self._channels]), trainable=True)

    def forward(self, inputs):
        # TODO: Compute the forward propagation through the convolution
        # with `tf.nn.relu` activation and return the result.
        #
        # In order for the computation to be reasonably fast, you cannot
        # manually iterate through the individual pixels, batch examples,
        # input channels or output channels. However, you can manually
        # iterate through the kernel size.
        h = tf.cast(tf.math.ceil((inputs.shape[1]-self._kernel_size+1)/self._stride), tf.int32)
        w = tf.cast(tf.math.ceil((inputs.shape[2]-self._kernel_size+1)/self._stride), tf.int32)
        max_h = h * self._stride
        max_w = w * self._stride
        out_shape = (inputs.shape[0], h, w, self._channels)
        outputs = tf.zeros(shape=out_shape)
        ks = self._kernel_size

        for kh in range(ks):
            for kw in range(ks):
                khw_elems = inputs[:, kh:max_h+kh:self._stride, kw:max_w+kw:self._stride, :]
                outputs += khw_elems @ self._kernel[kh, kw, :, :]
        
        return tf.nn.relu(outputs + self._bias)

    def extend_dim(self, dim, tensor, in_shape):
        # 1. add dimension
        extended_og = tf.expand_dims(tensor, axis=dim+1)

        # 2. pad dimension with zeros to compensate for stride
        paddings = [[0,0], [0,0], [0,0], [0,0], [0,0]]
        paddings[dim+1] = [0, self._stride-1]
        paddings = tf.constant(paddings)
        extended_og = tf.pad(extended_og, paddings, "CONSTANT")
        shape = list(tensor.shape)
        shape[dim] = shape[dim]*self._stride
        extended_og = tf.reshape(extended_og, shape=tuple(shape))
        
        # 3. add (if needed) zeros to end to get same size as input tensor
        row_dif = in_shape[dim] - extended_og.shape[dim] 
        if row_dif > 0:
            paddings = [[0,0], [0,0], [0,0], [0,0]]
            paddings[dim] = [0,row_dif]
            paddings = tf.constant(paddings)
            extended_og = tf.pad(extended_og, paddings, "CONSTANT")
        
        # 4. add zeros before dimension to be able to offset when multiplying by element in kernel
        paddings = [[0,0], [0,0], [0,0], [0,0]]
        paddings[dim] = [self._kernel_size-1,0]
        paddings = tf.constant(paddings)
        extended_og = tf.pad(extended_og, paddings, "CONSTANT")

        return extended_og

    def backward(self, inputs, outputs, outputs_gradient):
        # TODO: Given the inputs of the layer, outputs of the layer
        # (computed in forward pass) and the gradient of the loss
        # with respect to layer outputs, return a list with the
        # following three elements:
        # - gradient of the loss with respect to inputs
        # - list of variables in the layer, e.g.,
        #     [self._kernel, self._bias]
        # - list of gradients of the loss with respect to the layer
        #   variables (in the same order as the previous argument)
        
        h = tf.cast(tf.math.ceil((inputs.shape[1]-self._kernel_size+1)/self._stride), tf.int32)
        w = tf.cast(tf.math.ceil((inputs.shape[2]-self._kernel_size+1)/self._stride), tf.int32)
        max_h = h * self._stride
        max_w = w * self._stride
        ks = self._kernel_size
        stride = self._stride
        
        # RELU GRADIENT
        relu_der = tf.cast(outputs > 0, outputs.dtype)
        relu_gradient = relu_der * outputs_gradient

        #gradient_kernel = tf.zeros(shape=self._kernel.shape)
        in_sum = tf.math.reduce_sum(inputs, axis=0) # sum over batch
        relu_g_sum = tf.math.reduce_sum(relu_gradient, axis=0) # sum over batch

        # KERNEL GRADIENT
        kg_h_list = []
        for kh in range(ks):
            kg_w_list = []
            for kw in range(ks):
                khkw_in = inputs[:, kh:max_h+kh:self._stride, kw:max_w+kw:self._stride, :]
                khkw_g = tf.expand_dims(relu_gradient, axis=3) * tf.expand_dims(khkw_in, axis=4)
                khkw_g = tf.reduce_sum(tf.reduce_sum(tf.reduce_sum(khkw_g, axis=0), axis=0), axis=0)
                kg_w_list.append(khkw_g)
            kg_h_list.append(tf.stack(kg_w_list))
        gradient_kernel = tf.stack(kg_h_list)

        # BIAS GRADIENT
        # do sum over all gradients in the output_gradient
        gradient_bias = tf.math.reduce_sum(relu_gradient, axis=0) # sum over batch (!! Not mean as the gradient is alaready divided by number of elements in batch !!)
        gradient_bias = tf.math.reduce_sum(gradient_bias, axis=0) # sum over height
        gradient_bias = tf.math.reduce_sum(gradient_bias, axis=0) # sum over width

        # INPUTS GRADIENT
        # 1. we need to expand relu_gradient with zeros so that we can do tensor multiplication like in forward pass
        extended_rg = relu_gradient
        extended_rg = self.extend_dim(1, extended_rg, inputs.shape)
        extended_rg = self.extend_dim(2, extended_rg, inputs.shape)
        inputs_gradient = tf.zeros(shape=inputs.shape)
        
        ks = self._kernel_size
        for kh in range(ks):
            for kw in range(ks):
                e_dim_len = extended_rg.shape[1]
                e = extended_rg[:, 
                                ks - 1 -kh:e_dim_len-kh, 
                                ks - 1 -kw:e_dim_len-kw, :]
                gradient = e @ tf.transpose(self._kernel[kh, kw, :, :], perm=[1,0])
                inputs_gradient += gradient

        return inputs_gradient, [self._kernel, self._bias], [gradient_kernel, gradient_bias]
class Network:
    def __init__(self, args):
        self._args = args

        # Create the convolutional layers according to `args.cnn`.
        input_shape = [MNIST.H, MNIST.W, MNIST.C]
        self._convolutions = []
        for layer in args.cnn.split(","):
            channels, kernel_size, stride = map(int, layer.split("-"))
            self._convolutions.append(Convolution(channels, kernel_size, stride, input_shape))
            input_shape = [(input_shape[0] - kernel_size) // stride + 1,
                           (input_shape[1] - kernel_size) // stride + 1, channels]

        # Create the classification head
        self._flatten = tf.keras.layers.Flatten(input_shape=input_shape)
        self._classifier = tf.keras.layers.Dense(MNIST.LABELS, activation=tf.nn.softmax)

        # Create the loss, metric and the optimizer
        self._loss = tf.losses.SparseCategoricalCrossentropy()
        self._accuracy = tf.metrics.SparseCategoricalAccuracy()
        self._optimizer = tf.optimizers.Adam(args.learning_rate)

    def train_epoch(self, dataset):
        for batch in dataset.batches(self._args.batch_size):
            # Forward pass through the convolutions
            hidden = tf.constant(batch["images"])
            convolution_values = [hidden]
            for convolution in self._convolutions:
                hidden = convolution.forward(hidden)
                convolution_values.append(hidden)

            # Run the classification head and compute its gradient
            with tf.GradientTape() as tape:
                tape.watch(hidden)

                predictions = self._flatten(hidden)
                predictions = self._classifier(predictions)
                loss = self._loss(batch["labels"], predictions)

            variables = self._classifier.trainable_variables
            hidden_gradient, *gradients = tape.gradient(loss, [hidden] + variables)

            # Backpropagate the gradient throug the convolutions
            for convolution, inputs, outputs in reversed(list(zip(self._convolutions, convolution_values[:-1], convolution_values[1:]))):
                hidden_gradient, convolution_variables, convolution_gradients =convolution.backward(inputs, outputs, hidden_gradient)
                variables.extend(convolution_variables)
                gradients.extend(convolution_gradients)

            # Update the weights
            self._optimizer.apply_gradients(zip(gradients, variables))

    def evaluate(self, dataset):
        self._accuracy.reset_states()
        for batch in dataset.batches(self._args.batch_size):
            hidden = batch["images"]
            for convolution in self._convolutions:
                hidden = convolution.forward(hidden)
            hidden = self._flatten(hidden)
            predictions = self._classifier(hidden)
            self._accuracy(batch["labels"], predictions)
        return self._accuracy.result().numpy()


def main(args):
    # Fix random seeds and threads
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    if args.recodex:
        tf.keras.utils.get_custom_objects()["glorot_uniform"] = tf.initializers.GlorotUniform(seed=args.seed)
        tf.keras.utils.get_custom_objects()["orthogonal"] = tf.initializers.Orthogonal(seed=args.seed)
        tf.keras.utils.get_custom_objects()["uniform"] = tf.initializers.RandomUniform(seed=args.seed)

    # Load data, using only 10000 training images
    mnist = MNIST()
    mnist.train._size = 10000

    # Create the model
    network = Network(args)

    for epoch in range(args.epochs):
        network.train_epoch(mnist.train)

        accuracy = network.evaluate(mnist.dev)
        print("Dev accuracy after epoch {} is {:.2f}".format(epoch + 1, 100 * accuracy), flush=True)

    accuracy = network.evaluate(mnist.test)
    print("Test accuracy after epoch {} is {:.2f}".format(epoch + 1, 100 * accuracy), flush=True)

    # Return the test accuracy for ReCodEx to validate.
    return accuracy

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
