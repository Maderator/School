#!/usr/bin/env python3

# edbe2dad-018e-11eb-9574-ea7484399335
# 44752d3d-fdd8-11ea-9574-ea7484399335

import argparse
import datetime
import os
import re
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2") # Report only TF errors by default

import numpy as np
import tensorflow as tf

from mnist import MNIST

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
parser.add_argument("--epochs", default=5, type=int, help="Number of epochs.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# If you add more arguments, ReCodEx will keep them with your default values.

# The neural network model
class Network(tf.keras.Model):
    def __init__(self, args):
        # Create a model with two inputs, both images of size [MNIST.H, MNIST.W, MNIST.C].
        images = (
            tf.keras.layers.Input(shape=[MNIST.H, MNIST.W, MNIST.C]),
            tf.keras.layers.Input(shape=[MNIST.H, MNIST.W, MNIST.C]),
        )

        # TODO: The model starts by passing each input image through the same
        # subnetwork (with shared weights), which should perform
        # - convolution with 10 filters, 3x3 kernel size, stride 2, "valid" padding, ReLU activation,
        # - convolution with 20 filters, 3x3 kernel size, stride 2, "valid" padding, ReLU activation,
        # - flattening layer,
        # - fully connected layer with 200 neurons and ReLU activation,
        # obtaining a 200-dimensional feature representation of each image.
        input = tf.keras.layers.Input(shape=[MNIST.H, MNIST.W, MNIST.C])
        hidden = tf.keras.layers.Conv2D(10, 3, 2, "valid", activation=tf.nn.relu)(input)
        hidden = tf.keras.layers.Conv2D(20, 3, 2, "valid", activation=tf.nn.relu)(hidden)
        hidden = tf.keras.layers.Flatten()(hidden)
        hidden = tf.keras.layers.Dense(200, activation=tf.nn.relu)(hidden)
        img_subnet_model = tf.keras.Model(inputs=input, outputs=hidden)
        img_subnet_model.summary()

        img1_features = img_subnet_model(images[0])
        img2_features = img_subnet_model(images[1])

        # TODO: Using the computed representations, it should produce four outputs:
        # - first, compute _direct prediction_ whether the first digit is
        #   greater than the second, by
        #   - concatenating the two 200-dimensional image representations,
        #   - processing them using another 200-neuron ReLU dense layer
        #   - computing one output using a dense layer with `tf.nn.sigmoid` activation
        # - then, classify the computed representation of the first image using
        #   a densely connected softmax layer into 10 classes;
        # - then, classify the computed representation of the second image using
        #   the same connected layer (with shared weights) into 10 classes;
        # - finally, compute _indirect prediction_ whether the first digit
        #   is greater than second, by comparing the predictions from the above
        #   two outputs.
        imgs_features = [img1_features, img2_features]
        direct = tf.keras.layers.concatenate(imgs_features)
        direct = tf.keras.layers.Dense(200, activation=tf.nn.relu)(direct)
        direct = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)(direct)

        input_digit = tf.keras.layers.Input(shape=[200])
        digit_hidden = tf.keras.layers.Dense(10, activation=tf.nn.softmax)(input_digit)
        digit_subnet_model = tf.keras.Model(inputs=input_digit, outputs=digit_hidden)

        digit1 = digit_subnet_model(img1_features)
        digit2 = digit_subnet_model(img2_features)

        indirect = tf.greater(tf.argmax(digit1, axis=-1), tf.argmax(digit2, axis=-1))

        outputs = {
            "direct_prediction": direct,
            "digit_1": digit1,
            "digit_2": digit2,
            "indirect_prediction": indirect,
        }

        # Finally, construct the model.
        super().__init__(inputs=images, outputs=outputs)

        # TODO delete
        super().summary()

        # Note that for historical reasons, names of a functional model outputs
        # (used for displayed losses/metric names) are derived from the name of
        # the last layer of the corresponding output. Here we instead use
        # the keys of the `outputs` dictionary.
        self.output_names = sorted(outputs.keys())

        # TODO: Train the model by computing appropriate losses of
        # direct_prediction, digit_1, digit_2. Regarding metrics, compute
        # the accuracy of both the direct and indirect predictions; name both
        # metrics "accuracy" (i.e., pass "accuracy" as the first argument of
        # the metric object).
        self.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss={
                "direct_prediction": tf.losses.BinaryCrossentropy(),
                "digit_1": tf.losses.SparseCategoricalCrossentropy(),
                "digit_2": tf.losses.SparseCategoricalCrossentropy(),
            },
            metrics={
                "direct_prediction": [tf.metrics.BinaryAccuracy(name="accuracy")],
                "indirect_prediction": [tf.metrics.BinaryAccuracy(name="accuracy")],
            },
        )
        self.tb_callback = tf.keras.callbacks.TensorBoard(args.logdir, histogram_freq=1, update_freq=100, profile_batch=0)
        self.tb_callback._close_writers = lambda: None # A hack allowing to keep the writers open.

    # Create an appropriate dataset using the MNIST data.
    def create_dataset(self, mnist_dataset, args, training=False):
        # Start by using the original MNIST data
        dataset = tf.data.Dataset.from_tensor_slices((mnist_dataset.data["images"], mnist_dataset.data["labels"]))

        # TODO: If `training`, shuffle the data with `buffer_size=10000` and `seed=args.seed`
        if training:
            dataset = dataset.shuffle(buffer_size=10000, seed=args.seed)

        # TODO: Combine pairs of examples by creating batches of size 2
        dataset = dataset.batch(2)

        # TODO: Map pairs of images to elements suitable for our model. Notably,
        # the elements should be pairs `(input, output)`, with
        # - `input` being a pair of images,
        # - `output` being a dictionary with keys digit_1, digit_2, direct_prediction
        #   and indirect_prediction.
        def create_element(images, labels):
            print(images, labels)
            input = (images[0], images[1])
            pred = float(labels[0] > labels[1])
            output = {
                "direct_prediction": pred,
                "digit_1": labels[0],
                "digit_2": labels[1],
                "indirect_prediction": pred,
            }
            return (input, output)

        dataset = dataset.map(create_element)

        # TODO: Create batches of size `args.batch_size`
        dataset = dataset.batch(args.batch_size)

        return dataset

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

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    ))

    # Load the data
    mnist = MNIST()

    # Create the network
    network = Network(args)

    # Construct suitable datasets from the MNIST data.
    train = network.create_dataset(mnist.train, args, training=True)
    dev = network.create_dataset(mnist.dev, args)
    test = network.create_dataset(mnist.test, args)

    # Train
    network.fit(train, epochs=args.epochs, validation_data=dev, callbacks=[network.tb_callback])

    # Compute test set metrics and return them
    test_logs = network.evaluate(test, return_dict=True)
    network.tb_callback.on_epoch_end(args.epochs, {"val_test_" + metric: value for metric, value in test_logs.items()})

    return test_logs

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
