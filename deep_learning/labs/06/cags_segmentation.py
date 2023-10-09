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
import tensorflow_addons as tfa

from cags_dataset import CAGS
import efficient_net

# TODO: Define reasonable defaults and optionally more parameters
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=8, type=int, help="Batch size.")
parser.add_argument("--epochs", default=50, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=0, type=int, help="Maximum number of threads to use.")
parser.add_argument("--activation", default="relu", type=str, help="Activation type")
parser.add_argument("--optimizer", default="RMSProp", type=str, help="Optimizer type")
parser.add_argument("--decay", default="cosine", type=str, help="Decay type")

# Code copied from cifar_competition.solution.py 
# Author:Milan Straka
class Model(tf.keras.Model):
    def _activation(self, inputs, args):
        if args.activation == "relu":
            return tf.keras.layers.Activation(tf.nn.relu)(inputs)
        if args.activation == "lrelu":
            return tf.keras.layers.Activation(tf.nn.leaky_relu)(inputs)
        if args.activation == "elu":
            return tf.keras.layers.Activation(tf.nn.elu)(inputs)
        if args.activation == "swish":
            return tf.keras.layers.Activation(tf.nn.swish)(inputs)
        if args.activation == "gelu":
            return tf.keras.layers.Activation(tf.nn.gelu)(inputs)
        raise ValueError("Unknown activation '{}'".format(args.activation))

# Inspired by class ResNet in cifar_competition.solution.py (Author of ResNet class: Milan Straka)
class UNet(Model):
    def _cnn(self, inputs, args, filters, kernel_size, stride, activation):
        hidden = tf.keras.layers.Conv2D(filters, kernel_size=kernel_size, strides=stride, padding="same", use_bias=False)(inputs)
        hidden = tf.keras.layers.BatchNormalization()(hidden)
        hidden = self._activation(hidden, args) if activation else hidden
        hidden = tf.keras.layers.Dropout(0.3)(hidden)
        return hidden

    def _cnn_transpose(self, inputs, args, filters, kernel_size, stride, activation):
        hidden = tf.keras.layers.Conv2DTranspose(filters, kernel_size=kernel_size, strides=stride, padding="valid", use_bias=False)(inputs)
        hidden = tf.keras.layers.BatchNormalization()(hidden)
        hidden = self._activation(hidden, args) if activation else hidden
        hidden = tf.keras.layers.Dropout(0.3)(hidden)
        return hidden

    def _conv_transpose_block(self, inputs, next_layer_in, args, next_layer_filters, activation):
        hidden = self._cnn(inputs, args, next_layer_filters, 3, 1, activation)
        hidden = self._cnn(inputs, args, next_layer_filters, 3, 1, activation)
        hidden = self._cnn_transpose(hidden, args, next_layer_filters, 2, 2, activation)
        hidden = tf.keras.layers.Concatenate()([next_layer_in, hidden])
        return hidden

    def __init__(self, args):
        # Load the EfficientNet-B0 model
        efficientnet_b0 = efficient_net.pretrained_efficientnet_b0(include_top=False)

        efficientnet_b0.trainable = False
    
        inputs = tf.keras.Input(shape=[CAGS.H, CAGS.W, CAGS.C], dtype=tf.float32)
    
        efnet_model = efficientnet_b0(inputs)
        efnet_filters = [1280, 1280, 112, 40, 24, 16]

        hidden = self._conv_transpose_block(efnet_model[1], efnet_model[2], args, efnet_filters[2], activation=True)
        for i in range(3, len(efnet_model)):
            next_layer = efnet_model[i]
            hidden = self._conv_transpose_block(hidden, next_layer, args, efnet_filters[i], activation=True)
        hidden = self._cnn_transpose(hidden, args, 16, 2, 2, activation=True)
        hidden = tf.keras.layers.Dropout(0.3)(hidden)
        outputs = tf.keras.layers.Conv2D(2, 1, 1)(hidden)
        super().__init__(inputs, outputs)
            
            #for stage in range(3):
            #    for block in range(n):
            #        hidden = self._block(hidden, args, 16 * (1 << stage), 2 if stage > 0 and block == 0 else 1)
            #hidden = tf.keras.layers.GlobalAvgPool2D()(hidden)
            #outputs = tf.keras.layers.Dense(CAGS.LABELS, activation=tf.nn.softmax)(hidden)
            #super().__init__(inputs, outputs)

# End of copied code

def main(args):
    # Fix random seeds and threads
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    ))

    # Load the data
    cags = CAGS()

    model = UNet(args)
    model.summary()


    tb_callback = tf.keras.callbacks.TensorBoard(args.logdir, histogram_freq=1, update_freq=100, profile_batch=0)

    generator = tf.random.Generator.from_seed(args.seed)
    def train_augment(image, mask):
        
        image = tf.image.random_brightness(image, 0.3)
        image = tf.image.random_saturation(image, 0.5, 3)
        image = tf.image.random_contrast(image, 0.3, 0.6)
        
        return image, mask

    def one_hot_augment(image, mask):    
        mask = tf.one_hot(tf.cast(tf.math.round(mask), tf.int32), 2)
        mask = tf.squeeze(mask, axis=2)
        return image, mask

    #train = tf.data.Dataset.from_tensor_slices( (cags.train.data["images"], cags.train.data["labels"]) )
    #dev = tf.data.Dataset.from_tensor_slices( (cags.dev.data["images"], cags.dev.data["labels"]) )
    train = cags.train.map(lambda example: (example["image"], example["mask"]))
    train = train.map(one_hot_augment)
    train = train.map(train_augment)
    train = train.batch(args.batch_size)
    train = train.prefetch(tf.data.AUTOTUNE)

    dev = cags.dev.map(lambda example: (example["image"], example["mask"]))
    dev = dev.map(one_hot_augment)
    dev = dev.batch(args.batch_size)
    dev = dev.prefetch(tf.data.AUTOTUNE)

    # Decay
    training_batches = args.epochs * len(list(cags.train)) // args.batch_size
    if args.decay == "piecewise":
        decay_fn = lambda value: tf.optimizers.schedules.PiecewiseConstantDecay(
            [int(0.5 * training_batches), int(0.75 * training_batches)],
            [value, value / 10, value / 100])
    elif args.decay == "cosine":
        decay_fn = lambda value: tf.keras.experimental.CosineDecay(value, training_batches)
    else:
        raise ValueError("Uknown decay '{}'".format(args.decay))
    learning_rate = decay_fn(0.01)
    weight_decay = decay_fn(1e-4)

    # Optimizer
    if args.optimizer == "RMSProp":
        optimizer = tfa.optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay, beta_2=0.9, epsilon=1e-3)
    elif args.optimizer == "Adam":
        optimizer = tfa.optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay, epsilon=1e-3)
    else:
        raise ValueError("Uknown optimizer '{}'".format(args.optimizer))
    optimizer = tf.optimizers.Adam(learning_rate=learning_rate, beta_2=0.9, epsilon=1e-3)

    model.compile(
        optimizer=optimizer,
        loss=tf.losses.MeanSquaredError(),
        metrics=[cags.MaskIoUMetric(name="accuracy")],
        #metrics=[tf.metrics.CategoricalAccuracy()],
    )

    model.fit(train, epochs=args.epochs, validation_data=dev, callbacks=[tb_callback])

    model.trainable = True

    learning_rate = decay_fn(0.001)
    # Optimizer
    if args.optimizer == "RMSProp":
        optimizer = tfa.optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay, beta_2=0.9, epsilon=1e-3)
    elif args.optimizer == "Adam":
        optimizer = tfa.optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay, epsilon=1e-3)
    else:
        raise ValueError("Uknown optimizer '{}'".format(args.optimizer))
    optimizer = tf.optimizers.Adam(learning_rate=learning_rate, beta_2=0.9, epsilon=1e-3)
    model.compile(
        optimizer=optimizer,
        loss=tf.losses.MeanSquaredError(),
        metrics=[cags.MaskIoUMetric(name="accuracy")],
        #metrics=[tf.metrics.CategoricalAccuracy()],
    )

    model.fit(train, epochs=args.epochs, validation_data=dev, callbacks=[tb_callback])

    # Generate test set annotations, but in args.logdir to allow parallel execution.
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "cags_segmentation.txt"), "w", encoding="utf-8") as predictions_file:
        # TODO: Predict the masks on the test set
        test = cags.test.map(lambda example: (example["image"], example["mask"])).batch(args.batch_size)
        test_masks = model.predict(test, batch_size = args.batch_size)

        for mask in test_masks:
            mask = tf.argmax(mask, axis=-1) # TODO decode one_hot mask
            mask = tf.expand_dims(mask, axis=-1)
            mask = tf.cast(mask, dtype=tf.float32)

            zeros, ones, runs = 0, 0, []
            for pixel in np.reshape(mask >= 0.5, [-1]):
                if pixel:
                    if zeros or (not zeros and not ones):
                        runs.append(zeros)
                        zeros = 0
                    ones += 1
                else:
                    if ones:
                        runs.append(ones)
                        ones = 0
                    zeros += 1
            runs.append(zeros + ones)
            print(*runs, file=predictions_file)
    
    with open(os.path.join(args.logdir, "dev_cags_segmentation.txt"), "w", encoding="utf-8") as predictions_file:
        # TODO: Predict the masks on the test set
        dev = cags.dev.map(lambda example: (example["image"], example["mask"])).batch(args.batch_size)
        dev_masks = model.predict(dev, batch_size = args.batch_size)

        for mask in dev_masks:
            mask = tf.argmax(mask, axis=-1) # TODO decode one_hot mask
            mask = tf.expand_dims(mask, axis=-1)
            mask = tf.cast(mask, dtype=tf.float32)

            zeros, ones, runs = 0, 0, []
            for pixel in np.reshape(mask >= 0.5, [-1]):
                if pixel:
                    if zeros or (not zeros and not ones):
                        runs.append(zeros)
                        zeros = 0
                    ones += 1
                else:
                    if ones:
                        runs.append(ones)
                        ones = 0
                    zeros += 1
            runs.append(zeros + ones)
            print(*runs, file=predictions_file)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
