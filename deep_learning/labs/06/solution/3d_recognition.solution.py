#!/usr/bin/env python3
import argparse
import datetime
import os
import re
import sys
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2") # Report only TF errors by default

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from modelnet import ModelNet

# TODO: Define reasonable defaults and optionally more parameters
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
parser.add_argument("--depth", default=2, type=int, help="Blocks per stage")
parser.add_argument("--dropout", default=0., type=float, help="Dropout")
parser.add_argument("--epochs", default=50, type=int, help="Number of epochs.")
parser.add_argument("--label_smoothing", default=0., type=float, help="Label smoothing.")
parser.add_argument("--modelnet", default=20, type=int, help="ModelNet dimension.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
parser.add_argument("--width", default=1, type=int, help="Model width")

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
    modelnet = ModelNet(args.modelnet)

    # TODO: Create the model and train it
    for dataset in [modelnet.train, modelnet.dev, modelnet.test]:
        dataset.data["labels"] = tf.keras.utils.to_categorical(dataset.data["labels"], num_classes=len(ModelNet.LABELS))

    def cnn(inputs, filters, kernel_size, stride):
        return tf.keras.layers.Conv3D(filters, kernel_size=kernel_size, strides=stride, padding="same", use_bias=False)(inputs)
    def bn_activation(inputs):
        return tf.keras.layers.ReLU()(tf.keras.layers.BatchNormalization()(inputs))
    def block(inputs, filters, stride):
        hidden = bn_activation(inputs)
        hidden = cnn(hidden, filters, 3, stride)
        hidden = bn_activation(hidden)
        hidden = tf.keras.layers.Dropout(args.dropout)(hidden)
        hidden = cnn(hidden, filters, 3, 1)
        residual = inputs if stride == 1 and inputs.shape[-1] == filters else cnn(inputs, filters, 1, stride)
        return hidden + residual

    inputs = tf.keras.layers.Input([modelnet.D, modelnet.H, modelnet.W, modelnet.C])

    hidden = cnn(inputs, 16, 3, 1)
    for stage in range(3 if args.modelnet == 20 else 4):
        for layer in range(args.depth):
            hidden = block(hidden, 16 * args.width * (1 << stage), 2 if layer == 0 and stage > 0 else 1)
    hidden = bn_activation(hidden)
    hidden = tf.keras.layers.GlobalAvgPool3D()(hidden)
    outputs = tf.keras.layers.Dense(len(ModelNet.LABELS), activation=tf.nn.softmax)(hidden)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    tb_callback = tf.keras.callbacks.TensorBoard(args.logdir, write_graph=False, update_freq=100, profile_batch=0)

    lr = tf.keras.experimental.CosineDecay(0.1, args.epochs * modelnet.train.size / args.batch_size)
    wd = tf.keras.experimental.CosineDecay(1e-4, args.epochs * modelnet.train.size / args.batch_size)
    model.compile(
        optimizer=tfa.optimizers.SGDW(learning_rate=lr, weight_decay=wd, momentum=0.9, nesterov=True),
        loss=tf.losses.CategoricalCrossentropy(label_smoothing=args.label_smoothing),
        metrics=[tf.metrics.CategoricalAccuracy("accuracy")],
    )

    logs = model.fit(
        modelnet.train.data["voxels"], modelnet.train.data["labels"], batch_size=args.batch_size, epochs=args.epochs,
        validation_data=(modelnet.dev.data["voxels"], modelnet.dev.data["labels"]), callbacks=[tb_callback],
    )
    print("dev:{}".format(logs.history["val_accuracy"][-1]))

    # Generate test set annotations, but in args.logdir to allow parallel execution.
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "3d_recognition.txt"), "w", encoding="utf-8") as predictions_file:
        # TODO: Predict the probabilities on the test set
        test_probabilities = model.predict(modelnet.test.data["voxels"], batch_size=args.batch_size)

        for probs in test_probabilities:
            print(np.argmax(probs), file=predictions_file)

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
