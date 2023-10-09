#!/usr/bin/env python3

# edbe2dad-018e-11eb-9574-ea7484399335
# 44752d3d-fdd8-11ea-9574-ea7484399335

import argparse
import datetime
import os
import re

from tensorflow.python.ops.gen_math_ops import sigmoid
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2") # Report only TF errors by default

import numpy as np
import tensorflow as tf

from uppercase_data import UppercaseData

# TODO: Set reasonable values for the hyperparameters, notably
# for `alphabet_size` and `window` and others.
parser = argparse.ArgumentParser()
parser.add_argument("--alphabet_size", default=None, type=int, help="If nonzero, limit alphabet to this many most frequent chars.")
parser.add_argument("--batch_size", default=None, type=int, help="Batch size.")
parser.add_argument("--epochs", default=None, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=8, type=int, help="Maximum number of threads to use.")
parser.add_argument("--window", default=None, type=int, help="Window size to use.")

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

    # Load data
    uppercase_data = UppercaseData(args.window, args.alphabet_size)

    # TODO: Implement a suitable model, optionally including regularization, select
    # good hyperparameters and train the model.
    #
    # The inputs are _windows_ of fixed size (`args.window` characters on left,
    # the character in question, and `args.window` characters on right), where
    # each character is represented by a `tf.int32` index. To suitably represent
    # the characters, you can:
    # - Convert the character indices into _one-hot encoding_. There is no
    #   explicit Keras layer, but you can
    #   - use a Lambda layer which can encompass any function:
    #       tf.keras.Sequential([
    #         tf.keras.layers.Input(shape=[2 * args.window + 1], dtype=tf.int32),
    #         tf.keras.layers.Lambda(lambda x: tf.one_hot(x, len(uppercase_data.train.alphabet))),
    #   - or use Functional API and then any TF function can be used
    #     as a Keras layer:
    #       inputs = tf.keras.layers.Input(shape=[2 * args.window + 1], dtype=tf.int32)
    #       encoded = tf.one_hot(inputs, len(uppercase_data.train.alphabet))
    #   You can then flatten the one-hot encoded windows and follow with a dense layer.
    # - Alternatively, you can use `tf.keras.layers.Embedding` (which is an efficient
    #   implementation of one-hot encoding followed by a Dense layer) and flatten afterwards.
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=[2 * args.window + 1], dtype=tf.int32),
        tf.keras.layers.Lambda(lambda x: tf.one_hot(x, len(uppercase_data.train.alphabet))),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dense(256, activation=tf.nn.relu),
        tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
    ])

    model.compile(
        optimizer=tf.optimizers.Adam(),
        loss=tf.losses.BinaryCrossentropy(),
        metrics=[tf.metrics.BinaryAccuracy("accuracy")],
    )

    print(uppercase_data.train.data['windows'].shape)

    model.fit(
        x=uppercase_data.train.data['windows'],
        y=uppercase_data.train.data['labels'],
        batch_size=args.batch_size,
        epochs=args.epochs
    )

    # TODO: Generate correctly capitalized test set.
    # Use `uppercase_data.test.text` as input, capitalize suitable characters,
    # and write the result to predictions_file (which is
    # `uppercase_test.txt` in the `args.logdir` directory).

    alphabet = uppercase_data.test.alphabet
    alphabet[0] = chr(0) # pad
    alphabet[1] = 1 # unknown
    unknown = 1
    test_text = chr(0)*args.window + uppercase_data.test.text + chr(0)*args.window
    windows = []
    for i, char in enumerate(uppercase_data.test.text):
        tti = i + args.window
        lower_lim = tti-args.window
        upper_lim = tti+1+args.window
        window = test_text[lower_lim:upper_lim]
        char_window = []
        for i,letter in enumerate(window):
            if letter in alphabet or letter == chr(0):
                char_window.append(alphabet.index(letter))
            else:
                char_window.append(unknown)
        windows.append(char_window)
    windows = tf.convert_to_tensor(windows, dtype=tf.int32)
    preds = model.predict(windows)
    new_text = []
    for i, p in enumerate(preds):
        if p > 0.5:
            new_text.append(uppercase_data.test.text[i].upper())
        else:
            new_text.append(uppercase_data.test.text[i])
    txt = ''.join(new_text)

    correct = uppercase_data.evaluate(uppercase_data.test, txt)

    print("Percent of correct letters:{}".format(correct))

    with open("uppercase_test.txt", "w", encoding="utf-8") as predictions_file:
        predictions_file.write(txt)

    #with open(os.path.join(args.logdir, "uppercase_test.txt"), "w", encoding="utf-8") as predictions_file:
    #    predictions_file.write(txt)

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
