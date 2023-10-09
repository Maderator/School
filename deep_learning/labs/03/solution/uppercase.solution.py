#!/usr/bin/env python3
import argparse
import datetime
import os
import re
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2") # Report only TF errors by default

import numpy as np
import tensorflow as tf

from uppercase_data import UppercaseData

# TODO: Set reasonable values for the hyperparameters, notably
# for `alphabet_size` and `window` and others.
parser = argparse.ArgumentParser()
parser.add_argument("--alphabet_size", default=0, type=int, help="If nonzero, limit alphabet to this many most frequent chars.")
parser.add_argument("--batch_size", default=2048, type=int, help="Batch size.")
parser.add_argument("--char_dropout", default=0.0, type=float, help="Char dropout.")
parser.add_argument("--decay", default=False, action="store_true", help="Learning decay.")
parser.add_argument("--dropout", default=0.0, type=float, help="Dropout rate.")
parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
parser.add_argument("--hidden_layers", default=[2048], nargs="*", type=int, help="Hidden layer sizes.")
parser.add_argument("--label_smoothing", default=0.0, type=float, help="Label smoothing.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=4, type=int, help="Maximum number of threads to use.")
parser.add_argument("--window", default=7, type=int, help="Window size to use.")

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
    #         tf.keras.layers.InputLayer(input_shape=[2 * args.window + 1], dtype=tf.int32),
    #         tf.keras.layers.Lambda(lambda x: tf.one_hot(x, len(uppercase_data.train.alphabet))),
    #   - or use Functional API and then any TF function can be used
    #     as a Keras layer:
    #       inputs = tf.keras.layers.Input(shape=[2 * args.window + 1], dtype=tf.int32)
    #       encoded = tf.one_hot(inputs, len(uppercase_data.train.alphabet))
    #   You can then flatten the one-hot encoded windows and follow with a dense layer.
    # - Alternatively, you can use `tf.keras.layers.Embedding` (which is an efficient
    #   implementation of one-hot encoding followed by a Dense layer) and flatten afterwards.
    inputs = tf.keras.layers.Input(shape=[2 * args.window + 1], dtype=tf.int32)
    mask = tf.keras.layers.Dropout(args.char_dropout)(tf.ones_like(inputs, tf.float32))
    dropped = tf.cast(mask != 0, tf.int32) * inputs + tf.cast(mask == 0, tf.int32) * 1
    embeddings = [tf.keras.layers.Embedding(len(uppercase_data.train.alphabet), 128)(dropped[:, i]) for i in range(2 * args.window + 1)]
    hidden = tf.keras.layers.Concatenate()(embeddings)
    for hidden_layer in args.hidden_layers:
        hidden = tf.keras.layers.Dense(hidden_layer, activation=tf.nn.relu)(hidden)
        hidden = tf.keras.layers.Dropout(args.dropout)(hidden)
    outputs = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)(hidden)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    lr = 0.001
    if args.decay: lr = tf.keras.experimental.CosineDecay(lr, len(uppercase_data.train.data["windows"]) * args.epochs // args.batch_size)
    model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=lr),
        loss=tf.losses.BinaryCrossentropy(label_smoothing=args.label_smoothing),
        metrics=[tf.metrics.BinaryAccuracy("accuracy")],
    )

    tb_callback=tf.keras.callbacks.TensorBoard(args.logdir, histogram_freq=0, update_freq=100, profile_batch=0)
    tb_callback._close_writers = lambda: None # A hack allowing to keep the writers open.
    model.fit(
        uppercase_data.train.data["windows"], uppercase_data.train.data["labels"],
        batch_size=args.batch_size, epochs=args.epochs,
        validation_data=(uppercase_data.dev.data["windows"], uppercase_data.dev.data["labels"]),
        callbacks=[tb_callback],
    )

    # TODO: Generate correctly capitalized test set.
    # Use `uppercase_data.test.text` as input, capitalize suitable characters,
    # and write the result to predictions_file (which is
    # `uppercase_test.txt` in the `args.logdir` directory).
    with open(os.path.join(args.logdir, "uppercase_test.txt"), "w", encoding="utf-8") as predictions_file:
        predicted = model.predict(uppercase_data.test.data["windows"], batch_size=args.batch_size)

        text = []
        for i in range(len(predicted)):
            char = uppercase_data.test.text[i].lower()
            if np.round(predicted[i]): char = char.upper()
            text.append(char)
        text = "".join(text)
        print(text, end="", file=predictions_file)

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
