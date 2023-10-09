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

from cags_dataset import CAGS
import efficient_net

# TODO: Define reasonable defaults and optionally more parameters
parser = argparse.ArgumentParser()
parser.add_argument("--augment", default="", type=str, help="Augment data.")
parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
parser.add_argument("--epochs", default="frozen+cos:20:1e-3,finetune+cos:20:1e-4", type=str, help="Training epochs.")
parser.add_argument("--dropout", default=0., type=float, help="Dropout.")
parser.add_argument("--label_smoothing", default=0., type=float, help="Label smoothing.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")

def main(args):
    args.epochs = [(mode, int(epochs), float(lr)) for epoch in args.epochs.split(",") for mode, epochs, lr in [epoch.split(":")]]

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

    # Prepare data for classification
    def classification_dataset(example):
        return (example["image"], tf.one_hot(example["label"], len(CAGS.LABELS)))

    # Data augmentation
    generator = tf.random.Generator.from_seed(args.seed)
    def train_augment(image, label):
        if generator.uniform([]) >= 0.5:
            image = tf.image.flip_left_right(image)
        image = tf.image.resize_with_crop_or_pad(image, CAGS.H + 20, CAGS.W + 20)
        if "rotate" in args.augment:
            image = tfa.image.rotate(image, generator.uniform([], minval=np.deg2rad(-20), maxval=np.deg2rad(20)))
        image = tf.image.resize(image, [generator.uniform([], minval=CAGS.H, maxval=CAGS.H + 40, dtype=tf.int32),
                                        generator.uniform([], minval=CAGS.W, maxval=CAGS.W + 40, dtype=tf.int32)])
        image = tf.image.crop_to_bounding_box(
            image, target_height=CAGS.H, target_width=CAGS.W,
            offset_height=generator.uniform([], maxval=tf.shape(image)[0] - CAGS.H + 1, dtype=tf.int32),
            offset_width=generator.uniform([], maxval=tf.shape(image)[1] - CAGS.W + 1, dtype=tf.int32),
        )
        if "colors" in args.augment:
            image = tf.image.random_contrast(image, 0.8, 1.2)
            image = tf.image.random_hue(image, 0.2)
            image = tf.image.random_saturation(image, 0.8, 1.2)
            image = tf.image.random_brightness(image, 0.2)
        return image, label
    def train_mixup(images, labels):
        weight = generator.uniform([])
        return weight * images[0] + (1 - weight) * images[1], weight * labels[0] + (1 - weight) * labels[1]

    # Build the pipelines
    for dataset in ["train", "dev", "test"]:
        setattr(cags, dataset, getattr(cags, dataset).map(classification_dataset).cache())
    if "mixup" in args.augment:
        cags.train = cags.train.repeat(2)
    cags.train = cags.train.shuffle(10000, seed=args.seed)
    if "basic" in args.augment:
        cags.train = cags.train.map(train_augment)
    if "mixup" in args.augment:
        cags.train = cags.train.batch(2, drop_remainder=True).map(train_mixup)
    for dataset in ["train", "dev", "test"]:
        setattr(cags, dataset, getattr(cags, dataset).batch(args.batch_size).prefetch(tf.data.AUTOTUNE))
    # Iterate cags.train once to obtain cardinality.
    train_size = sum(1 for _ in cags.train)

    # Load the EfficientNet-B0 model
    efficientnet_b0 = efficient_net.pretrained_efficientnet_b0(include_top=False)

    # TODO: Create the model and train it
    inputs = tf.keras.layers.Input(shape=[CAGS.H, CAGS.W, CAGS.C])
    features, *_ = efficientnet_b0(inputs)
    features = tf.keras.layers.Dropout(args.dropout)(features)
    outputs = tf.keras.layers.Dense(len(CAGS.LABELS), activation=tf.nn.softmax)(features)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    tb_callback = tf.keras.callbacks.TensorBoard(args.logdir, write_graph=False, update_freq=100, profile_batch=0)
    tb_callback._close_writers = lambda: None # A hack allowing to keep the writers open.

    epochs = 0
    for stage, (mode, stage_epochs, stage_lr) in enumerate(args.epochs):
        efficientnet_b0.trainable = not mode.startswith("frozen")
        lr = tf.keras.experimental.CosineDecay(stage_lr, stage_epochs * train_size) if mode.endswith("cos") else stage_lr
        model.compile(
            optimizer=tf.optimizers.Adam(learning_rate=lr),
            loss=tf.losses.CategoricalCrossentropy(label_smoothing=args.label_smoothing),
            metrics=[tf.metrics.CategoricalAccuracy(name="accuracy")],
        )
        logs = model.fit(cags.train, epochs=epochs + stage_epochs, initial_epoch=epochs, validation_data=cags.dev, callbacks=[tb_callback])
        epochs += stage_epochs

    # Generate test set annotations, but in args.logdir to allow parallel execution.
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "cags_classification.txt"), "w", encoding="utf-8") as predictions_file:
        # TODO: Predict the probabilities on the test set
        test_probabilities = model.predict(cags.test)

        for probs in test_probabilities:
            print(np.argmax(probs), file=predictions_file)

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
