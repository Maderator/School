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

from cifar10 import CIFAR10

# TODO: Define reasonable defaults and optionally more parameters
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
parser.add_argument("--epochs", default=300, type=int, help="Number of epochs.")
parser.add_argument("--learning_rate", default=0.001, type=float, help="Learning rate.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=0, type=int, help="Maximum number of threads to use.")

def main(args):
    # Fix random seeds and threads
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Create logdir name
    args.logdir = os.path.join("logs", "last")
    #args.logdir = os.path.join("logs", "{}-{}-{}".format(
    #    os.path.basename(globals().get("__file__", "notebook")),
    #    datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
    #    ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    #))

    # Load data
    cifar = CIFAR10()

    def create_resnet_layer(hidden, filters=64, kernel_size=3, stride=1, residual_connection=True):
        hidden_in = hidden
        hidden = tf.keras.layers.Conv2D(filters, kernel_size, stride, "same", use_bias=False, activation=None)(hidden)
        hidden = tf.keras.layers.BatchNormalization()(hidden)
        hidden = tf.keras.activations.relu(hidden)
        hidden = tf.keras.layers.Dropout(0.3)(hidden)
        hidden = tf.keras.layers.Conv2D(filters, kernel_size, stride, "same", use_bias=False, activation=None)(hidden)
        hidden = tf.keras.layers.BatchNormalization()(hidden)
        hidden = tf.keras.activations.relu(hidden)
        hidden = tf.keras.layers.Dropout(0.3)(hidden)
        if residual_connection:
            hidden = tf.keras.layers.Add()([hidden, hidden_in])
        return hidden

    # TODO: Create the model and train it
    inputs = tf.keras.layers.Input(shape=[CIFAR10.H, CIFAR10.W, CIFAR10.C])
    hidden = tf.keras.layers.Conv2D(32, 7, 2, "same", use_bias=False, activation=None)(inputs)
    hidden = tf.keras.layers.BatchNormalization()(hidden)
    hidden = tf.keras.activations.relu(hidden)
    hidden = tf.keras.layers.Dropout(0.3)(hidden)
    hidden = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding="same")(hidden)
    hidden = create_resnet_layer(hidden, filters=32, kernel_size=3, stride=2, residual_connection=False)
    hidden = create_resnet_layer(hidden, filters=32, kernel_size=3, stride=2)
    hidden = create_resnet_layer(hidden, filters=32, kernel_size=3, stride=2)
    hidden = create_resnet_layer(hidden, filters=64, kernel_size=3, stride=2, residual_connection=False)
    hidden = create_resnet_layer(hidden, filters=64, kernel_size=3, stride=2)
    hidden = create_resnet_layer(hidden, filters=64, kernel_size=3, stride=2)
    hidden = create_resnet_layer(hidden, filters=64, kernel_size=3, stride=2)
    hidden = create_resnet_layer(hidden, filters=64, kernel_size=3, stride=2)
    hidden = create_resnet_layer(hidden, filters=64, kernel_size=3, stride=2)
    hidden = tf.keras.layers.AveragePooling2D(padding="same")(hidden)
    hidden = tf.keras.layers.Flatten()(hidden)
    hidden = tf.keras.layers.Dense(1000, activation=tf.nn.relu)(hidden)
    hidden = tf.keras.layers.Dropout(0.3)(hidden)
    outputs = tf.keras.layers.Dense(CIFAR10.LABELS, activation=tf.nn.softmax)(hidden)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    print(model.summary())

    ds = 10
    lr = tf.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=args.learning_rate,
        decay_steps=ds,
        end_learning_rate=args.learning_rate/10)

    optimizer = tf.optimizers.Adam(learning_rate=lr)

    model.compile(
        optimizer=optimizer,
        loss=tf.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )

    checkpoint_path = "logs/checkpoints/cp-{epoch:04d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    print(checkpoint_dir)

    if True:
        latest = tf.train.latest_checkpoint(checkpoint_dir)
        print("latest checkpoint=", latest)
        model.load_weights(latest)
 
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 save_freq=2000,
                                                 verbose=1)

    tb_callback = tf.keras.callbacks.TensorBoard(args.logdir, histogram_freq=1, update_freq=100, profile_batch=0)

    generator = tf.random.Generator.from_seed(args.seed)
    def train_augment(image, label):
        if generator.uniform([]) >= 0.5:
            image = tf.image.flip_left_right(image)
        image = tf.image.resize_with_crop_or_pad(image, CIFAR10.H + 6, CIFAR10.W + 6)
        image = tf.image.resize(image, [generator.uniform([], minval=CIFAR10.H, maxval=CIFAR10.H + 12, dtype=tf.int32),
                                        generator.uniform([], minval=CIFAR10.W, maxval=CIFAR10.W + 12, dtype=tf.int32)])
        image = tf.image.crop_to_bounding_box(
            image, target_height=CIFAR10.H, target_width=CIFAR10.W,
            offset_height=generator.uniform([], maxval=tf.shape(image)[0] - CIFAR10.H + 1, dtype=tf.int32),
            offset_width=generator.uniform([], maxval=tf.shape(image)[1] - CIFAR10.W + 1, dtype=tf.int32),
        )
        return image, label
    
    if False:
        train = tf.data.Dataset.from_tensor_slices( (cifar.train.data["images"], cifar.train.data["labels"]) )
        dev = tf.data.Dataset.from_tensor_slices( (cifar.dev.data["images"], cifar.dev.data["labels"]) )

        #train = train.take(5000).shuffle(5000, seed=args.seed).map(train_augment).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
        train = train.map(train_augment).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
        dev = dev.batch(args.batch_size)
        model.fit(train, epochs=args.epochs, validation_data=dev, callbacks=[tb_callback, cp_callback])
    else:
        train = cifar.train.data
        dev = cifar.dev.data

        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range = 30, 
                zoom_range = [0.8,1.5],
                horizontal_flip=True
            )
        datagen.fit(train["images"])

        #train_generator = datagen.flow(train["images"],train["labels"], batch_size=args.batch_size)
        #validation_generator = datagen.flow(dev["images"],dev["labels"], batch_size=args.batch_size)

        model.fit(
            datagen.flow(train["images"],train["labels"], batch_size=args.batch_size),
            epochs = args.epochs, 
            batch_size = args.batch_size,
            validation_data = datagen.flow(dev["images"],dev["labels"], batch_size=args.batch_size), 
            callbacks=[tb_callback, cp_callback]
        )

    # Generate test set annotations, but in args.logdir to allow parallel execution.
    with open(os.path.join(args.logdir, "cifar_competition_test.txt"), "w", encoding="utf-8") as predictions_file:
        for probs in model.predict(cifar.test.data["images"], batch_size=args.batch_size):
            print(np.argmax(probs), file=predictions_file)
    
    with open(os.path.join(args.logdir, "cifar_competition_dev.txt"), "w", encoding="utf-8") as predictions_file:
        for probs in model.predict(cifar.dev.data["images"], batch_size=args.batch_size):
            print(np.argmax(probs), file=predictions_file)

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
