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

from modelnet import ModelNet

# TODO: Define reasonable defaults and optionally more parameters
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
parser.add_argument("--modelnet", default=32, type=int, help="ModelNet dimension.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")

def init_model(modelnet):
    inputs = tf.keras.layers.Input(shape=[modelnet.H, modelnet.W, modelnet.D, modelnet.C])

    hidden = tf.keras.layers.Conv3D(filters=32, kernel_size=3, padding="valid", activation=None)(inputs)
    hidden = tf.keras.layers.BatchNormalization()(hidden)
    hidden = tf.keras.layers.Activation(tf.nn.relu)(hidden)
    hidden = tf.keras.layers.Dropout(0.5, noise_shape=None, seed=None)(hidden)

    hidden = tf.keras.layers.Conv3D(filters=32 ,kernel_size=3, padding="valid", activation=None)(hidden)
    hidden = tf.keras.layers.BatchNormalization()(hidden)
    hidden = tf.keras.layers.Activation(tf.nn.relu)(hidden)
    hidden = tf.keras.layers.Dropout(0.5, noise_shape=None, seed=None)(hidden)
    
    hidden = tf.keras.layers.Conv3D(filters=64, kernel_size=3, strides=2, padding="valid", activation=None)(hidden)
    hidden = tf.keras.layers.BatchNormalization()(hidden)
    hidden = tf.keras.layers.Activation(tf.nn.relu)(hidden)
    hidden = tf.keras.layers.Dropout(0.5, noise_shape=None, seed=None)(hidden)
    
    hidden = tf.keras.layers.Conv3D(filters=64, kernel_size=3, padding="valid", activation=None)(hidden)
    hidden = tf.keras.layers.BatchNormalization()(hidden)
    hidden = tf.keras.layers.Activation(tf.nn.relu)(hidden)
    
    hidden = tf.keras.layers.Conv3D(filters=64, kernel_size=3, padding="valid", activation=None)(hidden)
    hidden = tf.keras.layers.BatchNormalization()(hidden)
    hidden = tf.keras.layers.Activation(tf.nn.relu)(hidden)
    hidden = tf.keras.layers.Dropout(0.5, noise_shape=None, seed=None)(hidden)
    
    hidden = tf.keras.layers.Conv3D(filters=64, kernel_size=3, strides=2, padding="valid", activation=None)(hidden)
    hidden = tf.keras.layers.BatchNormalization()(hidden)
    hidden = tf.keras.layers.Activation(tf.nn.relu)(hidden)
    
    hidden = tf.keras.layers.Flatten()(hidden)
    hidden = tf.keras.layers.Dropout(0.5, noise_shape=None, seed=None)(hidden)
    
    hidden = tf.keras.layers.Dense(units=128, activation="relu")(hidden)

    outputs = tf.keras.layers.Dense(units=len(modelnet.LABELS), activation="softmax")(hidden)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

def train_model(args, model, modelnet):
    tb_callback = tf.keras.callbacks.TensorBoard(args.logdir, histogram_freq=1, update_freq=100, profile_batch=0)

    checkpoint_path = "logs/checkpoints/cp-{epoch:04d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    print(checkpoint_dir)

    if False:
        latest = tf.train.latest_checkpoint(checkpoint_dir)
        print("latest checkpoint=", latest)
        model.load_weights(latest)
 
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 save_freq=2000,
                                                 verbose=1)

    model.compile(
        optimizer=tf.optimizers.Adam(),
        loss=tf.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    tb_callback = tf.keras.callbacks.TensorBoard(args.logdir, histogram_freq=1, update_freq=100, profile_batch=0)
    
    train = tf.data.Dataset.from_tensor_slices( (modelnet.train.data["voxels"], modelnet.train.data["labels"]) )
    dev = tf.data.Dataset.from_tensor_slices( (modelnet.dev.data["voxels"], modelnet.dev.data["labels"]) )

    #train = train.map(train_augment).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
    train = train.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
    dev = dev.batch(args.batch_size)
    model.fit(train, epochs=args.epochs, validation_data=dev, callbacks=[tb_callback, cp_callback])
    return model

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

    model = init_model(modelnet)
    model.summary()

    model = train_model(args, model, modelnet)

    # Generate test set annotations, but in args.logdir to allow parallel execution.
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "3d_recognition.txt"), "w", encoding="utf-8") as predictions_file:
        # TODO: Predict the probabilities on the test set
        print(args.batch_size)
        test_probabilities = model.predict(modelnet.test.data["voxels"], batch_size=args.batch_size)

        for probs in test_probabilities:
            print(np.argmax(probs), file=predictions_file)

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
