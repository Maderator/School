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

from cags_dataset import CAGS
import efficient_net

# TODO: Define reasonable defaults and optionally more parameters
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
parser.add_argument("--epochs", default=2, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=0, type=int, help="Maximum number of threads to use.")

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

    # Load the EfficientNet-B0 model
    efficientnet_b0 = efficient_net.pretrained_efficientnet_b0(include_top=False)

    # TODO: Create the model and train it
    #for i in range(len(efficientnet_b0.layers)):
    #    efficientnet_b0.layers[i].trainable = False
    efficientnet_b0.trainable = False
    
    inputs = tf.keras.layers.Input(shape=[CAGS.H, CAGS.W, CAGS.C])
    #inputs = tf.keras.layers.Input(shape=[None, None, CAGS.C])
    
    efnet_model = efficientnet_b0(inputs)
    
    #hidden = tf.keras.layers.Dense(1000, activation=tf.nn.relu)(efnet_model[0])
    #hidden = tf.keras.layers.BatchNormalization()(efnet_model[0])
    hidden = tf.keras.layers.Dropout(0.5)(efnet_model[0])
    #print(CAGS.LABELS)
    #print(len(CAGS.LABELS))
    outputs = tf.keras.layers.Dense(units=len(CAGS.LABELS), activation="softmax")(hidden)
    
    #print(type(hidden))
    #print(hidden)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.summary()

    tb_callback = tf.keras.callbacks.TensorBoard(args.logdir, histogram_freq=1, update_freq=100, profile_batch=0)

    generator = tf.random.Generator.from_seed(args.seed)
    def train_augment(image, label):
        if generator.uniform([]) >= 0.5:
            image = tf.image.flip_left_right(image)
        
        image = tf.image.random_brightness(image, 0.3)
        image = tf.image.random_saturation(image, 0.5, 3)
        image = tf.image.random_contrast(image, 0.3, 0.6)
        
        image = tf.image.resize_with_crop_or_pad(image, CAGS.H + 6, CAGS.W + 6)
        image = tf.image.resize(image, [generator.uniform([], minval=CAGS.H, maxval=CAGS.H + 12, dtype=tf.int32),
                                        generator.uniform([], minval=CAGS.W, maxval=CAGS.W + 12, dtype=tf.int32)])
        image = tf.image.crop_to_bounding_box(
            image, target_height=CAGS.H, target_width=CAGS.W,
            offset_height=generator.uniform([], maxval=tf.shape(image)[0] - CAGS.H + 1, dtype=tf.int32),
            offset_width=generator.uniform([], maxval=tf.shape(image)[1] - CAGS.W + 1, dtype=tf.int32),
        )
        return image, label

    #frozen+cos 40, 0.001
    # finetune+cos, 40, 0.0001
    # dropout=0.5

    #train = tf.data.Dataset.from_tensor_slices( (cags.train.data["images"], cags.train.data["labels"]) )
    #dev = tf.data.Dataset.from_tensor_slices( (cags.dev.data["images"], cags.dev.data["labels"]) )
    train = cags.train.map(lambda example: (example["image"], example["label"]))
    train = train.map(train_augment)
    train = train.batch(args.batch_size)
    train = train.prefetch(tf.data.AUTOTUNE)

    dev = cags.dev.map(lambda example: (example["image"], example["label"]))
    dev = dev.batch(args.batch_size)
    dev = dev.prefetch(tf.data.AUTOTUNE)

    cos1 = tf.keras.experimental.CosineDecay(0.001, 100, alpha=0.1)

    model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=cos1),
        #optimizer=tf.optimizers.Adam(),
        loss=tf.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )

    model.fit(train, epochs=args.epochs, validation_data=dev, callbacks=[tb_callback])

    model.trainable = True

    cos2 = tf.keras.experimental.CosineDecay(0.0001, 100, alpha=0.1)
    model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=cos2),
        #optimizer=tf.optimizers.Adam(),
        loss=tf.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )

    model.fit(train, epochs=args.epochs, validation_data=dev, callbacks=[tb_callback])

    # Generate test set annotations, but in args.logdir to allow parallel execution.
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "cags_classification.txt"), "w", encoding="utf-8") as predictions_file:
        # TODO: Predict the probabilities on the test set
        test = cags.test.map(lambda example: (example["image"], example["label"])).batch(args.batch_size)
        test_probabilities = model.predict(test, batch_size=args.batch_size)

        for probs in test_probabilities:
            print(np.argmax(probs), file=predictions_file)
    
    with open(os.path.join(args.logdir, "dev_cags_classification.txt"), "w", encoding="utf-8") as predictions_file:
        dev_probabilities = model.predict(dev, batch_size=args.batch_size)

        for probs in dev_probabilities:
            print(np.argmax(probs), file=predictions_file)

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
