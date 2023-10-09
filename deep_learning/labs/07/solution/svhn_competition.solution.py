#!/usr/bin/env python3
import argparse
import datetime
import os
import re
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2") # Report only TF errors by default

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

import bboxes_utils
import efficient_net
from svhn_dataset import SVHN

# TODO: Define reasonable defaults and optionally more parameters
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
parser.add_argument("--c", default=[4], type=int, nargs="+", help="C stages to use")
parser.add_argument("--epochs", default="frozen+cos:20:1e-3,finetune+cos:20:1e-4", type=str, help="Training epochs.")
parser.add_argument("--dropout", default=0., type=float, help="Dropout.")
parser.add_argument("--fpn", default=False, action="store_true", help="Use FPN.")
parser.add_argument("--head_channels", default=128, type=int, help="Channels in the head part.")
parser.add_argument("--head_layers", default=3, type=int, help="Layers in the head part.")
parser.add_argument("--iou_prediction", default=0.5, type=float, help="Prediction NMS IoU threshold.")
parser.add_argument("--iou_training", default=0.5, type=float, help="Training IoU threshold.")
parser.add_argument("--score_threshold", default=0.2, type=float, help="Score threshold.")
parser.add_argument("--scales", default=1, type=int, help="Anchor scales to use")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")

def main(args):
    args.c.sort(reverse=True)
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
    svhn = SVHN()

    # Create anchors on stages specified in args.c (C3, C4 and C5 are supported).
    anchors = []
    for c in args.c:
        N = 224 // (2 ** c)
        for y in range(N):
            cy = (y + 0.5) / N
            for x in range(N):
                cx = (x + 0.5) / N
                for i in range(args.scales):
                    s = 1/N * 2 * (2 ** ((i + 1) / args.scales))
                    anchors.append([cy - s, cx - s/2, cy + s, cx + s/2])
    anchors = np.array(anchors, np.float32)

    # Training and prediction data preparation
    bboxes_utils.BACKEND = tf
    def prepare_data(example):
        bboxes = example["bboxes"] / tf.cast(tf.shape(example["image"])[0], tf.float32)
        image = tf.image.resize(example["image"], [224, 224])

        anchor_classes, anchor_bboxes = bboxes_utils.bboxes_training(anchors, example["classes"], bboxes, args.iou_training)

        return image, (tf.one_hot(anchor_classes, SVHN.LABELS + 1)[:, 1:], anchor_bboxes), (1., tf.cast(anchor_classes > 0, tf.float32))

    # Build the pipelines
    for dataset in ["train", "dev", "test"]:
        data = getattr(svhn, dataset).map(prepare_data).prefetch(args.threads).cache()
        if dataset == "train":
            data = data.shuffle(len(data), seed=args.seed)
        globals()[dataset] = data.batch(args.batch_size)

    # Load the EfficientNet-B0 model
    efficientnet_b0 = efficient_net.pretrained_efficientnet_b0(include_top=False)

    # TODO: Create the model and train it

    # Create the model and start by running the EfficientNet.
    inputs = tf.keras.layers.Input(shape=[224, 224, 3])
    features = efficientnet_b0(inputs)
    stages = {5: features[1], 4: features[2], 3: features[3]}
    # If requrested, use FPN architecture to compute P3, P4 and P5
    if args.fpn:
        stages[5] = tf.keras.layers.Conv2D(args.head_channels, kernel_size=1)(stages[5])
        stages[5] = tf.keras.layers.BatchNormalization()(stages[5])
        stages[5] = tf.keras.layers.ReLU()(stages[5])
        for stage in [4, 3]:
            hidden = tf.keras.layers.Concatenate()([
                tf.keras.layers.Conv2D(args.head_channels, kernel_size=1)(stages[stage]),
                tf.keras.layers.UpSampling2D(2)(stages[stage + 1]),
            ])
            hidden = tf.keras.layers.BatchNormalization()(hidden)
            hidden = tf.keras.layers.ReLU()(hidden)
            hidden = tf.keras.layers.Conv2D(args.head_channels, kernel_size=3, padding="same")(hidden)
            hidden = tf.keras.layers.BatchNormalization()(hidden)
            hidden = tf.keras.layers.ReLU()(hidden)

            stages[stage] = hidden

    # Classification head
    classification_head = tf.keras.Sequential()
    for _ in range(args.head_layers):
        classification_head.add(tf.keras.layers.Conv2D(args.head_channels, 3, padding="same"))
        classification_head.add(tf.keras.layers.BatchNormalization())
        classification_head.add(tf.keras.layers.ReLU())
    classification_head.add(tf.keras.layers.Conv2D(SVHN.LABELS * args.scales, 3, padding="same", activation=tf.nn.sigmoid))

    # BBoxes head
    bboxes_head = tf.keras.Sequential()
    for _ in range(args.head_layers):
        bboxes_head.add(tf.keras.layers.Conv2D(args.head_channels, 3, padding="same"))
        bboxes_head.add(tf.keras.layers.BatchNormalization())
        bboxes_head.add(tf.keras.layers.ReLU())
    bboxes_head.add(tf.keras.layers.Conv2D(4 * args.scales, 3, padding="same"))

    # Apply the heads on the chosen stages
    classifications, bboxes = [], []
    for c in args.c:
        features = tf.keras.layers.Dropout(args.dropout)(stages[c])
        classifications.append(tf.keras.layers.Reshape([-1, SVHN.LABELS])(classification_head(features)))
        bboxes.append(tf.keras.layers.Reshape([-1, 4])(bboxes_head(features)))
    classifications = tf.keras.layers.Concatenate(axis=1)(classifications) if len(classifications) > 1 else classifications[0]
    bboxes = tf.keras.layers.Concatenate(axis=1)(bboxes) if len(bboxes) > 1 else bboxes[0]

    model = tf.keras.Model(inputs=inputs, outputs=(classifications, bboxes))

    tb_callback = tf.keras.callbacks.TensorBoard(args.logdir, write_graph=False, update_freq="epoch", profile_batch=0)
    tb_callback._close_writers = lambda: None # A hack allowing to keep the writers open.

    # The prediction function; using combined_non_max_suppression would be more efficient.
    def predict(dataset, original):
        predicted_classes, predicted_bboxes = [], []

        for classes, bboxes, example in zip(*model.predict(dataset), original):
            bboxes = tf.clip_by_value(bboxes_utils.bboxes_from_fast_rcnn(anchors, bboxes), 0, 1) * tf.cast(example["image"].shape[0], tf.float32)
            classes_probs = tf.math.reduce_max(classes, axis=1)
            classes = tf.math.argmax(classes, axis=1)
            indices = tf.image.non_max_suppression(bboxes, classes_probs, 5, args.iou_prediction, score_threshold=args.score_threshold)
            predicted_classes.append(tf.gather(classes, indices).numpy())
            predicted_bboxes.append(tf.gather(bboxes, indices).numpy())
        return predicted_classes, predicted_bboxes

    # Train according to the instructions in args.epochs.
    epochs = 0
    for stage, (mode, stage_epochs, stage_lr) in enumerate(args.epochs):
        efficientnet_b0.trainable = not mode.startswith("frozen")
        lr = tf.keras.experimental.CosineDecay(stage_lr, stage_epochs * len(train)) if mode.endswith("cos") else stage_lr
        model.compile(
            optimizer=tf.optimizers.Adam(learning_rate=lr),
            loss=(tfa.losses.SigmoidFocalCrossEntropy(reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE), tf.losses.Huber()),
        )
        for _ in range(stage_epochs):
            model.fit(train, epochs=epochs + 1, initial_epoch=epochs, callbacks=[tb_callback], verbose=2)
            epochs += 1

            accuracy = SVHN.evaluate(svhn.dev, list(zip(*predict(dev, svhn.dev))))
            print("Finished epoch {}, dev accuracy {:.2f}".format(epochs, accuracy), flush=True)
            tb_callback.on_epoch_end(epochs, {"val_accuracy": accuracy})

    # Generate test set annotations, but in args.logdir to allow parallel execution.
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "svhn_competition.txt"), "w", encoding="utf-8") as predictions_file:
        # TODO: Predict the digits and their bounding boxes on the test set.
        # Assume that for a single test image we get
        # - `predicted_classes`: a 1D array with the predicted digits,
        # - `predicted_bboxes`: a [len(predicted_classes), 4] array with bboxes;
        for predicted_classes, predicted_bboxes in zip(*predict(test, svhn.test)):
            output = []
            for label, bbox in zip(predicted_classes, predicted_bboxes):
                output += [label] + list(bbox)
            print(*output, file=predictions_file)

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
