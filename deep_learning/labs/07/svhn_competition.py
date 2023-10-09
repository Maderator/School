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

import bboxes_utils
import efficient_net
from svhn_dataset import SVHN

IMG_W = 224
IMG_H = 224
C = 4

# TODO: Define reasonable defaults and optionally more parameters
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=128, type=int, help="Batch size.")
parser.add_argument("--epochs", default=40, type=int, help="Number of epochs.")
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
class RetinaNetLike(Model):
    def _cnn(self, inputs, args, filters, kernel_size, stride, activation, pad="same"):
        hidden = tf.keras.layers.Conv2D(filters, kernel_size=kernel_size, strides=stride, padding=pad, use_bias=False)(inputs)
        hidden = tf.keras.layers.BatchNormalization()(hidden)
        hidden = self._activation(hidden, args) if activation else hidden
        #hidden = tf.keras.layers.Dropout(0.3)(hidden)
        return hidden
        
    def _classes_predictor(self, inputs, args):
        hidden = self._cnn(inputs, args, 128, 3, 1, activation=True)
        hidden = self._cnn(hidden, args, 128, 3, 1, activation=True)
        hidden = self._cnn(hidden, args, 128, 3, 1, activation=True)
        output = tf.keras.layers.Conv2D(10, 1, 1, use_bias=False, activation=tf.keras.activations.sigmoid)(hidden)
        return output

    def _bboxes_predictor(self, inputs, args):
        hidden = self._cnn(inputs, args, 128, 3, 1, activation=True)
        hidden = self._cnn(hidden, args, 128, 3, 1, activation=True)
        hidden = self._cnn(hidden, args, 128, 3, 1, activation=True)
        output = tf.keras.layers.Conv2D(4, 1, 1, use_bias=False, activation=None)(hidden)
        return output

    def create_optimizer(self, args, dataset, lr_value=0.001):
        # Decay
        training_batches = args.epochs * len(list(dataset.train)) // args.batch_size
        if args.decay == "piecewise":
            decay_fn = lambda value: tf.optimizers.schedules.PiecewiseConstantDecay(
                [int(0.5 * training_batches), int(0.75 * training_batches)],
                [value, value / 10, value / 100])
        elif args.decay == "cosine":
            decay_fn = lambda value: tf.keras.experimental.CosineDecay(value, training_batches, alpha = 0.01)
        else:
            raise ValueError("Uknown decay '{}'".format(args.decay))
        learning_rate = decay_fn(lr_value)
        weight_decay = decay_fn(1e-4)

        # Optimizer
        if args.optimizer == "RMSProp":
            optimizer = tfa.optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay, beta_2=0.9, epsilon=1e-3)
        elif args.optimizer == "Adam":
            optimizer = tfa.optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay, epsilon=1e-3)
        else:
            raise ValueError("Uknown optimizer '{}'".format(args.optimizer))
        optimizer = tf.optimizers.Adam(learning_rate=learning_rate, beta_2=0.9, epsilon=1e-3)
        return optimizer


    def __init__(self, args, c=4):
        # c is one of values 3,4,5
        ci = 6 - c # convert c to index in efficient net outputs

        # Load the EfficientNet-B0 model
        efficientnet_b0 = efficient_net.pretrained_efficientnet_b0(include_top=False, dynamic_input_shape=False)

        efficientnet_b0.trainable = False
    
        img_in = tf.keras.Input(shape=[IMG_W, IMG_H,3], dtype=tf.float32)
    
        inputs = {
            "image":img_in,
        }

        efnet_model = efficientnet_b0(img_in)
            
        cl_sc_in = tf.keras.Input(shape=[14,14,112], dtype=tf.float32)
        classes_score = self._classes_predictor(cl_sc_in, args) # WxHxClasses
        cl_sc_model = tf.keras.Model(inputs=cl_sc_in, outputs=classes_score)


        bb_pred_in = tf.keras.Input(shape=[14,14,112], dtype=tf.float32)
        bboxes_pred = self._bboxes_predictor(bb_pred_in, args) # WxHx4
        bb_pred_model = tf.keras.Model(inputs=bb_pred_in, outputs=bboxes_pred)

        common_in = tf.keras.Input(shape=[14,14,112], dtype=tf.float32)
        bb_pred = bb_pred_model(common_in)
        cl_sc = cl_sc_model(common_in)
        outputs = {
            "anchor_classes" : cl_sc,
            "anchor_bboxes" : bb_pred
        }

        heads = tf.keras.Model(inputs=common_in, outputs=outputs)

        out = heads(efnet_model[ci])

        super().__init__(inputs, out)

def map_dataset(dataset, anchors):
    dataset = dataset.map(lambda example: {
        "image":example["image"],
        "anchors":bboxes_utils.bboxes_training(anchors, example["classes"], example["bboxes"], 0.5)}
    )
    dataset = dataset.map(lambda example: {
        "image":example["image"],
        "anchor_classes":example["anchors"][0],
        "anchor_bboxes":example["anchors"][1]}
    )

    def reshape_anchors(example):
        anch_classes = example["anchor_classes"]
        anch_bboxes = example["anchor_bboxes"]
        nw_anchors = tf.sqrt(tf.cast(tf.shape(anch_classes)[0], tf.float32))
        anch_classes = tf.reshape(anch_classes, [nw_anchors, nw_anchors])
        anch_bboxes = tf.reshape(anch_bboxes, [nw_anchors, nw_anchors, 4])
        
        ex = {"image":example["image"],
        "anchor_classes":anch_classes,
        "anchor_bboxes":anch_bboxes}
        return ex
    
    dataset = dataset.map(reshape_anchors)

    return dataset
    

def prepare_dataset(args, svhn, c=4, scale=2):
    height = scale * 4 * (2 ** c)
    height = height / IMG_H
    width = height/2
    # TODO 1. create anchors
    nw_anchors = int(IMG_W / (2 ** c))
    nh_anchors = int(IMG_H / (2 ** c))
    anch_w_center_diff = 1.0 / nw_anchors
    anch_h_center_diff = 1.0 / nh_anchors
    anchors = []
    for h in range(nh_anchors):
        for w in range(nw_anchors):
            center_h = anch_h_center_diff*h + anch_h_center_diff / 2.0
            center_w = anch_w_center_diff*w + anch_w_center_diff / 2.0

            top = center_h - height / 2.0
            bottom = top + height
            left = center_w - width / 2.0
            right = left + width
            anchors.append([top, left, bottom, right])
    anchors = np.array(anchors, dtype=np.float32)

    # TODO 2. scale bboxes
    def scale_bboxes(example):
        bboxes = example["bboxes"]
        img_w = tf.cast(tf.shape(example["image"])[0], tf.float32)
        bboxes = tf.divide(bboxes, img_w)
        return {"image":example["image"],
                "classes":example["classes"], 
                "bboxes":bboxes}
        
    train = svhn.train.map(scale_bboxes)
    dev = svhn.dev.map(scale_bboxes)
    
    # TODO 3. use bboxes_training

    #bboxes - bboxes from bboxes_training
    # weights from bboxes_training anchor_classes 1 for nonzero class 0 for zero classes

    train = map_dataset(train, anchors)#.take(64*10) # TODO delete
    dev = map_dataset(dev, anchors)#.take(64*10)

    #train = train.padded_batch(args.batch_size, padded_shapes={"classes":5, "image":[293,293,3], "bboxes":[5,4], "anchor_classes":[14,14], "anchor_bboxes":[14,14,4]})
    #test = test.padded_batch(args.batch_size, padded_shapes={"classes":5, "image":[293,293,3], "bboxes":[5,4], "anchor_classes":[14,14], "anchor_bboxes":[14,14,4]})
    #dev = dev.padded_batch(args.batch_size, padded_shapes={"classes":5, "image":[293,293,3], "bboxes":[5,4], "anchor_classes":[14,14], "anchor_bboxes":[14,14,4]})
    def create_element(element):
        in_el = tf.image.resize(element["image"], [224,224])
        #in_el = tf.image.crop_to_bounding_box(element["image"], 0, 0, 224, 224)
        ac_zeros = tf.cast(tf.equal(element["anchor_classes"], tf.constant(0, dtype=tf.int64)), dtype=tf.int64)
        ac_minus_one = tf.subtract(element["anchor_classes"], tf.constant(1, dtype=tf.int64))
        #ac_zeros = tf.multiply(ac_zeros, 5)
        #ac_final = tf.add(ac_minus_one, ac_zeros) # TODO use this
        ac_final = ac_minus_one

        oh_anchor_classes = tf.one_hot(ac_final, depth=10)

        out_el = {
            "anchor_classes":oh_anchor_classes,
            "anchor_bboxes":element["anchor_bboxes"]
        }
        sample_weights = tf.cast(tf.not_equal(element["anchor_classes"], tf.constant(0, dtype=tf.int64)), dtype=tf.float32)

        sw = {
            #"anchor_classes":sample_weights,
            "anchor_bboxes":sample_weights
        }

        return (in_el, out_el, sw)

    train = train.map(create_element)
    #test = test.map(create_element)
    dev = dev.map(create_element)

    train = train.batch(args.batch_size)
    dev = dev.batch(args.batch_size)

    return train, dev, anchors


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
    svhn = SVHN()

    train, dev, anchors = prepare_dataset(args, svhn) # TODO try to change scale to 2


    #import sys
    #np.set_printoptions(threshold=sys.maxsize)    
    #ab = [e[1]["anchor_bboxes"] for e in train]
    #print(ab[0][0])     
    #ac = [np.argmax(e[1]["anchor_classes"],-1) for e in train]     
    #sw = [e[2] for e in train]      
    #for i in range(10):
    #    for j in range(14):
    #        print(ac[0][i][j], sw[0][i][j])

    model = RetinaNetLike(args, c=C)

    tb_callback = tf.keras.callbacks.TensorBoard(args.logdir, histogram_freq=1, update_freq=100, profile_batch=0)

    generator = tf.random.Generator.from_seed(args.seed)

    optimizer = model.create_optimizer(args, svhn, lr_value = 0.001)

    model.compile(
        optimizer=optimizer,
        loss={
            "anchor_classes":tfa.losses.SigmoidFocalCrossEntropy(reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE),
            "anchor_bboxes":tf.losses.Huber(),
        },
        weighted_metrics={
            "anchor_classes":[tf.metrics.CategoricalCrossentropy(name="cat_crossentropy")],
            "anchor_bboxes":[tf.metrics.MeanSquaredError(name="mse")],
        } # TODO
        #metrics=[tf.metrics.CategoricalAccuracy()],
    )

    model.fit(train, epochs=args.epochs, validation_data=dev, callbacks=[tb_callback])

    #model.trainable = True

    #optimizer = model.create_optimizer(args, svhn, lr_value=0.0001)

    #model.compile(
    #    optimizer=optimizer,
    #    loss={
    #        "anchor_classes":tfa.losses.SigmoidFocalCrossEntropy(reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE),
    #        "anchor_bboxes":tf.losses.Huber(),
    #    },
    #    weighted_metrics={
    #        "anchor_classes":[tf.metrics.CategoricalCrossentropy(name="cat_crossentropy")],
    #        "anchor_bboxes":[tf.metrics.MeanSquaredError(name="mse")],
    #    } # TODO
    #    #metrics=[tf.metrics.CategoricalAccuracy()],
    #)

    #model.fit(train, epochs=args.epochs, validation_data=dev, callbacks=[tb_callback])

    # Generate test set annotations, but in args.logdir to allow parallel execution.
    os.makedirs(args.logdir, exist_ok=True)
    #with open(os.path.join(args.logdir, "svhn_competition.txt"), "w", encoding="utf-8") as predictions_file:
    with open("svhn_competition.txt", "w", encoding="utf-8") as predictions_file:
        # TODO: Predict the digits and their bounding boxes on the test set.
        # Assume that for a single test image we get
        # - `predicted_classes`: a 1D array with the predicted digits,
        # - `predicted_bboxes`: a [len(predicted_classes), 4] array with bboxes;
        
        #import scipy.stats
        #test = svhn.test.padded_batch(args.batch_size, padded_shapes={"classes":5, "image":[300,300,3], "bboxes":[5,4]})
        def create_element(element):
            in_el = tf.image.resize(element["image"], [224,224])
            #in_el = tf.image.crop_to_bounding_box(element["image"], 0, 0, 224, 224)
            return in_el

        test = svhn.test.map(create_element).batch(args.batch_size)

        out = model.predict(test, batch_size = args.batch_size)
        anch_classes_all = out["anchor_classes"]
        fast_rcnn_all = out["anchor_bboxes"] # actually fast_rcnn
        
        images_size = [e["image"].shape[0] for e in svhn.test]

        nw_anchors = 14
        nh_anchors = 14 
        reshaped_ac_all = []
        reshaped_ab_all = []
        for i in range(len(anch_classes_all)):
            reshaped_ac = []
            reshaped_fr = []
            for h in range(nh_anchors):
                for w in range(nw_anchors):
                    reshaped_ac.append(anch_classes_all[i,h,w])
                    reshaped_fr.append(fast_rcnn_all[i,h,w])
            
            reshaped_fr = np.array(reshaped_fr)
            bboxes = bboxes_utils.bboxes_from_fast_rcnn(anchors, reshaped_fr)
            bboxes = tf.math.multiply(bboxes, images_size[i])
            reshaped_ac_all.append(reshaped_ac)
            reshaped_ab_all.append(bboxes)
        reshaped_ac_all = np.array(reshaped_ac_all)
        reshaped_ab_all = np.array(reshaped_ab_all)

        for i in range(len(reshaped_ac_all)):
        #for predicted_classes, predicted_bboxes, valid_detections in zip(nms_classes, nms_boxes, nms_valid_det):
            classes = reshaped_ac_all[i]
            score = np.max(classes, axis=-1)
            boxes = reshaped_ab_all[i]

            chosen = np.ones(len(classes))
            for j in range(len(boxes)):
                if (boxes[j][0] < 0 or boxes[j][1] < 0 or boxes[j][2]<0 or boxes[j][3] < 0
                   or boxes[j][0] > images_size[i] or boxes[j][1] > images_size[i] or boxes[j][2] > images_size[i] or boxes[j][3] > images_size[i]):
                    chosen[j] = 0
            #boxes = np.ma.masked_array(boxes, np.repeat(chosen[:, np.newaxis], 4, axis=1))
            #score = np.ma.masked_array(score, chosen)
            #classes = np.ma.masked_array(classes, np.repeat(chosen[:, np.newaxis], 10, axis=1))
            nb, ns, nc = [], [], []
            ch_n = 0
            for j in range(len(chosen)):
                if chosen[j] == 1:
                    nb.append(boxes[j,:])
                    ns.append(score[j])
                    nc.append(classes[j,:])
                    ch_n += 1
            if ch_n == 0:
                print(*[], file=predictions_file)
                continue
            boxes, score, classes = np.array(nb), np.array(ns), np.array(nc)
            selected_indices = tf.image.non_max_suppression(boxes, score, max_output_size=3, iou_threshold=0.40,
                        score_threshold=0.2)

            #max_cl = np.argmax(classes, axis=-1)
            #print(max_cl)

            output = []
            predicted_bboxes = boxes[selected_indices,:]
            predicted_bboxes = predicted_bboxes.astype('int32')
            predicted_classes = np.argmax(classes[selected_indices,:], axis=-1)
            for label, bbox in zip(predicted_classes, predicted_bboxes):
                label = int(label)
                output += [label] + list(bbox) # TODO label-1 and if 0 (0-1) then do not add to output
            print(*output, file=predictions_file)

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
