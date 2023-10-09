#!/usr/bin/env python3
import argparse
import datetime
import os
import re
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2") # Report only TF errors by default

import numpy as np
import tensorflow as tf

from electra_czech_small_lc import ElectraCzechSmallLc
from text_classification_dataset import TextClassificationDataset

# TODO: Define reasonable defaults and optionally more parameters
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
parser.add_argument("--epochs", default=20, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=0, type=int, help="Maximum number of threads to use.")

def labels_to_oh(labels):
    oh = []
    for label in labels:
        if label == "n":
            oh.append([1,0,0])
        elif label == "0":
            oh.append([0,1,0])
        else:
            oh.append([0,0,1])
    return tf.constant(oh)

def oh_to_labels(oh):
    labels = []
    oh_argmax = tf.argmax(oh["output"], axis=1)
    for idx in oh_argmax:
        if idx == 0:
            labels.append("n")
        elif idx == 1:
            labels.append("0")
        else:
            labels.append("p")
    return labels

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

    # Load the Electra Czech small lowercased
    electra = ElectraCzechSmallLc()

    # TODO: Load the data. Consider providing a `tokenizer` to the
    # constructor of the TextClassificationDataset.
    tokenizer = electra.create_tokenizer()
    facebook =  TextClassificationDataset("czech_facebook", tokenizer=tokenizer.encode)
    
    def create_dataset(data, train, max_length):
        num_tokens = len(data["tokens"])
        ids = np.zeros([num_tokens, max_length], dtype=np.int32)
        masks = np.zeros([num_tokens, max_length], dtype=np.int32)
        for i in range(num_tokens):
            ids[i, :len(data["tokens"][i])] = data["tokens"][i]
            masks[i, :len(data["tokens"][i])] = 1
        return (ids,masks, data["labels"])

    max_length_train = max(len(tokens) for tokens in facebook.train.data["tokens"])
    max_length_dev = max(len(tokens) for tokens in facebook.dev.data["tokens"])
    max_length_test = max(len(tokens) for tokens in facebook.test.data["tokens"])
    max_length = max(max_length_train, max_length_dev, max_length_test)
    train, dev, test = (create_dataset(facebook.train.data, True, max_length), 
                        create_dataset(facebook.dev.data, False, max_length), 
                        create_dataset(facebook.test.data, False, max_length))
    

    oh_labels_train = labels_to_oh(train[2])
    oh_labels_dev = labels_to_oh(dev[2])

    # TODO: Create the model and train it
    model = electra.create_model()
    model.trainable = False

    tokens = tf.keras.layers.Input(shape=[max_length], dtype=tf.int32)
    masks = tf.keras.layers.Input(shape=[max_length], dtype=tf.int32)
    inputs = {"input_ids":tokens, "attention_mask":masks}
    bert_out = model(inputs)
    x = tf.keras.layers.Flatten()(bert_out.last_hidden_state)                            
    x = tf.keras.layers.Dropout(0.5)(x)
    out = tf.keras.layers.Dense(2048, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    out = tf.keras.layers.Dense(3, activation=tf.nn.softmax)(x)
    outputs = {"output":out}

    whole_model = tf.keras.Model(inputs, outputs)


    train_data = ({"input_ids":train[0], "attention_mask":train[1]}, {"output":oh_labels_train})
    dev_data = ({"input_ids":dev[0], "attention_mask":dev[1]}, {"output":oh_labels_dev})
    test_data = ({"input_ids":test[0], "attention_mask":test[1]})

    train_dataset = tf.data.Dataset.from_tensor_slices(train_data).batch(args.batch_size)
    train_dataset = train_dataset.shuffle(len(train_data[0]["input_ids"]), seed=args.seed, reshuffle_each_iteration=True)

    val_dataset = tf.data.Dataset.from_tensor_slices(dev_data).batch(args.batch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices(test_data).batch(args.batch_size)
    
    training_batches = args.epochs * len(train[0])
    decay_fn = lambda value: tf.keras.experimental.CosineDecay(value, decay_steps=training_batches, alpha = 0.01)
    learning_rate = decay_fn(1e-3)
    optimizer = tf.optimizers.Adam(learning_rate=learning_rate, beta_2=0.9, epsilon=1e-3)
    whole_model.compile(optimizer=optimizer,
                    metrics=[tf.keras.metrics.CategoricalCrossentropy()],
                    loss=tf.keras.losses.CategoricalCrossentropy())

    tb_callback = tf.keras.callbacks.TensorBoard(args.logdir, update_freq=100, profile_batch=0)
    tb_callback._close_writers = lambda: None # A hack allowing to keep the writers open.
    
    whole_model.fit(train_dataset, batch_size=args.batch_size, epochs=args.epochs, 
                      validation_data=val_dataset, callbacks=[tb_callback])

    whole_model.trainable = True

    learning_rate = decay_fn(4e-5)
    optimizer = tf.optimizers.Adam(learning_rate=learning_rate, beta_2=0.9, epsilon=1e-3)
    whole_model.compile(optimizer=optimizer,
                    metrics=[tf.keras.metrics.CategoricalCrossentropy()],
                    loss=tf.keras.losses.CategoricalCrossentropy())

    tb_callback = tf.keras.callbacks.TensorBoard(args.logdir, update_freq=100, profile_batch=0)
    tb_callback._close_writers = lambda: None # A hack allowing to keep the writers open.
    
    whole_model.fit(train_dataset, batch_size=args.batch_size, epochs=10, 
                      validation_data=val_dataset, callbacks=[tb_callback])

    # Generate test set annotations, but in args.logdir to allow parallel execution.
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "sentiment_analysis.txt"), "w", encoding="utf-8") as predictions_file:
        # TODO: Predict the tags on the test set.
        predictions = whole_model.predict(test_dataset)
        labels = oh_to_labels(predictions)
        for label in labels:
            print(label, file=predictions_file)

        #label_strings = facebook.test.label_mapping.get_vocabulary()
        #for sentence in predictions:
        #    print(label_strings[np.argmax(sentence)], file=predictions_file)
    
    with open(os.path.join(args.logdir, "sentiment_analysis_dev.txt"), "w", encoding="utf-8") as predictions_file:
        # TODO: Predict the tags on the test set.
        predictions = whole_model.predict(val_dataset)
        labels = oh_to_labels(predictions)
        for label in labels:
            print(label, file=predictions_file)

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
