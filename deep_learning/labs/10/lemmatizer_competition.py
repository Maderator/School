#!/usr/bin/env python3
import argparse
import datetime
import os
import re
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2") # Report only TF errors by default

import numpy as np
import tensorflow as tf

from morpho_analyzer import MorphoAnalyzer
from morpho_dataset import MorphoDataset

from lemmatizer_attn import Network

# TODO: Define reasonable defaults and optionally more parameters
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
parser.add_argument("--epochs", default=40, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=0, type=int, help="Maximum number of threads to use.")

parser.add_argument("--cle_dim", default=256, type=int, help="CLE embedding dimension.")
parser.add_argument("--max_sentences", default=None, type=int, help="Maximum number of sentences to load.")
parser.add_argument("--rnn_dim", default=256, type=int, help="RNN cell dimension.")
parser.add_argument("--init_learning_rate", default=0.001, type=float, help="RNN cell dimension.")
parser.add_argument("--end_learning_alpha", default=0.001, type=float, help="RNN cell dimension.")


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

    # Load the data. Using analyses is only optional.
    morpho = MorphoDataset("czech_pdt", add_bow_eow=True)
    analyses = MorphoAnalyzer("czech_pdt_analyses")

    # TODO: Create the model and train it
    model = Network(args, morpho.train)
    
    # Construct dataset for lemmatizer training
    def create_dataset(name):
        dataset = getattr(morpho, name).dataset
        dataset = dataset.map(lambda forms, lemmas, tags: (forms, lemmas))
        dataset = dataset.shuffle(len(dataset), seed=args.seed) if name == "train" else dataset
        dataset = dataset.apply(tf.data.experimental.dense_to_ragged_batch(args.batch_size))
        return dataset
    train, dev, test = create_dataset("train"), create_dataset("dev"), create_dataset("test")
    
    training_batches = args.epochs * len(morpho.train.dataset) // args.batch_size
    decay_fn = lambda value: tf.keras.experimental.CosineDecay(value, training_batches, alpha = args.end_learning_alpha)
    learning_rate = decay_fn(args.init_learning_rate)
    optimizer = tf.optimizers.Adam(learning_rate=learning_rate, beta_2=0.9, epsilon=1e-3)
    
    model.compile(
        optimizer=optimizer,
        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.metrics.Accuracy(name="accuracy")],
    )

    model.fit(train, epochs=args.epochs, validation_data=dev, verbose=1,
                callbacks=[model.tb_callback])

    # Generate test set annotations, but in args.logdir to allow parallel execution.
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "lemmatizer_competition.txt"), "w", encoding="utf-8") as predictions_file:
        # Predict the tags on the test set; update the following prediction
        # command if you use other output structre than in lemmatizer_noattn.
        predictions = model.predict(test)
        for sentence in predictions:
            for word in sentence:
                print(word.numpy().decode("utf-8"), file=predictions_file)
            print(file=predictions_file)

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
