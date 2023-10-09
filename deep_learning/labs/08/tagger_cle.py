#!/usr/bin/env python3
import argparse
import datetime
import os
import re
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2") # Report only TF errors by default

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from morpho_dataset import MorphoDataset

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=10, type=int, help="Batch size.")
parser.add_argument("--cle_dim", default=32, type=int, help="CLE embedding dimension.")
parser.add_argument("--epochs", default=5, type=int, help="Number of epochs.")
parser.add_argument("--max_sentences", default=None, type=int, help="Maximum number of sentences to load.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--rnn_cell", default="LSTM", type=str, help="RNN cell type.")
parser.add_argument("--rnn_cell_dim", default=64, type=int, help="RNN cell dimension.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
parser.add_argument("--we_dim", default=64, type=int, help="Word embedding dimension.")
parser.add_argument("--word_masking", default=0.0, type=float, help="Mask words with the given probability.")
# If you add more arguments, ReCodEx will keep them with your default values.

class Network(tf.keras.Model):
    # A layer setting given rate of elements to zero.
    class MaskElements(tf.keras.layers.Layer):
        def __init__(self, rate):
            super().__init__()
            self._rate = rate
        def get_config(self):
            return {"rate": self._rate}
        def call(self, inputs, training):
            if training:
                # TODO: Generate as many random uniform numbers in range [0, 1) as there are
                # values in `tf.RaggedTensor` `inputs` using a single `tf.random.uniform` call.
                # Then, set the values in `inputs` to zero if the corresponding generated
                # random number is less than `self._rate`.
                probs = tf.random.uniform(shape=[tf.size(inputs.flat_values)])
                flatten_in = inputs.flat_values
                flatten_in = flatten_in * tf.cast(tf.greater_equal(probs, self._rate), flatten_in.dtype)
                inputs = inputs.with_values(flatten_in)
                return inputs

            else:
                return inputs

    def __init__(self, args, train):
        # Implement a one-layer RNN network. The input `words` is
        # a RaggedTensor of strings, each batch example being a list of words.
        words = tf.keras.layers.Input(shape=[None], dtype=tf.string, ragged=True)

        # TODO(tagger_we): Map strings in `words` to indices by using the `word_mapping` of `train.forms`.
        words_indices = train.forms.word_mapping(words)

        # TODO: With a probability of `args.word_masking`, replace the input word by an
        # unknown word (which has index 0).
        #
        # There are two approaches you can use:
        # 1) use the above defined `MaskElements` layer, in which you need to implement
        #    one TODO note. If you do not want to implement it, you can instead
        # 2) use a `tf.keras.layers.Dropout` to achieve this, even if it is a bit
        #    hacky, because Dropout cannot proacess integral inputs. Start by using
        #    `tf.ones_like` to create a ragged tensor of float32 ones with the same
        #    structure as the indices of the input words, pass them through a dropout layer
        #    with `args.word_masking` rate, and finally set the input word ids to 0 where
        #    the result of dropout is zero.
        mask_layer = self.MaskElements(args.word_masking)
        words_indices = mask_layer(words_indices)

        # TODO(tagger_we): Embed input words with dimensionality `args.we_dim`. Note that the `word_mapping`
        # provides a `vocab_size()` call returning the number of unique words in the mapping.
        vocab_size = train.forms.word_mapping.vocab_size()
        embedding_layer = tf.keras.layers.Embedding(vocab_size, args.we_dim)
        embedded_words = embedding_layer(words_indices)

        # TODO: Create a vector of input words from all batches using `words.values`
        # and pass it through `tf.unique`, obtaining a list of unique words and
        # indices of the original flattened words in the unique word list.
        uni, idx = tf.unique(words.values)

        # TODO: Create sequences of letters by passing the unique words through
        # `tf.strings.unicode_split` call; use "UTF-8" as `input_encoding`.
        letters_sequence = tf.strings.unicode_split(uni,input_encoding="UTF-8")

        # TODO: Map the letters into ids by using `char_mapping` of `train.forms`.
        letters_ids = train.forms.char_mapping(letters_sequence)


        # TODO: Embed the input characters with dimensionality `args.cle_dim`.
        alphabet_size = train.forms.char_mapping.vocab_size()
        embedding_cle_layer = tf.keras.layers.Embedding(alphabet_size, args.cle_dim)
        embedded_characters = embedding_cle_layer(letters_ids)

        # TODO: Pass the embedded letters through a bidirectional GRU layer
        # with dimensionality `args.cle_dim`, obtaining character-level representations
        # of the whole words, **concatenating** the outputs of the forward and backward RNNs.
        char_rnn_cell = tf.keras.layers.GRU(units=args.cle_dim, return_sequences=False)
        char_rnn_bi_layer = tf.keras.layers.Bidirectional(char_rnn_cell, merge_mode='concat')
        char_rnn_output = char_rnn_bi_layer(embedded_characters.to_tensor(), mask=tf.sequence_mask(embedded_characters.row_lengths()))

        # TODO: Use `tf.gather` with the indices generated by `tf.unique` to transform
        # the computed character-level representations of the unique words to representations
        # of the flattened (non-unique) words.
        flat_cl_representation = tf.gather(char_rnn_output, idx)

        # TODO: Then, convert these character-level word representations into
        # a RaggedTensor of the same shape as `words` using `words.with_values` call.
        ragged_t_cl = words.with_values(flat_cl_representation)

        # TODO: Concatenate the word-level embeddings and the computed character-level WEs
        # (in this order).
        concat_we_cle = tf.keras.layers.concatenate([embedded_words, ragged_t_cl])

        # TODO(tagger_we): Create the specified `args.rnn_cell` RNN cell (LSTM, GRU) with
        # dimension `args.rnn_cell_dim`. The cell should produce an output for every
        # sequence element (so a 3D output). Then apply it in a bidirectional way on
        # the word representations, **summing** the outputs of forward and backward RNNs.
        if args.rnn_cell == "GRU":
            rnn_cell = tf.keras.layers.GRU(units=args.rnn_cell_dim, return_sequences=True)
        elif args.rnn_cell == "LSTM":
            rnn_cell = tf.keras.layers.LSTM(units=args.rnn_cell_dim, return_sequences=True)
        
        rnn_layer = tf.keras.layers.Bidirectional(rnn_cell, merge_mode='sum')

        rnn_output = rnn_layer(concat_we_cle.to_tensor(), mask=tf.sequence_mask(concat_we_cle.row_lengths()))

        rnn_output = tf.RaggedTensor.from_tensor(rnn_output, concat_we_cle.row_lengths())

        # TODO(tagge_we): Add a softmax classification layer into as many classes as there are unique
        # tags in the `word_mapping` of `train.tags`. However, because we are applying the
        # the Dense layer to a ragged tensor, we need to wrap the Dense layer in
        # a tf.keras.layers.TimeDistributed.
        tags_size = train.tags.word_mapping.vocab_size()
        classification_layer = tf.keras.layers.Dense(tags_size,activation=tf.nn.softmax)
        predictions = tf.keras.layers.TimeDistributed(classification_layer)(rnn_output)

        # Check that the created predictions are a 3D tensor.
        assert predictions.shape.rank == 3
        super().__init__(inputs=words, outputs=predictions)
        #self.compile(optimizer=tf.optimizers.Adam(),
        self.compile(optimizer=tfa.optimizers.LazyAdam(beta_2=0.99),
                     loss=tf.losses.SparseCategoricalCrossentropy(),
                     metrics=[tf.metrics.SparseCategoricalAccuracy(name="accuracy")])

        self.tb_callback = tf.keras.callbacks.TensorBoard(args.logdir, update_freq=100, profile_batch=0)
        self.tb_callback._close_writers = lambda: None # A hack allowing to keep the writers open.

    # Note that in TF 2.4, computing losses and metrics on RaggedTensors is not yet
    # supported (it will be in TF 2.5). Therefore, we override the `train_step` method
    # to support it, passing the "flattened" predictions and gold data to the loss
    # and metrics.
    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            # Check that both the gold data and predictions are RaggedTensors.
            assert isinstance(y_pred, tf.RaggedTensor) and isinstance(y, tf.RaggedTensor)
            loss = self.compiled_loss(y.values, y_pred.values, regularization_losses=self.losses)
        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        self.compiled_metrics.update_state(y.values, y_pred.values)
        return {m.name: m.result() for m in self.metrics}

    # Analogously to `train_step`, we also need to override `test_step`.
    def test_step(self, data):
        x, y = data
        y_pred = self(x, training=False)
        loss = self.compiled_loss(y.values, y_pred.values, regularization_losses=self.losses)
        self.compiled_metrics.update_state(y.values, y_pred.values)
        return {m.name: m.result() for m in self.metrics}

def main(args):
    # Fix random seeds and threads
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    if args.recodex:
        tf.keras.utils.get_custom_objects()["glorot_uniform"] = tf.initializers.GlorotUniform(seed=args.seed)
        tf.keras.utils.get_custom_objects()["orthogonal"] = tf.initializers.Orthogonal(seed=args.seed)
        tf.keras.utils.get_custom_objects()["uniform"] = tf.initializers.RandomUniform(seed=args.seed)

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    ))

    # Load the data
    morpho = MorphoDataset("czech_cac", max_sentences=args.max_sentences)

    # Create the network and train
    network = Network(args, morpho.train)

    # TODO(tagger_we): Construct dataset for training, which should contain pairs of
    # - tensor of string words (forms) as input
    # - tensor of integral tag ids as targets.
    # To create the identifiers, use the `word_mapping` of `morpho.train.tags`.
    def tagging_dataset(forms, lemmas, tags):
        tags_ids = morpho.train.tags.word_mapping(tags)
        return (forms, tags_ids)

    def create_dataset(name):
        dataset = getattr(morpho, name).dataset
        dataset = dataset.map(tagging_dataset)
        dataset = dataset.shuffle(len(dataset), seed=args.seed) if name == "train" else dataset
        dataset = dataset.apply(tf.data.experimental.dense_to_ragged_batch(args.batch_size))
        return dataset
    train, dev, test = create_dataset("train"), create_dataset("dev"), create_dataset("test")

    network.fit(train, epochs=args.epochs, validation_data=dev, callbacks=[network.tb_callback])

    test_logs = network.evaluate(test, return_dict=True)
    network.tb_callback.on_epoch_end(args.epochs, {"val_test_" + metric: value for metric, value in test_logs.items()})

    # Return test set accuracy for ReCodEx to validate
    return test_logs["accuracy"]

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
