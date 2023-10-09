#!/usr/bin/env python3
import argparse
import datetime
import os
import re
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2") # Report only TF errors by default

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from electra_czech_small_lc import ElectraCzechSmallLc
from reading_comprehension_dataset import ReadingComprehensionDataset

ELECTRA_LEN = 512

# TODO: Define reasonable defaults and optionally more parameters
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=6, type=int, help="Batch size.")
parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=41, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
parser.add_argument("--learning_rate", default=1e-4, type=float, help="Learning rate.")

class ComprehensionModel(tf.keras.Model):
    def __init__(self, args, electra, train_len, max_len=ELECTRA_LEN):
        
        el_model = electra.create_model()
        #el_model.trainable = False # We train everything if this is not used

        input_ids = tf.keras.layers.Input(shape=[max_len], dtype=tf.int32)
        attention_mask = tf.keras.layers.Input(shape=[max_len], dtype=tf.int32)
        softmax_mask = tf.keras.layers.Input(shape=[max_len], dtype=tf.int32)
        inputs = {"input_ids":input_ids, "attention_mask":attention_mask, "softmax_mask":softmax_mask}

        electra_out = el_model(inputs)
        #subwords_out_start = []
        #subwords_out_end = []
        #for i in range(ELECTRA_LEN):
            #subwords_out_start.append(tf.keras.layers.Dense(1)(electra_out.last_hidden_state[:,i,:]))
            #subwords_out_end.append(tf.keras.layers.Dense(1)(electra_out.last_hidden_state[:,i,:]))
        #subwords_out_start = tf.concat(subwords_out_start, axis=1)
        #subwords_out_end = tf.concat(subwords_out_end, axis=1)
        subwords_out_start = tf.keras.layers.Dense(1)(electra_out.last_hidden_state)
        subwords_out_end = tf.keras.layers.Dense(1)(electra_out.last_hidden_state)
        subwords_out_start = tf.reduce_sum(subwords_out_start, axis=2)
        subwords_out_end = tf.reduce_sum(subwords_out_end, axis=2)
        #x = tf.keras.layers.Flatten()(x)
        casted_mask = tf.cast(inputs["softmax_mask"], dtype=tf.float32)
        mult_mask = casted_mask
        add_mask = (1 - casted_mask) * -1e9
        subwords_out_start_masked = subwords_out_start * mult_mask + add_mask
        subwords_out_end_masked = subwords_out_end * mult_mask + add_mask
        ans_start = tf.keras.layers.Softmax(axis=1)(subwords_out_start_masked)
        ans_end = tf.keras.layers.Softmax(axis=1)(subwords_out_end_masked)
        outputs = {"answer_start":ans_start, "answer_end":ans_end}
        super().__init__(inputs=inputs, outputs=outputs)
        training_batches = args.epochs * train_len // args.batch_size
        decay_fn = lambda value: tf.keras.experimental.CosineDecay(value, decay_steps=training_batches)
        learning_rate = decay_fn(args.learning_rate)
        #optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
        optimizer = tfa.optimizers.LazyAdam(learning_rate=learning_rate, beta_2=0.98)
        self.compile(optimizer=optimizer,
                     loss=tf.keras.losses.CategoricalCrossentropy(),
                     metrics=[tf.keras.metrics.CategoricalCrossentropy(name="accuracy")])

        self.tb_callback = tf.keras.callbacks.TensorBoard(args.logdir, update_freq=100, profile_batch=0)
        self.tb_callback._close_writers = lambda: None # A hack allowing to keep the writers open.
        self.earlystopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', restore_best_weights=True, patience=2)

def create_example(qas, context, tokenizer, test=False, max_len=ELECTRA_LEN):
    question = qas["question"]
    encoded_context = tokenizer(context)
    tokenized_context = encoded_context["input_ids"]
    if test == False:
        answers = qas["answers"]
        answer_text = answers[0]["text"]
        answer_start = answers[0]["start"]
        answer_end = answer_start + len(answer_text)-1 # Last symbol of answer
        answer_start = encoded_context.char_to_token(answer_start)
        answer_end = encoded_context.char_to_token(answer_end)
        
        #answer_len = answer_end-answer_start
        
        #start_char = encoded_context.token_to_chars(answer_start).start
        #end_char = encoded_context.token_to_chars(answer_end).end
        #answer = context[start_char:end_char]
        #print(answer)
        
        question_tokenized = tokenizer.encode(question)
        question_tokenized[0] = tokenizer.sep_token_id
    else:
        answer_text = None
        answer_start = None
        answer_end = None
        question_tokenized = tokenizer.encode(question)
        question_tokenized[0] = tokenizer.sep_token_id
    if len(tokenized_context)-1 + len(question_tokenized) > max_len:
        context_max_len = max_len - len(question_tokenized)
        if not test and context_max_len <= answer_end:
            return None
        cut_context = tokenized_context[:context_max_len]
    else:
        cut_context = tokenized_context[:-1]
    inputs = cut_context + question_tokenized
    context_len = len(cut_context)
    return (inputs, answer_start, answer_end, context_len)

def prepare_data(dataset, tokenizer, test=False, max_len=ELECTRA_LEN):
    paragraphs = dataset.paragraphs
    data = []
    answer_len_max = 0
    for paragraph in paragraphs:
        context = paragraph["context"]
        qass = paragraph["qas"]
        for qas in qass:
            if len(qas["answers"]) == 0 and test == False:
                continue
            example = create_example(qas, context, tokenizer, test=test)
            #if example is not None:
            #    answer_len = example[4]
            #    if answer_len > answer_len_max:
            #        answer_len_max = answer_len
            if example is not None:
                data.append(example)
    #print("max_answer len: ", answer_len_max)
    
    inputs = np.zeros([len(data), max_len])
    masks = np.zeros([len(data), max_len])
    softmax_masks = np.zeros([len(data), max_len])
    answer_start = np.zeros([len(data)])
    answer_end = np.zeros([len(data)])
    for i in range(len(data)):
        inputs[i, :len(data[i][0])] = data[i][0]
        masks[i, :len(data[i][0])] = 1
        context_len = data[i][3]
        softmax_masks[i, :context_len] = 1
        answer_start[i] = data[i][1]
        answer_end[i] = data[i][2]
    if test:
        return ({"input_ids":inputs, "attention_mask":masks, "softmax_mask":softmax_masks},)
    else:
        answer_start_oh = tf.one_hot(answer_start, depth=max_len)
        answer_end_oh = tf.one_hot(answer_end, depth=max_len)
        #st = tf.argmax(answer_start_oh, axis=1)
        #end = tf.argmax(answer_end_oh, axis=1)
        #for i in range(len(st)):
        #    print(st[i], end[i])

        return ({"input_ids":inputs, "attention_mask":masks, "softmax_mask":softmax_masks}, {"answer_start":answer_start_oh, "answer_end":answer_end_oh})

def prepare_dataset(args, data, tokenizer, train=False, test=False, max_len=ELECTRA_LEN):
    prepared_data = prepare_data(data, tokenizer, test, max_len)
    data_len = len(prepared_data[0]["input_ids"])
    if train == True:
        dataset = tf.data.Dataset.from_tensor_slices(prepared_data).shuffle(data_len, reshuffle_each_iteration=True).batch(args.batch_size)
    else:
        dataset = tf.data.Dataset.from_tensor_slices(prepared_data).batch(args.batch_size)
    return dataset, data_len

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

    # Load the data.
    dataset = ReadingComprehensionDataset()

    # TODO: prepare data
    tokenizer = electra.create_tokenizer()

    train, train_len = prepare_dataset(args, dataset.train, tokenizer, train=True)
    print("train data prepared (len {})".format(train_len))
    dev, dev_len = prepare_dataset(args, dataset.dev, tokenizer)
    print("dev data prepared (len {})".format(dev_len))
    test, test_len = prepare_dataset(args, dataset.test, tokenizer, test=True)
    print("test data prepared (len {})".format(test_len))

    # TODO: create model
    model = ComprehensionModel(args, electra, train_len)
    model.trainable = True

    # TODO: train model
    model.fit(train, batch_size=args.batch_size, epochs=args.epochs, verbose=1,
                validation_data=dev, callbacks=[model.tb_callback, model.earlystopping_callback])

    # TODO: finetune electra
    #training_batches = args.epochs * train_len
    #decay_fn = lambda value: tf.keras.experimental.CosineDecay(value, decay_steps=training_batches, alpha = 0.01)
    #learning_rate = decay_fn(1e-4)
    #optimizer = tf.optimizers.Adam(learning_rate=learning_rate, beta_2=0.9, epsilon=1e-3)
    #model.trainable = True
    #model.compile(optimizer=optimizer,
    #                loss=tf.losses.SparseCategoricalCrossentropy(),
    #                metrics=[tf.metrics.SparseCategoricalAccuracy(name="accuracy")])

    # TODO: choose best answer
    print("learning ended")
    context_len = []
    context_texts = []
    encoded_contexts = []
    pars = dataset.test.paragraphs
    for par in pars:
        context = par["context"]
        encoded_context = tokenizer(context)
        tokenized_context = encoded_context["input_ids"]
        for qas in par["qas"]:
            context_len.append(len(tokenized_context) - 1)
            encoded_contexts.append(encoded_context)
            context_texts.append(context)

    print("predict started")
    pred = model.predict(test)
    pred_start = pred["answer_start"]
    pred_end = pred["answer_end"]
    print("pred ended")

    def get_most_probable_answers(pred_start, pred_end, context_text, context_encoded, context_len):
        answers = []
        for i in range(len(pred_start)):
            best_start1 = tf.argmax(pred_start[i, :tf.math.minimum(context_len[i], ELECTRA_LEN)])
            best_end1 = tf.argmax(pred_end[i, best_start1:tf.math.minimum(context_len[i], ELECTRA_LEN)])
            best_end1 = best_end1 + best_start1
            start1_prob = pred_start[i, best_start1]
            end1_prob = pred_end[i, best_end1]
            
            best_end2 = tf.argmax(pred_end[i, :tf.math.minimum(context_len[i], ELECTRA_LEN)])
            best_start2 = tf.argmax(pred_start[i, :best_end2+1])
            start2_prob = pred_start[i, best_start2]
            end2_prob = pred_end[i, best_end2]

            if start1_prob * end1_prob > start2_prob*end2_prob:
                best_start = best_start1
                best_end = best_end1
            else:
                best_start = best_start2
                best_end = best_end2
            
            con_en = context_encoded[i]
            if best_start == None:
                best_start = 0
                best_end = 0
            start_char = con_en.token_to_chars(best_start).start
            end_char = con_en.token_to_chars(best_end).end
            con_txt = context_text[i]
            answer = con_txt[start_char:end_char]
            answers.append(answer)
        return answers

    # Generate test set annotations, but in args.logdir to allow parallel execution.
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "reading_comprehension.txt"), "w", encoding="utf-8") as predictions_file:
        # TODO: Predict the answers as strings (if the answer is not
        # in the context, use an empty string).
        print("prob computation started")
        predictions = get_most_probable_answers(pred_start, pred_end, context_texts, encoded_contexts, context_len)
        print("prob ended")

        for answer in predictions:
            print(answer, file=predictions_file)

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
