#!/usr/bin/env python3
import argparse
import lzma
import pickle
import os
import urllib.request
import sys
import re

import sklearn.neural_network
import sklearn.feature_extraction.text
import sklearn.pipeline

import diacritization_dictionary as dictionary

import numpy as np

class Dataset:
    LETTERS_NODIA = "acdeeinorstuuyz"
    LETTERS_DIA = "áčďéěíňóřšťúůýž"

    # A translation table usable with `str.translate` to rewrite characters with dia to the ones without them.
    DIA_TO_NODIA = str.maketrans(LETTERS_DIA + LETTERS_DIA.upper(), LETTERS_NODIA + LETTERS_NODIA.upper())

    def __init__(self,
                 name="fiction-train.txt",
                 url="https://ufal.mff.cuni.cz/~straka/courses/npfl129/2021/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve(url + name, filename=name)
            urllib.request.urlretrieve(url + name.replace(".txt", ".LICENSE"), filename=name.replace(".txt", ".LICENSE"))

        # Load the dataset and split it into `data` and `target`.
        with open(name, "r", encoding="utf-8") as dataset_file:
            self.target = dataset_file.read()
        self.data = self.target.translate(self.DIA_TO_NODIA)

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default="fiction-train.txt", type=str, help="Run prediction on given data")
#parser.add_argument("--predict", default=None, type=str, help="Run prediction on given data")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="diacritization.model", type=str, help="Model path")

PADDING = 3
NGRAM = 3

CLASS0 = "acdeinorstuyz"
CLASS1 = "áéíóúý"
CLASS2 = "ěčďňřšťžů"
# TODO rozdelit na lepsi tridy

def get_class(ch):
    if ch in CLASS0 + CLASS0.upper():
        return 0
    elif ch in CLASS1 + CLASS1.upper():
        return 1
    elif ch in CLASS2 + CLASS2.upper():
        return 2
    else:
        print("wrong value in get_class function:", ch)

def char_options(ch):
    if ch in 'aá':
        return 'aáa'
    elif ch in 'cč':
        return 'ccč'
    elif ch in 'dď':
        return 'ddď'
    elif ch in 'eéě':
        return 'eéě'
    elif ch in 'ií':
        return 'iíi'
    elif ch in 'nň':
        return 'nnň'
    elif ch in 'oó':
        return 'oóo'
    elif ch in 'rř':
        return 'rrř'
    elif ch in 'sš':
        return 'ssš'
    elif ch in 'tť':
        return 'ttť'
    elif ch in 'uúů':
        return 'uúů'
    elif ch in 'yý':
        return 'yýy'
    elif ch in 'zž':
        return 'zzž'
    elif ch in 'AÁ':
        return 'AÁA'
    elif ch in 'CČ':
        return 'CCČ'
    elif ch in 'DĎ':
        return 'DDĎ'
    elif ch in 'EÉĚ':
        return 'EÉĚ'
    elif ch in 'IÍ':
        return 'IÍI'
    elif ch in 'NŇ':
        return 'NNŇ'
    elif ch in 'IÍ':
        return 'IÍI'
    elif ch in 'OÓ':
        return 'OÓO'
    elif ch in 'RŘ':
        return 'RRŘ'
    elif ch in 'SŠ':
        return 'SSŠ'
    elif ch in 'TŤ':
        return 'TTŤ'
    elif ch in 'UÚŮ':
        return 'UÚŮ'
    elif ch in 'YÝ':
        return 'YÝY'
    elif ch in 'ZŽ':
        return 'ZZŽ'
    else:
        print("wrong value in char_options function:", ch)

# max length of word = 20
# maximum znaku = 4 + " " (!!!" )
# max zaznam ve slovniku 9
# pocet slov ve slovniku 29015

classes = np.zeros(9)

def getTargetDictWordIdx(word, target_word, dict):
    dict_words = dict[word]
    for i, w in enumerate(dict_words):
        if w == target_word:
            classes[i] += 1
            return i
    return -1
    #print(word, " ", target_word, " ", dict[word])

def main(args):
    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset()
        dict = dictionary.Dictionary().variants
        #padding = " "*PADDING
        #padded_data = padding + train.data + padding
        #padded_target = padding + train.target + padding 
        word_data = train.data.split()
        word_target = train.target.split()

        # DICT length = 29015
        # DICT max values = 9
        # DICT max word length = 20

        is_in_dict = []
        whitespaces = 0   
        word_cum_begin = []
        alpha_words = []
        alpha_word = []
        total_alpha = 0
        cur_w_begin = 0  
        for i, w in enumerate(word_data):
            if w in dict:
                is_in_dict.append(True)
            else:
                is_in_dict.append(False)
            if w.isalpha():
                alpha_words.append(w)
                alpha_word.append(True)
                total_alpha += 1
            else:
                alpha_word.append(False)
            word_cum_begin.append(cur_w_begin)
            cur_w_begin += len(w) + 1

        train_data = []
        train_target = []
        prev_n_alphas = [""]*(NGRAM-1)

        cur_alpha = 0
        for i, w in enumerate(word_data):
            if alpha_word[i]: 
                cur_alpha += 1
                if w in dict:
                    if cur_alpha > NGRAM-1 and cur_alpha <= total_alpha-NGRAM+1:
                        tg = getTargetDictWordIdx(w, word_target[i], dict)
                        if tg != -1:
                            sample = []
                            for j in range(NGRAM):
                                ng = "".join([word for word in alpha_words[cur_alpha-1-j : cur_alpha-1+(NGRAM-1)-j]])
                                sample.append(ng)
                        
                            train_data.append(sample)
                            #print(tg)
                            train_target.append(tg)

        #print(classes)

        #for i in range(len(padded_data)):
        #    if(padded_data[i].lower() in train.LETTERS_NODIA):
        #        d = padded_data[i-PADDING:i+PADDING+1].lower()
        #        dwords = []
        #        for j in range(PADDING+1):
        #            dwords.append(d[j:j+PADDING+1])
        #        train_data.append(dwords)#.astype(np.int))
        #        train_target.append(get_class(padded_target[i]))

        #print(np.array(train_data).shape)
        #print(np.array(train_target).shape)
        #print(np.any(np.array(train_target)>0))
        
        #print(len(dict))
        #max_l = 0
        #max = []
        #k_max = []
        #max_word_l = 0
        #for k in dict:
        #    l = dict[k]
        #    for w in l:
        #        if len(w) > max_word_l:
        #            max_word_l = len(w)
        #    if len(dict[k]) == 9:
        #        print(k, " ", dict[k])
        #        k_max = k
        #        max_l = len(dict[k])
        #        max = dict[k]
        #    #print(k, " ", len(dict[k]))
        #print(max_l, " ", k_max, " ", max)
        #print(max_word_l)

        ohe = sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore')

        clf = sklearn.neural_network.MLPClassifier(solver='adam', warm_start=True, early_stopping=False, tol=1e-9, alpha=0.0001, epsilon=1e-08, 
                max_iter=1, n_iter_no_change=100, hidden_layer_sizes=(32,32,), random_state=1, verbose=True)
        model = sklearn.pipeline.Pipeline([('prepr', ohe),
                                       #('feat', pf),
                                        ('clf', clf)])

        with lzma.open("diacritization.model", "rb") as model_file:
            model = pickle.load(model_file)

        #model.named_steps.clf.warm_start = True
        #model.named_steps.clf.early_stopping = False
        #model.named_steps.clf.n_iter_no_change = 150

        print("starting")
        #for i in range(5):
        #    model.fit(train_data, train_target)
        # https://github.com/arahusky/diacritics_restoration not used

        # If you trained one or more MLPs, you can use the following code
        # to compress it significantly (approximately 12 times). The snippet
        # assumes the trained MLPClassifier is in `mlp` variable.
        mlp = model.named_steps.clf
        mlp._optimizer = None
        for i in range(len(mlp.coefs_)): mlp.coefs_[i] = mlp.coefs_[i].astype(np.float16)
        for i in range(len(mlp.intercepts_)): mlp.intercepts_[i] = mlp.intercepts_[i].astype(np.float16)

        # Serialize the model.
        #with lzma.open("compdiacritization.model", "wb") as model_file:
        #    pickle.dump(model, model_file)
        with lzma.open("diacritization.model", "wb") as model_file:
            pickle.dump(model, model_file)

    else:
        # Use the model and return test set predictions, as either a Python list or a NumPy array.
        test = Dataset(args.predict)
        dict = dictionary.Dictionary().variants

        with lzma.open("completterdiacritization.model", "rb") as model_file:
            letter_model = pickle.load(model_file)
        with lzma.open("compworddiacritization.model", "rb") as model_file:
            word_model = pickle.load(model_file)

        padding = " "*PADDING
        #padded_data = padding + test.data + padding
        padded_data = padding + test.data + padding 

        pred_data = []
        td_positions = []
        for i in range(len(padded_data)):
            if(padded_data[i].lower() in test.LETTERS_NODIA):
                td_positions.append(i)
                d = padded_data[i-PADDING:i+PADDING+1].lower()
                dwords = []
                for j in range(PADDING+1):
                    dwords.append(d[j:j+PADDING+1])
                pred_data.append(dwords)#.astype(np.int))
        #print(np.array(pred_data).shape)
        predicted_classes = letter_model.predict(np.array(pred_data))
        #print(np.any(predicted_classes>0))
        padded_data = list(padded_data)
        for i, pos in enumerate(td_positions):
            opts = list(char_options(padded_data[pos]))
            padded_data[pos] = opts[predicted_classes[i]]

        # TODO: Generate `predictions` with the test set predictions. Specifically,
        # produce a diacritized `str` with exactly the same number of words as `test.data`.
        #print(predictions)
        predictions = "".join(padded_data[PADDING:-PADDING])
        pred_split = predictions.split()
        
        word_data = test.data.split()
        
        is_in_dict = []
        whitespaces = 0   
        word_cum_begin = []
        alpha_word = []
        alpha_words = []
        total_alpha = 0
        cur_w_begin = 0  
        for i, w in enumerate(word_data):
            if w in dict:
                is_in_dict.append(True)
            else:
                is_in_dict.append(False)
            if w.isalpha():
                alpha_word.append(True)
                alpha_words.append(w)
                total_alpha += 1
            else:
                alpha_word.append(False)
            word_cum_begin.append(cur_w_begin)
            cur_w_begin += len(w) + 1

        pred_data = []
        pred_idx = []
        pred_word = []
        train_target = []
        
        cur_alpha = 0
        for i, w in enumerate(word_data):
            if alpha_word[i]: 
                cur_alpha += 1
                if w in dict:
                    if cur_alpha > NGRAM-1 and cur_alpha <= total_alpha-NGRAM+1:
                        sample = []
                        for j in range(NGRAM):
                            ng = "".join([word for word in alpha_words[cur_alpha-1-j : cur_alpha-1+(NGRAM-1)-j]])
                            #ng = "".join([word for word in word_data[i-j : i+(NGRAM-1)-j]])
                            sample.append(ng)
                        
                        pred_idx.append(i)
                        pred_data.append(sample)
                        pred_word.append(w)
        
        predicted_dict_classes = word_model.predict(np.array(pred_data))
        #diac_words = []
        #diac_words_idxs = []
        for i, w in enumerate(pred_word):                
            options = dict[w]
            if predicted_dict_classes[i] < len(options)-1:
                #pred_split[pred_idx[i]] = options[predicted_dict_classes[i]] 
                pred_split[pred_idx[i]] = options[predicted_dict_classes[i]] 

        predictions = " ".join(pred_split)
        
        with open("predictions.txt", "w") as text_file:
            text_file.write(predictions)
        with open("target.txt", "w") as text_file:
            text_file.write(test.target)
        #print(accuracy(test.target, predictions))
        return predictions


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
