#!/usr/bin/env python3
import argparse
import lzma
import pickle
import os
import urllib.request
import sys

import sklearn.neural_network
import sklearn.feature_extraction.text
import sklearn.pipeline

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

PADDING = 5

CLASS0 = "acdeinorstuyz"
CLASS1 = "áéíóúý"
CLASS2 = "ěčďňřšťž"
CLASS3 = "ů"
# TODO rozdelit na lepsi tridy

def get_class(ch):
    if ch in CLASS0 + CLASS0.upper():
        return 0
    elif ch in CLASS1 + CLASS1.upper():
        return 1
    elif ch in CLASS2 + CLASS2.upper():
        return 2
    elif ch in CLASS3 + CLASS3.upper():
        return 3
    else:
        print("wrong value in get_class function:", ch)

def char_options(ch):
    if ch in 'aá':
        return 'aáaa'
    elif ch in 'cč':
        return 'ccčc'
    elif ch in 'dď':
        return 'ddďd'
    elif ch in 'eéě':
        return 'eéěe'
    elif ch in 'ií':
        return 'iíii'
    elif ch in 'nň':
        return 'nnňn'
    elif ch in 'oó':
        return 'oóoo'
    elif ch in 'rř':
        return 'rrřr'
    elif ch in 'sš':
        return 'ssšs'
    elif ch in 'tť':
        return 'ttťt'
    elif ch in 'uúů':
        return 'uúuů'
    elif ch in 'yý':
        return 'yýyy'
    elif ch in 'zž':
        return 'zzžz'
    elif ch in 'AÁ':
        return 'AÁAA'
    elif ch in 'CČ':
        return 'CCČC'
    elif ch in 'DĎ':
        return 'DDĎD'
    elif ch in 'EÉĚ':
        return 'EÉĚE'
    elif ch in 'IÍ':
        return 'IÍII'
    elif ch in 'NŇ':
        return 'NNŇN'
    elif ch in 'IÍ':
        return 'IÍII'
    elif ch in 'OÓ':
        return 'OÓOO'
    elif ch in 'RŘ':
        return 'RRŘR'
    elif ch in 'SŠ':
        return 'SSŠS'
    elif ch in 'TŤ':
        return 'TTŤT'
    elif ch in 'UÚŮ':
        return 'UÚUŮ'
    elif ch in 'YÝ':
        return 'YÝYY'
    elif ch in 'ZŽ':
        return 'ZZŽZ'
    else:
        print("wrong value in char_options function:", ch)

def main(args):
    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset()
        padding = " "*PADDING
        padded_data = padding + train.data + padding
        padded_target = padding + train.target + padding 

        train_data = []
        train_target = []
        for i in range(len(padded_data)):
            if(padded_data[i].lower() in train.LETTERS_NODIA):
                d = list(padded_data[i-PADDING:i+PADDING+1].lower())
                dint = []
                for ch in d:
                    dint.append(ord(ch))
                train_data.append(dint)#.astype(np.int))
                train_target.append(get_class(padded_target[i]))

        #print(np.array(train_data).shape)
        #print(np.array(train_target).shape)
        #print(np.any(np.array(train_target)>0))

        ohe = sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore')
        #pf = sklearn.preprocessing.PolynomialFeatures(2,include_bias=True) # pridat include_bias=False ?

        #clf = sklearn.neural_network.MLPClassifier(solver='adam', warm_start=True, early_stopping=False, tol=1e-9, alpha=0.0001, epsilon=1e-08, max_iter=5, n_iter_no_change=100, hidden_layer_sizes=(500,300,), random_state=1, verbose=True)
        #model = sklearn.pipeline.Pipeline([('prepr', ohe),
                                        #('feat', pf),
        #                                ('clf', clf)])

        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)

        #model.named_steps.clf.warm_start = True
        #model.named_steps.clf.early_stopping = False
        #model.named_steps.clf.n_iter_no_change = 150

        for i in range(5):
            model.fit(train_data, train_target)
        # https://github.com/arahusky/diacritics_restoration not used

        # Serialize the model.
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(model, model_file)

    else:
        # Use the model and return test set predictions, as either a Python list or a NumPy array.
        test = Dataset(args.predict)

        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)

        #print(model.named_steps)
        #model.named_steps.clf._optimizer = None
        #for i in range(len(model.named_steps.clf.coefs_)): model.named_steps.clf.coefs_[i] = model.named_steps.clf.coefs_[i].astype(np.float16)
        #for i in range(len(model.named_steps.clf.intercepts_)): model.named_steps.clf.intercepts_[i] = model.named_steps.clf.intercepts_[i].astype(np.float16)
        #with lzma.open(args.model_path, "wb") as model_file:
        #    pickle.dump(model, model_file)

        padding = " "*PADDING
        #padded_data = padding + test.data + padding
        test.data = padding + test.data + padding 

        pred_data = []
        td_positions = []
        for i in range(len(test.data)):
            if(test.data[i].lower() in test.LETTERS_NODIA):
                td_positions.append(i)
                d = list(test.data[i-PADDING:i+PADDING+1].lower())
                dint = []
                for ch in d:
                    dint.append(ord(ch))
                pred_data.append(dint)#.astype(np.int))

        #print(np.array(pred_data).shape)
        predicted_classes = model.predict(np.array(pred_data))
        #print(np.any(predicted_classes>0))
        test.data = list(test.data)
        for i, pos in enumerate(td_positions):
            opts = list(char_options(test.data[pos]))
            test.data[pos] = opts[predicted_classes[i]]

        # TODO: Generate `predictions` with the test set predictions. Specifically,
        # produce a diacritized `str` with exactly the same number of words as `test.data`.
        #print(predictions)
        predictions = "".join(test.data[PADDING:-PADDING])
        with open("predictions.txt", "w") as text_file:
            text_file.write(predictions)
        with open("target.txt", "w") as text_file:
            text_file.write(test.target)
        #print(accuracy(test.target, predictions))
        return predictions


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
