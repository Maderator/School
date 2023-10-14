#!/usr/bin/env python3
import argparse
import lzma
import pickle
import os
import urllib.request
import sys
import zipfile
from numpy.lib.function_base import vectorize
from sklearn.feature_extraction.text import TfidfVectorizer
import sklearn.model_selection
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
import sklearn.pipeline
from sklearn.svm import LinearSVC
import sklearn.model_selection

import numpy as np

class Dataset:
    def __init__(self,
                 name="isnt_it_ironic.train.zip",
                 url="https://ufal.mff.cuni.cz/~straka/courses/npfl129/2021/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve(url + name, filename=name)

        # Load the dataset and split it into `data` and `target`.
        self.data = []
        self.target = []

        with zipfile.ZipFile(name, "r") as dataset_file:
            with dataset_file.open(os.path.basename(name).replace(".zip", ".txt"), "r") as train_file:
                for line in train_file:
                    label, text = line.decode("utf-8").rstrip("\n").split("\t")
                    self.data.append(text)
                    self.target.append(int(label))
        self.target = np.array(self.target, np.int32)

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Run prediction on given data")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="isnt_it_ironic.model", type=str, help="Model path")

def main(args):
    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset()
    

        vectorizer = TfidfVectorizer(ngram_range=(1,2))
        #X = vectorizer.fit_transform(train.data)
        #X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X,train.target, test_size = 0.2, random_state = 0)

        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(train.data, train.target, test_size = 0.2, random_state = 0)

        #clf = MultinomialNB()
        clf = LinearSVC(max_iter=10000)
        #clf = MLPClassifier()

        model = sklearn.pipeline.Pipeline([('vec', vectorizer),
                                        ('clf', clf)])

        #param_grid={'clf__tol' : [1e-4],
        #            'clf__C': [0.01, 0.1, 1, 10, 100, 1000]}
        #skf = sklearn.model_selection.StratifiedKFold(5)

        #gs = sklearn.model_selection.GridSearchCV(model, param_grid, cv=skf)
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(classification_report(y_test, y_pred))
        
        #gs.fit(X_train, y_train)
        #y_pred = gs.predict(X_test)
        #print(classification_report(y_test, y_pred))

        #gs.fit(train.data, train.target)
        #print(sorted(gs.cv_results_.keys()))
        #model.fit(train.data, train.target)

        # TODO: Train a model on the given dataset and store it in `model`.
        #model = None

        # Serialize the model.
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(model, model_file)

    else:
        # Use the model and return test set predictions.
        test = Dataset(args.predict)

        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)

        # TODO: Generate `predictions` with the test set predictions, either
        # as a Python list of a NumPy array.
        predictions = model.predict(test.data)

        return predictions


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
