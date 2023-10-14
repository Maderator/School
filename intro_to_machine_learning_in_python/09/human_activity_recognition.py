#!/usr/bin/env python3
import argparse
import lzma
import pickle
import os
import urllib.request
import sys
import zipfile
import sklearn.model_selection
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import sklearn.pipeline

import numpy as np
import pandas as pd


class Dataset:
    CLASSES = ["sitting", "sittingdown", "standing", "standingup", "walking"]

    def __init__(self,
                 name="human_activity_recognition.train.csv.xz",
                 url="https://ufal.mff.cuni.cz/~straka/courses/npfl129/2021/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve(url + name, filename=name)

        # Load the dataset and if it contains column "class", split it to `targets`.
        self.data = pd.read_csv(name)
        if "class" in self.data:
            self.target = np.array([Dataset.CLASSES.index(target) for target in self.data["class"]], np.int32)
            self.data = self.data.drop("class", axis=1)

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Run prediction on given data")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="human_activity_recognition.model", type=str, help="Model path")

def main(args):
    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset()

        # TODO: Train a model on the given dataset and store it in `model`.
        npdata = train.data.to_numpy()
        #print(npdata[0])
        #print(train.target[0])


        #X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(train.data, train.target, test_size = 0.2, random_state = 0)

        quant = QuantileTransformer()
        minmax = MinMaxScaler()


        #clf = MultinomialNB()
        #clf = LinearSVC(max_iter=10000)
        #clf = MLPClassifier()
        #clf = DecisionTreeClassifier()
        clf = RandomForestClassifier()

        model = sklearn.pipeline.Pipeline([('quant', quant),
                                        ('minmax', minmax),
                                        ('clf', clf)])

        #param_grid={'clf__tol' : [1e-4],
        #            'clf__C': [0.01, 0.1, 1, 10, 100, 1000]}
        #skf = sklearn.model_selection.StratifiedKFold(5)

        #gs = sklearn.model_selection.GridSearchCV(model, param_grid, cv=skf)
        
        model.fit(train.data, train.target)
        #y_pred = model.predict(X_test)
        #print(classification_report(y_test, y_pred))

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
