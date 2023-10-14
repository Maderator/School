#!/usr/bin/env python3
import argparse
import lzma
import os
import pickle
import urllib.request

import numpy as np
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection
import sklearn.pipeline
import sklearn.preprocessing


class Dataset:
    """Thyroid Dataset.

    The dataset contains real medical data related to thyroid gland function,
    classified either as normal or irregular (i.e., some thyroid disease).
    The data consists of the following features in this order:
    - 15 binary features
    - 6 real-valued features

    The target variable is binary, with 1 denoting a thyroid disease and
    0 normal function.
    """
    def __init__(self,
                 name="thyroid_competition.train.npz",
                 url="https://ufal.mff.cuni.cz/~straka/courses/npfl129/2021/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name))
            urllib.request.urlretrieve(url + name, filename=name)

        # Load the dataset and return the data and targets.
        dataset = np.load(name)
        for key, value in dataset.items():
            setattr(self, key, value)


parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Run prediction on given data")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="thyroid_competition.model", type=str, help="Model path")

def main(args):
    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset()
        data = train.data
        target = train.target

        # TODO: Train a model on the given dataset and store it in `model`.
        minmax = sklearn.preprocessing.MinMaxScaler()
        pf = sklearn.preprocessing.PolynomialFeatures(include_bias=True) # pridat include_bias=False ?
        skf = sklearn.model_selection.StratifiedKFold(15)
        clf = sklearn.linear_model.LogisticRegression(random_state=args.seed)

        model = sklearn.pipeline.Pipeline([('prepr', minmax),
                                        ('feat', pf),
                                        ('clf', clf)])

        param_grid={'clf__C' : [10000000],
                'clf__solver' : ['newton-cg'],
                'feat__degree' : [3],
                'clf__max_iter' : [50]}

        gs = sklearn.model_selection.GridSearchCV(model, param_grid, n_jobs=-1, pre_dispatch=6, cv=skf)

        print(gs.get_params())

        #for train_inds, test_inds in skf.split(train_data, train_target):
        gs.fit(data, target)

        print(gs.best_estimator_)
        print(gs.best_index_)
        print(gs.best_score_)
        print(gs.best_params_)

        model = gs

        # Serialize the model.
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(model, model_file)

    else:
        # Use the model and return test set predictions, as either a Python list or a NumPy array.
        test = Dataset(args.predict)

        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)

        # TODO: Generate `predictions` with the test set predictions.
        predictions = model.predict(test.data)

        return predictions


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)