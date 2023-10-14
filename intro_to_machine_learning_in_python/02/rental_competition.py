#!/usr/bin/env python3
import argparse
import lzma
import os
import pickle
import urllib.request
import sklearn.compose
import sklearn.model_selection
import sklearn.pipeline
import sklearn.preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import ShuffleSplit
import sklearn.metrics
import time

import numpy as np

class Dataset:
    def __init__(self,
                 name="rental_competition.train.npz",
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
parser.add_argument("--predict", default="rental_competition.train_quadrat_feat.npz", type=str, help="Run prediction on given data")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="", type=str, help="Model path")

def create_classificator():
    l1_ratios = 1-np.geomspace(0.001,1,4)
    eps = 1e-3
    alphas = np.geomspace(0.01, 100, num=10)
    cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)

    clf = ElasticNetCV(
        l1_ratio=l1_ratios,
        eps = eps,
        alphas = alphas,
        normalize=False,
        max_iter=600,
        cv=cv, # default 5
        verbose=2,
        n_jobs=-1,
            )
    return clf

def main(args):
    if args.predict is None:
        t0 = time.time()
        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset()
        X = train.data
        y = train.target

        categorical_features = np.all(train.data.astype(int) == train.data, axis=0)
        float_features = ~categorical_features

        ohe = sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore') # sparse=False
        stdsc = sklearn.preprocessing.StandardScaler()
        preprocessor = sklearn.compose.ColumnTransformer(
        [('ohe', ohe, categorical_features), 
        ('std_scl', stdsc, float_features)], 
        remainder='passthrough')

        pol_feat = sklearn.preprocessing.PolynomialFeatures(4, include_bias=False)

        clf = create_classificator()

        model = sklearn.pipeline.Pipeline([('prep', preprocessor),
                                    ('features', pol_feat),
                                    ('classificator', clf)])

        #cv = ShuffleSplit(n_splits=4, test_size=0.25, random_state=0)
        print(model)
        print('training')
        
        # TODO: Train a model on the given dataset and store it in `model`.
        model.fit(X,y)

        t1 = time.time() - t0
        print("Time elapsed: " + str(t1)+ ' seconds') # CPU seconds elapsed (floating point)
        #print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

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
        #rmse = sklearn.metrics.mean_squared_error(test.target, predictions, squared=False)
        #print(rmse)

        return predictions


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
