#!/usr/bin/env python3
import argparse

import numpy as np

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
# If you add more arguments, ReCodEx will keep them with your default values.

def main(args):
    # TODO: Load data distribution, each line containing a datapoint -- a string.
    data_data = {}
    with open("numpy_entropy_data.txt", "r") as data:
        for line in data:
            line = line.rstrip("\n")
            # TODO: Process the line, aggregating data with built-in Python
            # data structures (not NumPy, which is not suitable for incremental
            # addition and string mapping).
            data_data[line] = data_data.get(line, 0) + 1

    # TODO: Create a NumPy array containing the data distribution. The
    # NumPy array should contain only data, not any mapping. Alternatively,
    # the NumPy array might be created after loading the model distribution.

    # TODO: Load model distribution, each line `string \t probability`.
    data_model = {}
    with open("numpy_entropy_model.txt", "r") as model:
        for line in model:
            line = line.rstrip("\n")
            # TODO: process the line, aggregating using Python data structures
            data_point, probability = line.split("\t")
            data_model[data_point] = float(probability)
    del data_point

    for k in data_model:
        if k not in data_data:
            data_data[k] = 0

    for k in data_data:
        if k not in data_model:
            data_model[k] = 0

    data_sum = sum(map(lambda k: data_data[k], data_data))
    data_np = np.array([data_data[k] / data_sum for k in sorted(data_data)])

    # TODO: Create a NumPy array containing the model distribution.
    model_np = np.array([data_model[k] for k in sorted(data_model)])

    # TODO: Compute the entropy H(data distribution). You should not use
    # manual for/while cycles, but instead use the fact that most NumPy methods
    # operate on all elements (for example `*` is vector element-wise multiplication).
    entropy = - np.sum(
        np.where(data_np > 0, data_np * np.log(data_np), 0)
        )

    # TODO: Compute cross-entropy H(data distribution, model distribution).
    # When some data distribution elements are missing in the model distribution,
    # return `np.inf`.
    crossentropy = - np.sum(
        np.where(model_np > 0, data_np * np.log(model_np), - np.inf)
        )

    # TODO: Compute KL-divergence D_KL(data distribution, model_distribution),
    # again using `np.inf` when needed.
    kl_divergence = - np.sum(
        np.where(data_np == 0, 0, data_np * np.log(np.divide(model_np, data_np)))
        )

    # Return the computed values for ReCodEx to validate
    return entropy, crossentropy, kl_divergence

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    entropy, crossentropy, kl_divergence = main(args)
    print("Entropy: {:.2f} nats".format(entropy))
    print("Crossentropy: {:.2f} nats".format(crossentropy))
    print("KL divergence: {:.2f} nats".format(kl_divergence))
