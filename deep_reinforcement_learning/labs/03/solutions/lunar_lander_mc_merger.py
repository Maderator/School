#!/usr/bin/env python3
import argparse
import pickle

import gym
import numpy as np

import wrappers

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--input", type=str, nargs="+", help="Input estimate paths.")
parser.add_argument("--output", type=str, help="Output path.")

def main(env, args):
    # Fix random seed
    np.random.seed(args.seed)

    Q = np.zeros([env.observation_space.n, env.action_space.n])
    C = np.zeros([env.observation_space.n, env.action_space.n], np.int32)

    for input_path in args.input:
        with open(input_path, "rb") as input_file:
            Q2, C2 = pickle.load(input_file)
        Q = (Q * C + Q2 * C2) / np.where(C + C2 != 0, C + C2, 1)
        C += C2

    print("Merged {} inputs".format(len(args.input)))

    with open(args.output, "wb") as output_file:
        pickle.dump((Q, C), output_file)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    env = wrappers.EvaluationEnv(wrappers.DiscreteLunarLanderWrapper(gym.make("LunarLander-v2")), args.seed, evaluate_for=1000)

    main(env, args)
