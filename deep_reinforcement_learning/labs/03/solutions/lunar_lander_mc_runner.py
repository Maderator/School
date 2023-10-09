#!/usr/bin/env python3
import argparse

import gym
import numpy as np

import wrappers

import lunar_lander_policy

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
# For these and any other arguments you add, ReCodEx will keep your default value.

def main(env, args):
    # Fix random seed
    np.random.seed(args.seed)

    policy = lunar_lander_policy.policy

    while True:
        state, done = env.reset(True), False
        while not done:
            if args.render_each and env.episode > 0 and env.episode % args.render_each == 0:
                env.render()

            action = policy[state]
            state, reward, done, _ = env.step(action)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    env = wrappers.EvaluationEnv(wrappers.DiscreteLunarLanderWrapper(gym.make("LunarLander-v2")), args.seed, evaluate_for=1000)

    main(env, args)
