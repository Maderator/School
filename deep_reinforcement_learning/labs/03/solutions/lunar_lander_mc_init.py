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
parser.add_argument("--episodes", default=100000, type=int, help="Training episodes.")
parser.add_argument("--output", type=str, help="Output path.") # Private

def main(env: wrappers.EvaluationEnv, args: argparse.Namespace) -> None:
    # Fix random seed
    np.random.seed(args.seed)
    env.env._expert.seed(args.seed)

    Q = np.zeros([env.observation_space.n, env.action_space.n])
    C = np.zeros([env.observation_space.n, env.action_space.n], np.int32)

    for episode in range(args.episodes):
        if episode and episode % 100 == 0:
            print("Episode {}".format(episode), flush=True)

        state, expert = env.expert_trajectory()
        expert.insert(0, (None, 0, state))

        G, next_action = 0, None
        for action, reward, state in reversed(expert):
            if next_action is not None:
                C[state, next_action] += 1
                Q[state, next_action] += 1 / C[state, next_action] * (G - Q[state, next_action])
            G += reward
            next_action = action

    with open(args.output, "wb") as output_file:
        pickle.dump((Q, C), output_file)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    env = wrappers.EvaluationEnv(wrappers.DiscreteLunarLanderWrapper(gym.make("LunarLander-v2")), args.seed)

    main(env, args)
