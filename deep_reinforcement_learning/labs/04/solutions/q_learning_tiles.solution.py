#!/usr/bin/env python3
import argparse

import gym
import numpy as np

import wrappers

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--alpha", default=0.1, type=float, help="Learning rate.")
parser.add_argument("--epsilon", default=0.3, type=float, help="Exploration factor.")
parser.add_argument("--epsilon_final", default=0.1, type=float, help="Final exploration factor.")
parser.add_argument("--epsilon_final_at", default=1000, type=int, help="Training episodes.")
parser.add_argument("--gamma", default=1.0, type=float, help="Discounting factor.")
parser.add_argument("--tiles", default=8, type=int, help="Number of tiles.")

def main(env: wrappers.EvaluationEnv, args: argparse.Namespace) -> None:
    # Fix random seed
    np.random.seed(args.seed)

    # Implement Q-learning RL algorithm, using linear approximation.
    W = np.zeros([env.observation_space.nvec[-1], env.action_space.n])
    epsilon = args.epsilon

    training = True
    while training:
        # Perform episode
        state, done = env.reset(), False
        while not done:
            if args.render_each and env.episode and env.episode % args.render_each == 0:
                env.render()

            # TODO: Choose an action.
            action = np.argmax(W[state].sum(axis=0)) if np.random.uniform() >= epsilon else env.action_space.sample()

            next_state, reward, done, _ = env.step(action)

            # TODO: Update the action-value estimates
            W[state, action] = W[state, action] + args.alpha / args.tiles * (reward + (not done) * args.gamma * np.max(W[next_state].sum(axis=0)) - W[state, action].sum())

            state = next_state

        # End when reaching an average reward of -105
        if env.episode % 100 == 0:
            returns = 0
            for _ in range(100):
                state, done = env.reset(logging=False), False
                while not done:
                    action = np.argmax(W[state].sum(axis=0))
                    state, reward, done, _ = env.step(action)
                    returns += reward / 100
            print("Evaluation after episode {} returned {:.2f}".format(env.episode, returns))
            if returns >= -105:
                training = False

        if args.epsilon_final_at:
            epsilon = np.interp(env.episode + 1, [0, args.epsilon_final_at], [args.epsilon, args.epsilon_final])

    # Final evaluation
    while True:
        state, done = env.reset(start_evaluation=True), False
        while not done:
            # TODO: Choose (greedy) action
            action = np.argmax(W[state].sum(axis=0))
            state, reward, done, _ = env.step(action)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    env = wrappers.EvaluationEnv(wrappers.DiscreteMountainCarWrapper(gym.make("MountainCar1000-v0"), tiles=args.tiles), args.seed)

    main(env, args)
