#!/usr/bin/env python3

# IDs:
#edbe2dad-018e-11eb-9574-ea7484399335
#aa20f311-c7a2-11e8-a4be-00505601122b
#c716f6b0-25ab-11ec-986f-f39926f24a9c

import argparse
import sys

import gym
import numpy as np

import wrappers

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--alpha", default=None, type=float, help="Learning rate.")
parser.add_argument("--epsilon", default=None, type=float, help="Exploration factor.")
parser.add_argument("--gamma", default=None, type=float, help="Discounting factor.")

def mean_episode_return(env: wrappers.EvaluationEnv):
    return np.mean(env._episode_returns[-500:])

def main(env: wrappers.EvaluationEnv, args: argparse.Namespace) -> None:
    # Fix random seed
    np.random.seed(args.seed)

    n_states = env.observation_space.n
    n_actions = env.action_space.n

    Q = np.zeros((n_states, n_actions))
    C = np.zeros((n_states, n_actions))

    np.set_printoptions(threshold=np.inf)

    epsilon = args.epsilon or 0.2

    RETURN_CRITERION = 20
    Q_MATRIX_FILENAME = 'Q.bin'

    training = not args.recodex
    while training:
        if env.episode % 1000 == 0 and env.episode > 0:
            epsilon /= 2
            print(f"Epsilon: {epsilon}")

        # To generate expert trajectory, you can use
        state, trajectory = env.expert_trajectory()

        last_reward = None
        history = list()

        # Perform a training episode
        state, done = env.reset(), False
        while not done:
            if args.render_each and env.episode and env.episode % args.render_each == 0:
                env.render()

            if np.random.uniform() >= epsilon:
                # Greedy
                action = np.argmax(Q[state, :])
            else:
                # Uniform
                action = np.random.randint(0, n_actions)

            next_state, reward, done, _ = env.step(action)

            history.append((state, action, last_reward))

            state = next_state
            last_reward = reward

        history.append((None, None, last_reward))

        STATE = 0
        ACTION = 1
        REWARD = 2

        gamma = 0.99

        T = len(history) - 1
        G = 0
        for t in range(T - 1, -1, -1):
            G = gamma * G + history[t + 1][REWARD]
            C[history[t][STATE], history[t][ACTION]] += 1
            Q[history[t][STATE], history[t][ACTION]] += 1 / C[history[t][STATE], history[t][ACTION]] * \
                                                        (G - Q[history[t][STATE], history[t][ACTION]])

        if mean_episode_return(env) > RETURN_CRITERION:
            Q.tofile(Q_MATRIX_FILENAME)
            sys.exit(0)

    # Final evaluation
    Q = np.fromfile(Q_MATRIX_FILENAME).reshape((n_states, n_actions))
    while True:
        state, done = env.reset(start_evaluation=True), False
        while not done:
            # Choose (greedy) action
            action = np.argmax(Q[state, :])
            state, reward, done, _ = env.step(action)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    env = wrappers.EvaluationEnv(wrappers.DiscreteLunarLanderWrapper(gym.make("LunarLander-v2")), args.seed)

    main(env, args)
