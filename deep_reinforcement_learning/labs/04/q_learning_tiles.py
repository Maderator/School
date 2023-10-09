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
parser.add_argument("--alpha", default=0.01, type=float, help="Learning rate.")
parser.add_argument("--epsilon", default=0.005, type=float, help="Exploration factor.")
parser.add_argument("--epsilon_final", default=0.001, type=float, help="Final exploration factor.")
parser.add_argument("--epsilon_final_at", default=5000, type=int, help="Training episodes.")
parser.add_argument("--gamma", default=0.999, type=float, help="Discounting factor.")
parser.add_argument("--tiles", default=8, type=int, help="Number of tiles.")

def epsilon_greedy(env: wrappers.EvaluationEnv, epsilon: float, greedy_action: int) -> int:
    if np.random.uniform() < epsilon:
        return np.random.randint(env.action_space.n)
    return greedy_action

def choose_action(env: wrappers.EvaluationEnv, epsilon: float, state: list[int],
 weight_matrix: np.ndarray) -> int:
    state_action_prob = np.sum(weight_matrix[state], axis=0)
    greedy_action = np.argmax(state_action_prob)
    return epsilon_greedy(env, epsilon, greedy_action)

def main(env: wrappers.EvaluationEnv, args: argparse.Namespace) -> None:
    # Fix random seed
    np.random.seed(args.seed)

    # Implement Q-learning RL algorithm, using linear approximation.
    W = np.load("w.npy")
    #W = np.zeros([env.observation_space.nvec[-1], env.action_space.n])
    epsilon = args.epsilon

    training = not args.recodex
    while training:
        # Perform episode
        state, done = env.reset(), False
        while not done:
            if args.render_each and env.episode and env.episode % args.render_each == 0:
                env.render()

            # TODO: Choose an action.
            action = choose_action(env, epsilon, state, W)

            next_state, reward, done, _ = env.step(action)

            # TODO: Update the action-value estimates
            gradient = np.zeros(W.shape)
            gradient[state, action] = 1
            next_state_action_prob = np.sum(W[next_state], axis=0)
            state_action_prob = np.sum(W[state], axis=0)
            W +=    args.alpha/args.tiles * (reward + args.gamma * 
                    np.max(next_state_action_prob)-
                    state_action_prob[action]) * gradient

            state = next_state

        if args.epsilon_final_at:
            epsilon = np.interp(env.episode + 1, [0, args.epsilon_final_at], [args.epsilon, args.epsilon_final])

        if env.episode == 5000:
            np.save("w.npy", W)
            training = False
    

    # Final evaluation
    while True:
        state, done = env.reset(start_evaluation=True), False
        while not done:
            # TODO: Choose (greedy) action
            action = np.argmax(np.sum(W[state], axis=0))
            state, reward, done, _ = env.step(action)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    env = wrappers.EvaluationEnv(wrappers.DiscreteMountainCarWrapper(gym.make("MountainCar1000-v0"), tiles=args.tiles), args.seed)

    main(env, args)
