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
parser.add_argument("--alpha", default=0.5, type=float, help="Learning rate.")
parser.add_argument("--alpha_decay", default=0.8, type=float, help="Learning rate decay.")
parser.add_argument("--epsilon", default=0.4, type=float, help="Exploration factor.")
parser.add_argument("--epsilon_decay", default=0.4, type=float, help="Exploration factor decay.")
parser.add_argument("--epsilon_decay_rate", default=200, type=float, help="Exploration factor decay rate (after how many episodes should it decay).")
parser.add_argument("--gamma", default=0.99, type=float, help="Discounting factor.")
parser.add_argument("--train_episodes", default=4000, type=int, help="Number of training episodes.")
    
def epsilon_greedy(env: wrappers.EvaluationEnv, epsilon: float, greedy_action: int) -> int:
    if np.random.uniform() < epsilon:
        return np.random.randint(env.action_space.n)
    return greedy_action

def main(env: wrappers.EvaluationEnv, args: argparse.Namespace) -> None:
    # Fix random seed
    np.random.seed(args.seed)

    # TODO: Variable creation and initialization
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    policy = np.zeros((env.observation_space.n), dtype=int)
    epsilon = args.epsilon
    alpha = args.alpha

    training = True
    while training:
        # Perform episode
        state, done = env.reset(), False
        while not done:
            if args.render_each and env.episode > 0 and env.episode % args.render_each == 0:
                env.render()

            # TODO: Perform an action.

            action = epsilon_greedy(env, epsilon, policy[state])
            next_state, reward, done, _ = env.step(action)

            # TODO: Update the action-value estimates
            Q[state, action] += alpha * (reward + args.gamma * np.max(Q[next_state, :]) - Q[state, action])
            policy[state] = np.argmax(Q[state,:])
            state = next_state
        if env.episode > 0 and env.episode % args.epsilon_decay_rate == 0:
            epsilon *= args.epsilon_decay
            alpha *= args.alpha_decay
        if env.episode == args.train_episodes:
            training = False

        

    # Final evaluation
    while True:
        state, done = env.reset(start_evaluation=True), False
        while not done:
            # TODO: Choose (greedy) action
            action = np.argmax(Q[state,:])
            state, reward, done, _ = env.step(action)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    env = wrappers.EvaluationEnv(wrappers.DiscreteMountainCarWrapper(gym.make("MountainCar1000-v0")), args.seed)

    main(env, args)
