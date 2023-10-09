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
parser.add_argument("--alpha", default=0.2, type=float, help="Learning rate.")
parser.add_argument("--alpha_final", default=0.2, type=float, help="Final learning rate.")
parser.add_argument("--episodes", default=5000, type=int, help="Training episodes.")
parser.add_argument("--epsilon", default=0.5, type=float, help="Exploration factor.")
parser.add_argument("--epsilon_final", default=0.01, type=float, help="Final exploration factor.")
parser.add_argument("--gamma", default=1.0, type=float, help="Discounting factor.")

def main(env: wrappers.EvaluationEnv, args: argparse.Namespace) -> None:
    # Fix random seed
    np.random.seed(args.seed)

    # TODO: Variable creation and initialization
    Q = np.zeros([env.observation_space.n, env.action_space.n])
    epsilon = args.epsilon
    alpha = args.alpha
    rewards = []

    training = True
    while training:
        # Perform episode
        state, done = env.reset(), False
        total_reward = 0
        while not done:
            if args.render_each and env.episode > 0 and env.episode % args.render_each == 0:
                env.render()

            # TODO: Perform an action.
            action = np.argmax(Q[state]) if np.random.uniform() > epsilon else env.action_space.sample()

            next_state, reward, done, _ = env.step(action)

            # TODO: Update the action-value estimates
            Q[state][action] = Q[state][action] + alpha * (reward + (not done) * args.gamma * np.max(Q[next_state]) - Q[state][action])
            total_reward += reward

            state = next_state

        # End early if possible
        rewards.append(total_reward)
        if np.mean(rewards[-100:]) >= -135:
            training = False

        if env.episode < args.episodes:
            if args.epsilon_final:
                epsilon = np.exp(np.interp(env.episode + 1, [0, args.episodes], [np.log(args.epsilon), np.log(args.epsilon_final)]))
            if args.alpha_final:
                alpha = np.exp(np.interp(env.episode + 1, [0, args.episodes], [np.log(args.alpha), np.log(args.alpha_final)]))

    # Final evaluation
    while True:
        state, done = env.reset(start_evaluation=True), False
        while not done:
            # TODO: Choose (greedy) action
            action = np.argmax(Q[state])
            state, reward, done, _ = env.step(action)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    env = wrappers.EvaluationEnv(wrappers.DiscreteMountainCarWrapper(gym.make("MountainCar1000-v0")), args.seed)

    main(env, args)
