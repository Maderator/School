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
parser.add_argument("--episodes", default=2000, type=int, help="Training episodes.")
parser.add_argument("--epsilon", default=0.3, type=float, help="Exploration factor.")
parser.add_argument("--epsilon_final", default=0.1, type=float, help="Final exploration factor.")

def main(env: wrappers.EvaluationEnv, args: argparse.Namespace):
    # Fix random seed
    np.random.seed(args.seed)

    # TODO:
    # - Create Q, a zero-filled NumPy array with shape [number of states, number of actions],
    #   representing estimated Q value of a given (state, action) pair.
    # - Create C, a zero-filled NumPy array with the same shape,
    #   representing number of observed returns of a given (state, action) pair.
    Q = np.zeros([env.observation_space.n, env.action_space.n])
    C = np.zeros([env.observation_space.n, env.action_space.n])
    epsilon = args.epsilon

    for _ in range(args.episodes):
        # TODO: Perform an episode, collecting states, actions and rewards.

        state, done = env.reset(), False
        states, actions, rewards = [], [], []
        while not done:
            if args.render_each and env.episode > 0 and env.episode % args.render_each == 0:
                env.render()

            # TODO: Compute `action` using epsilon-greedy policy.
            action = np.argmax(Q[state]) if env.episode >= args.episodes or np.random.uniform() > epsilon else env.action_space.sample()

            # Perform the action.
            next_state, reward, done, _ = env.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            state = next_state

        # TODO: Compute returns from the recieved rewards and update Q and C.
        for i in reversed(range(len(rewards) - 1)):
            rewards[i] += rewards[i + 1]

        for i in range(len(rewards)):
            C[states[i]][actions[i]] += 1
            Q[states[i]][actions[i]] = Q[states[i]][actions[i]] + 1 / C[states[i]][actions[i]] * (rewards[i] - Q[states[i]][actions[i]])

        if args.epsilon_final:
            epsilon = np.interp(env.episode + 1, [0, args.episodes], [args.epsilon, args.epsilon_final])

    # Final evaluation
    while True:
        state, done = env.reset(start_evaluation=True), False
        while not done:
            # TODO: Choose a greedy action
            action = np.argmax(Q[state])
            state, reward, done, _ = env.step(action)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    env = wrappers.EvaluationEnv(wrappers.DiscreteCartPoleWrapper(gym.make("CartPole-v1")), args.seed)

    main(env, args)
