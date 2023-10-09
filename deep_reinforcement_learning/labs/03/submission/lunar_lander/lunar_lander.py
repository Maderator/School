#!/usr/bin/env python3

# IDs:
#edbe2dad-018e-11eb-9574-ea7484399335
#aa20f311-c7a2-11e8-a4be-00505601122b
#c716f6b0-25ab-11ec-986f-f39926f24a9c

import argparse

import gym
import numpy as np
import sys
import os.path

import wrappers


Q1_MATRIX_FILENAME = 'Q1.bin'
Q2_MATRIX_FILENAME = 'Q2.bin'

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--alpha", default=0.05, type=float, help="Learning rate.")
parser.add_argument("--alpha_decay", default=1.0, type=float, help="Learning rate decay.")
parser.add_argument("--epsilon", default=0.00005, type=float, help="Exploration factor.")
parser.add_argument("--epsilon_decay", default=0.78, type=float, help="Exploration factor decay.")
parser.add_argument("--epsilon_decay_rate", default=2000, type=float, help="Exploration factor decay rate (after how many episodes should it decay).")
parser.add_argument("--gamma", default=0.95, type=float, help="Discounting factor.")
parser.add_argument("--train_episodes", default=0, type=int, help="Number of training episodes.")
parser.add_argument("--evaluate_for", default=1000, type=int, help="Number of evaluation episodes.")
parser.add_argument("--return_criterion", default=180, type=int, help="Points needed for training to stop.")
parser.add_argument("--expert_training", default=True, action="store_true", help="Running in ReCodEx")
parser.add_argument("--normal_training", default=True, action="store_true", help="Running in ReCodEx")


def mean_episode_return(env: wrappers.EvaluationEnv):
    return np.mean(env._episode_returns[-args.evaluate_for:])

def epsilon_greedy(env: wrappers.EvaluationEnv, epsilon: float, greedy_action: int) -> int:
    if np.random.uniform() < epsilon:
        return np.random.randint(env.action_space.n)
    return greedy_action

def load_or_init_matrix(filename: str, first_d: int, second_d: int) -> np.ndarray:
    if os.path.isfile(filename):
        return np.fromfile(filename).reshape((first_d, second_d))
    else:
        return np.zeros((first_d, second_d))


def main(env: wrappers.EvaluationEnv, args: argparse.Namespace) -> None:
    # Fix random seed
    np.random.seed(args.seed)

    n_states = env.observation_space.n
    n_actions = env.action_space.n

    # TODO: Implement a suitable RL algorithm.
    Q1 = load_or_init_matrix(Q1_MATRIX_FILENAME, n_states, n_actions)
    Q2 = load_or_init_matrix(Q2_MATRIX_FILENAME, n_states, n_actions)
    #Q1 = np.zeros((env.observation_space.n, env.action_space.n))
    epsilon = args.epsilon
    alpha = args.alpha

    expert_episode = 0

    training = not args.recodex
    while training:
        # To generate expert trajectory, you can use
        if args.expert_training:
            state, trajectory = env.expert_trajectory()
            #print(state)
            for step in trajectory:
                action, reward, next_state = step
                if np.random.uniform() < 0.5:
                    Q1[state, action] += alpha * 2 * (reward + args.gamma * Q2[next_state, np.argmax(Q1[next_state,:])] - Q1[state, action]) # gamma = 1
                else:
                    Q2[state, action] += alpha * 2 * (reward + args.gamma * Q1[next_state, np.argmax(Q2[next_state,:])] - Q2[state, action]) # gamma = 1
                state = next_state
            if expert_episode % 10 == 0:
                print(f"Total episode {expert_episode + env.episode}")
            expert_episode += 1
        
        ## TODO: Perform a training episode
        if args.normal_training:
            state, done = env.reset(), False
            #print(state)
            while not done:
                if args.render_each and env.episode and env.episode % args.render_each == 0:
                    env.render()

                action = epsilon_greedy(env, epsilon, np.argmax(Q1[state,:] + Q2[state,:]))
                next_state, reward, done, _ = env.step(action)
                if np.random.uniform() < 0.5:
                    Q1[state, action] += alpha * (reward + args.gamma * Q2[next_state, np.argmax(Q1[next_state,:])] - Q1[state, action])
                else:
                    Q2[state, action] += alpha * (reward + args.gamma * Q1[next_state, np.argmax(Q2[next_state,:])] - Q2[state, action])
                state = next_state

            if (env.episode + expert_episode > 0 and env.episode + expert_episode % args.epsilon_decay_rate == 0):
                epsilon *= args.epsilon_decay
                alpha *= args.alpha_decay
        
        #if mean_episode_return(env) > args.return_criterion:
        if env.episode + expert_episode > args.train_episodes:
            print("saving Q1 and Q2")
            Q1.tofile(Q1_MATRIX_FILENAME)
            Q2.tofile(Q2_MATRIX_FILENAME)
            training = False

    # Final evaluation
    Q1 = np.fromfile(Q1_MATRIX_FILENAME).reshape((n_states, n_actions))
    Q2 = np.fromfile(Q2_MATRIX_FILENAME).reshape((n_states, n_actions))
    while True:
        state, done = env.reset(start_evaluation=True), False
        while not done:
            # Choose (greedy) action
            action = np.argmax(Q1[state,:] + Q2[state,:])
            state, reward, done, _ = env.step(action)

    #while True:
    #    state, done = env.reset(start_evaluation=True), False
    #    while not done:
    #        # TODO: Choose (greedy) action
    #        action = np.argmax(Q[state, :])
    #        state, reward, done, _ = env.step(action)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    env = wrappers.EvaluationEnv(wrappers.DiscreteLunarLanderWrapper(gym.make("LunarLander-v2")), args.seed)

    main(env, args)
