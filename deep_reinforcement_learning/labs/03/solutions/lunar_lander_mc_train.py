#!/usr/bin/env python3
import argparse
import lzma
import os
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
parser.add_argument("--epsilon", default=None, type=float, help="Exploration factor.")
parser.add_argument("--epsilon_final", default=None, type=float, help="Final exploration factor.")
parser.add_argument("--epsilon_final_at", default=None, type=int, help="Final exploration factor frame.")
parser.add_argument("--init", default=None, type=str, help="Initial estimate path.")
parser.add_argument("--init_count_limit", default=None, type=int, help="Limit on initial counts.")
parser.add_argument("--output", default=None, type=str, help="Output path.")
parser.add_argument("--threads", default=1, type=int, help="Threads to use.")

def main(env, args):
    # Fix random seed
    np.random.seed(args.seed)

    num_envs, num_actions = args.threads, env.action_space.n
    aenv = gym.vector.AsyncVectorEnv([lambda: wrappers.DiscreteLunarLanderWrapper(gym.make("LunarLander-v2"))] * num_envs)

    if args.init:
        with open(args.init, "rb") as init_file:
            Q, C = pickle.load(init_file)
        if args.init_count_limit:
            C = np.minimum(C, args.init_count_limit)
    else:
        Q = np.zeros([env.observation_space.n, num_actions])
        C = np.zeros([env.observation_space.n, num_actions])

    epsilon, episode = args.epsilon, 0
    while True:
        # Training
        state = aenv.reset()
        states = [[] for _ in range(num_envs)]
        actions = [[] for _ in range(num_envs)]
        rewards = [[] for _ in range(num_envs)]
        returns = []

        while len(returns) < 10000:
            action = np.argmax(Q[state], axis=-1)

            greedy_mask = np.random.uniform(size=num_envs) >= args.epsilon
            random_action = np.random.randint(num_actions, size=num_envs)
            action = greedy_mask * action + (1 - greedy_mask) * random_action

            # Perform the action.
            next_state, reward, done, _ = aenv.step(action)

            for i in range(num_envs):
                states[i].append(state[i])
                actions[i].append(action[i])
                rewards[i].append(reward[i])

                if done[i]:
                    episode += 1
                    if args.epsilon_final_at:
                        epsilon = np.interp(episode, [0, args.epsilon_final_at], [args.epsilon, args.epsilon_final])

                    G = 0
                    for (s, a, r) in zip(reversed(states[i]), reversed(actions[i]), reversed(rewards[i])):
                        G += r
                        C[s, a] += 1
                        Q[s, a] += 1 / C[s, a] * (G - Q[s, a])
                    states[i], actions[i], rewards[i] = [], [], []

                    returns.append(G)
                    if len(returns) >= 10000:
                        break
            state = next_state
        training = np.mean(returns)

        # Evalution
        state = aenv.reset()
        rewards, returns = np.zeros(num_envs), []
        while len(returns) < 1000:
            action = np.argmax(Q[state], axis=-1)
            state, reward, done, _ = aenv.step(action)
            rewards += reward
            for i in range(num_envs):
                if done[i]:
                    returns.append(rewards[i])
                    rewards[i] = 0
        evaluation = np.mean(returns)

        if args.output:
            with lzma.open(os.path.join(args.output, "{}.py.xz".format(episode)), "wt") as output_file:
                print("policy = [{}]".format(",".join(str(action) for action in np.argmax(Q, axis=-1))), file=output_file)

        print("Episode {}, epsilon {}, 10000-average training {}, 1000-average evaluation {}".format(episode, epsilon, training, evaluation), flush=True)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    env = wrappers.EvaluationEnv(wrappers.DiscreteLunarLanderWrapper(gym.make("LunarLander-v2")), args.seed, evaluate_for=1000)

    main(env, args)
