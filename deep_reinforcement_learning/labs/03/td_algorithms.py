#!/usr/bin/env python3
import argparse

import gym
import numpy as np

import wrappers

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--alpha", default=0.1, type=float, help="Learning rate alpha.")
parser.add_argument("--episodes", default=1000, type=int, help="Training episodes.")
parser.add_argument("--epsilon", default=0.1, type=float, help="Exploration epsilon factor.")
parser.add_argument("--gamma", default=0.99, type=float, help="Discount factor gamma.")
parser.add_argument("--mode", default="sarsa", type=str, help="Mode (sarsa/expected_sarsa/tree_backup).")
parser.add_argument("--n", default=1, type=int, help="Use n-step method.")
parser.add_argument("--off_policy", default=False, action="store_true", help="Off-policy; use greedy as target")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=47, type=int, help="Random seed.")
# If you add more arguments, ReCodEx will keep them with your default values.

def argmax_with_tolerance(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Argmax with small tolerance, choosing the value with smallest index on ties"""
    x = np.asarray(x)
    return np.argmax(x + 1e-6 >= np.max(x, axis=axis, keepdims=True), axis=axis)

def main(args: argparse.Namespace) -> np.ndarray:
    # Create a random generator with a fixed seed
    generator = np.random.RandomState(args.seed)

    # Create the environment
    env = wrappers.EvaluationEnv(gym.make("Taxi-v3"), seed=args.seed, report_each=min(200, args.episodes))

    Q = np.zeros((env.observation_space.n, env.action_space.n))

    # The next action is always chosen in the epsilon-greedy way.
    def choose_next_action(Q: np.ndarray) -> tuple[int, float]:
        greedy_action = argmax_with_tolerance(Q[next_state])
        next_action = greedy_action if generator.uniform() >= args.epsilon else env.action_space.sample()
        return next_action, args.epsilon / env.action_space.n + (1 - args.epsilon) * (greedy_action == next_action)

    # The target policy is either the behavior policy (if not args.off_policy),
    # or the greedy policy (if args.off_policy).
    def compute_target_policy(Q: np.ndarray) -> np.ndarray:
        target_policy = np.eye(env.action_space.n)[argmax_with_tolerance(Q, axis=-1)]
        if not args.off_policy:
            target_policy = (1 - args.epsilon) * target_policy + args.epsilon / env.action_space.n
        return target_policy

    ACTION = 0
    ACT_PROB = 1
    STATE = 2
    REWARD = 3

    # Run the TD algorithm
    for _ in range(args.episodes):
        next_state, done = env.reset(), False
        step_history = []

        # Generate episode and update Q using the given TD method
        next_action, next_action_prob = choose_next_action(Q)
        step_history.append((next_action, next_action_prob, next_state, 0))
        T = float("inf")
        t = 0 # episode step
        tau = t - args.n + 1
        while tau != T-1:
            action, action_prob, state = next_action, next_action_prob, next_state
            if t < T:
                next_state, reward, done, _ = env.step(action)
                if done:
                    T = t+1
                    step_history.append((None, None, next_state, reward))
                else:
                    next_action, next_action_prob = choose_next_action(Q)
                    step_history.append((next_action, next_action_prob, next_state, reward))

            # TODO: Perform the update to the state-action value function `Q`, using
            # a TD update with the following parameters:
            # - `args.n`: use `args.n`-step method
            # - `args.off_policy`:
            #    - if False, the epsilon-greedy behaviour policy is also the target policy
            #    - if True, the target policy is the greedy policy
            #      - for SARSA (with any `args.n`) and expected SARSA (with `args.n` > 1),
            #        importance sampling must be used
            # - `args.mode`: this argument can have the following values:
            #   - "sarsa": regular SARSA algorithm
            #   - "expected_sarsa": expected SARSA algorithm
            #   - "tree_backup": tree backup algorithm
            #
            # Perform the updates as soon as you can -- whenever you have all the information
            # to update `Q[state, action]`, do it. For each `action` use its corresponding
            # `action_prob` at the time of taking the `action` as the behaviour policy probability,
            # and the `compute_target_policy(Q)` with the current `Q` as the target policy.
            #
            # Do not forget that when `done` is True, bootstrapping on the
            # `next_state` is not used.
            #
            # Also note that when the episode ends and `args.n` > 1, there will
            # be several state-action pairs that also need to be updated. Perform
            # the updates in the order in which you encountered the state-action
            # pairs and during these updates, use the `compute_target_policy(Q)`
            # with the up-to-date value of `Q`.
            tau = t - args.n + 1 # index of state being updated

            if tau >= 0:
                G = 0
                if args.mode == "sarsa" or args.mode == "expected_sarsa":
                    for i in range(tau + 1, min(tau+args.n+1, T+1)):
                        i_reward = step_history[i][REWARD] 
                        G += (args.gamma ** (i - tau - 1)) * i_reward
                    if tau+args.n < T:
                        state_taun = step_history[tau+args.n][STATE]
                        action_taun = step_history[tau+args.n][ACTION]
                        if args.mode == "sarsa":
                            G += (args.gamma ** args.n) * Q[state_taun, action_taun]
                        elif args.mode == "expected_sarsa":
                            state_tp = current_target_policy(Q)[state_taun]
                            expected_q = np.dot(state_tp, Q[state_taun])
                            G += (args.gamma ** args.n) * expected_q
                else: # args.mode == "tree_backup"
                    ctp = current_target_policy(Q)
                    if t + 1 >= T:
                        G = step_history[T][REWARD]
                    else:
                        cur_step = step_history[t+1]
                        cur_state = cur_step[STATE]
                        state_tp = ctp[cur_state]
                        G = cur_step[REWARD] + args.gamma * np.dot(state_tp, Q[cur_state])
                    for k in range(min(t, T-1), tau, -1):
                        action_k, _, state_k, reward_k = step_history[k]
                        G = reward_k + args.gamma * \
                            np.dot(
                                np.delete(ctp[state_k], action_k),
                                np.delete(Q[state_k], action_k)
                            ) + args.gamma * ctp[state_k, action_k] * G


                state_tau = step_history[tau][STATE]
                action_tau = step_history[tau][ACTION]
                if not args.off_policy:
                    prob_ratio = 1
                else:
                    tp = current_target_policy(Q)
                    prob_ratio = 1
                    if args.mode == "sarsa" :
                        for i in range(tau + 1, min(tau + args.n+1, T)):
                            i_prob = step_history[i][ACT_PROB]
                            i_state = step_history[i][STATE]
                            i_action = step_history[i][ACTION]
                            prob_ratio *= tp[i_state, i_action] / i_prob
                    elif args.mode == "expected_sarsa" and args.n > 1:
                        for i in range(tau + 1, min(tau + args.n, T)):
                            i_prob = step_history[i][ACT_PROB]
                            i_state = step_history[i][STATE]
                            i_action = step_history[i][ACTION]
                            prob_ratio *= tp[i_state, i_action] / i_prob
                Q[state_tau, action_tau] += args.alpha * prob_ratio * (G - Q[state_tau, action_tau])

            t += 1

    return Q

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
