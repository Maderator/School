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
    def current_target_policy(Q: np.ndarray) -> np.ndarray:
        target_policy = np.eye(env.action_space.n)[argmax_with_tolerance(Q, axis=-1)]
        if not args.off_policy:
            target_policy = (1 - args.epsilon) * target_policy + args.epsilon / env.action_space.n
        return target_policy

    # Run the TD algorithm
    for _ in range(args.episodes):
        next_state, done = env.reset(), False
        states, actions, action_probs, rewards = [], [], [], []

        # Generate episode and update Q using the given TD method
        next_action, next_action_prob = choose_next_action(Q)
        while not done:
            action, action_prob, state = next_action, next_action_prob, next_state
            next_state, reward, done, _ = env.step(action)
            if not done:
                next_action, next_action_prob = choose_next_action(Q)

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
            # `action_prob` at the time of taking the `action` as the behaviour policy action
            # probability, and the `current_target_policy(Q)` as the target policy (everywhere
            # in the update).
            #
            # Do not forget that when `done` is True, bootstrapping on the
            # `next_state` is not used.
            #
            # Also note that when the episode ends and `args.n` > 1, there will
            # be several state-action pairs that also need to be updated. Perform
            # the updates in the order in which you encountered the state-action
            # pairs and during these updates, use the `current_target_policy(Q)`
            # with the up-to-date value of `Q`.
            states.append(state)
            actions.append(action)
            action_probs.append(action_prob)
            rewards.append(reward)

            while len(states) >= (args.n if not done else 1):
                target_policy = current_target_policy(Q)

                if args.mode == "sarsa":
                    return_ = sum(args.gamma ** i * r for i, r in enumerate(rewards + [(not done) * Q[next_state, next_action]]))
                if args.mode == "expected_sarsa":
                    return_ = sum(args.gamma ** i * r for i, r in enumerate(rewards + [(not done) * target_policy[next_state] @ Q[next_state]]))
                if args.mode == "tree_backup":
                    return_ = rewards[-1] + (not done) * args.gamma * target_policy[next_state] @ Q[next_state]
                    one_hot = np.eye(env.action_space.n)
                    for s, a, r in zip(reversed(states[1:]), reversed(actions[1:]), reversed(rewards[:-1])):
                        return_ = r + args.gamma * target_policy[s] @ (one_hot[a] * return_ + (1 - one_hot[a]) * Q[s])

                is_ratio = 1
                if args.off_policy and "sarsa" in args.mode:
                    is_ratio = np.prod([target_policy[s][a] / a_p for s, a, a_p in zip(states[1:], actions[1:], action_probs[1:])])
                    if args.mode == "sarsa" and not done: is_ratio *= target_policy[next_state][next_action] / next_action_prob

                Q[states[0], actions[0]] += args.alpha * is_ratio * (return_ - Q[states[0], actions[0]])
                states, actions, action_probs, rewards = states[1:], actions[1:], action_probs[1:], rewards[1:]

    return Q

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)