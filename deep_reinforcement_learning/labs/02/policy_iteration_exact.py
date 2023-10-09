#!/usr/bin/env python3
import argparse

import numpy as np

class GridWorld:
    # States in the gridworld are the following:
    # 0 1 2 3
    # 4 x 5 6
    # 7 8 9 10

    # The rewards are +1 in state 10 and -100 in state 6

    # Actions are ↑ → ↓ ←; with probability 80% they are performed as requested,
    # with 10% move 90° CCW is performed, with 10% move 90° CW is performed.
    states: int = 11

    actions: list[str] = ["↑", "→", "↓", "←"]

    @staticmethod
    def step(state: int, action: int) -> list[tuple[float, float, int]]:
        return [GridWorld._step(0.8, state, action),
                GridWorld._step(0.1, state, (action + 1) % 4),
                GridWorld._step(0.1, state, (action + 3) % 4)]

    @staticmethod
    def _step(probability: float, state: int, action: int) -> tuple[float, float, int]:
        if state >= 5: state += 1
        x, y = state % 4, state // 4
        offset_x = -1 if action == 3 else action == 1
        offset_y = -1 if action == 0 else action == 2
        new_x, new_y = x + offset_x, y + offset_y
        if not(new_x >= 4 or new_x < 0  or new_y >= 3 or new_y < 0 or (new_x == 1 and new_y == 1)):
            state = new_x + 4 * new_y
        if state >= 5: state -= 1
        return (probability, +1 if state == 10 else -100 if state == 6 else 0, state)

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--gamma", default=1.0, type=float, help="Discount factor.")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--steps", default=10, type=int, help="Number of policy evaluation/improvements to perform.")
# If you add more arguments, ReCodEx will keep them with your default values.

def argmax_with_tolerance(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Argmax with small tolerance, choosing the value with smallest index on ties"""
    x = np.asarray(x)
    return np.argmax(x + 1e-6 >= np.max(x, axis=axis, keepdims=True), axis=axis)

def get_best_policy(state: int, value_function: list[float], args: argparse.Namespace):
    policies_values = np.zeros(4)
    for a in range(4):
        step_results = GridWorld.step(state, a)
        for result in step_results:
            prob, reward, new_state = result
            policies_values[a] += prob * (reward + args.gamma * value_function[new_state])
    return argmax_with_tolerance(policies_values)

def main(args: argparse.Namespace) -> tuple[list[float], list[int]]:
    # Start with zero value function and "go North" policy
    value_function = [0.0] * GridWorld.states
    policy = [0] * GridWorld.states

    # TODO: Implement policy iteration algorithm, with `args.steps` steps of
    # policy evaluation/policy improvement. During policy evaluation, compute
    # the value function exactly by solving the system of linear equations.
    # During the policy improvement, use the `argmax_with_tolerance` to
    # choose the best action.
    
    for i in range(args.steps):
        # Policy Evaluation

        a = np.zeros((GridWorld.states, GridWorld.states))
        b = np.zeros((GridWorld.states))
        for s in range(GridWorld.states):
            a[s,s] += 1
            step_results = GridWorld.step(s, policy[s])
            for result in step_results:
                prob, reward, new_state = result
                b[s] += prob * reward
                a[s, new_state] -= prob * args.gamma
        
        value_function = np.linalg.solve(a, b)

        # Policy Improvement
        policy_stable = True
        for s in range(GridWorld.states):
            old_action = policy[s]
            policy[s] = get_best_policy(s, value_function, args)
            if old_action != policy[s]:
                policy_stable = False
        #if policy_stable: # policy can be stable but value_function V can still change until policy changes in some future iteration
        #    break

    # TODO: The final value function should be in `value_function` and final greedy policy in `policy`.
    return value_function, policy

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    value_function, policy = main(args)

    # Print results
    for l in range(3):
        for c in range(4):
            state = l * 4 + c
            if state >= 5: state -= 1
            print("        " if l == 1 and c == 1 else "{:-8.2f}".format(value_function[state]), end="")
            print(" " if l == 1 and c == 1 else GridWorld.actions[policy[state]], end="")
        print()
