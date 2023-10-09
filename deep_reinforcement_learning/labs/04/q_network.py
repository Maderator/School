#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import os
import random

from numpy.core.fromnumeric import choose
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2") # Report only TF errors by default

import gym
import numpy as np
import tensorflow as tf

import wrappers

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
parser.add_argument("--epsilon", default=0.5, type=float, help="Exploration factor.")
parser.add_argument("--epsilon_final", default=0.05, type=float, help="Final exploration factor.")
parser.add_argument("--epsilon_final_at", default=500, type=int, help="Training episodes.")
parser.add_argument("--gamma", default=0.95, type=float, help="Discounting factor.")
parser.add_argument("--hidden_layer_size", default=50, type=int, help="Size of hidden layer.")
parser.add_argument("--learning_rate", default=0.001, type=float, help="Learning rate.")
parser.add_argument("--target_update_freq", default=1, type=int, help="Target update frequency.")
parser.add_argument("--evaluation_freq", default=50, type=int, help="Periodically evaluate after given number episodes.")
parser.add_argument("--evaluation_length", default=50, type=int, help="Evaluate on given number of episodes.")

class Network:
    def __init__(self, env: wrappers.EvaluationEnv, args: argparse.Namespace) -> None:
        # TODO: Create a suitable model. The rest of the code assumes
        # it is stored as `self._model` and has been `.compile()`-d.
        input_layer = tf.keras.layers.Input(shape=(4)) # state has 4 features
        hidden_layer = tf.keras.layers.Dense(args.hidden_layer_size, activation=tf.keras.activations.relu)(input_layer)
        output_layer = tf.keras.layers.Dense(2)(hidden_layer) # there are 2 possible actions
        self._model = tf.keras.models.Model(input_layer, output_layer)
        self._model.compile(
            loss=tf.keras.losses.MSE,
            optimizer=tf.keras.optimizers.RMSprop(learning_rate=args.learning_rate)
        )


    # Define a training method. Generally you have two possibilities
    # - pass new q_values of all actions for a given state; all but one are the same as before
    # - pass only one new q_value for a given state, and include the index of the action to which
    #   the new q_value belongs
    # The code below implements the first option, but you can change it if you want.
    # Also note that we need to use @tf.function for efficiency (using `train_on_batch`
    # on extremely small batches/networks has considerable overhead).
    #
    # The `wrappers.typed_np_function` automatically converts input arguments
    # to NumPy arrays of given type, and converts the result to a NumPy array.
    @wrappers.typed_np_function(np.float32, np.float32)
    @tf.function
    def train(self, states: np.ndarray, q_values: np.ndarray) -> None:
        self._model.optimizer.minimize(
            lambda: self._model.compiled_loss(q_values, self._model(states, training=True)),
            var_list=self._model.trainable_variables
        )

    # Predict method, again with manual @tf.function for efficiency.
    @wrappers.typed_np_function(np.float32)
    @tf.function
    def predict(self, states: np.ndarray) -> np.ndarray:
        return self._model(states)

    # If you want to use target network, the following method copies weights from
    # a given Network to the current one.
    @tf.function
    def copy_weights_from(self, other: Network) -> None:
        for var, other_var in zip(self._model.variables, other._model.variables):
            var.assign(other_var)

def epsilon_greedy(env: wrappers.EvaluationEnv, epsilon: float, greedy_action: int) -> int:
    if np.random.uniform() < epsilon:
        return np.random.randint(env.action_space.n)
    return greedy_action

def main(env: wrappers.EvaluationEnv, args: argparse.Namespace) -> None:
    # Fix random seeds and number of threads
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Construct the network
    network = Network(env, args)

    # Replay memory; maxlen parameter can be passed to deque for a size limit,
    # which we however do not need in this simple task.
    replay_buffer = collections.deque()
    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "done", "next_state"])

    epsilon = args.epsilon
    training = True
    while training:
        # Perform episode
        state, done = env.reset(), False
        while not done:
            if args.render_each and env.episode > 0 and env.episode % args.render_each == 0:
                env.render()

            # TODO: Choose an action.
            # You can compute the q_values of a given state by
            #   q_values = network.predict([state])[0]
            q_values = network.predict([state])[0]
            action = epsilon_greedy(env, epsilon, np.argmax(q_values))

            next_state, reward, done, _ = env.step(action)

            # Append state, action, reward, done and next_state to replay_buffer
            replay_buffer.append(Transition(state, action, reward, done, next_state))

            # TODO: If the replay_buffer is large enough, preform a training batch
            # from `args.batch_size` uniformly randomly chosen transitions.
            #
            # After you choose `states` and suitable targets, you can train the network as
            #   network.train(states, ...)
            if len(replay_buffer) >= args.batch_size * 10 and \
                len(replay_buffer) % args.target_update_freq == 0:
                transitions = random.sample(replay_buffer, args.batch_size)
                
                states = [tr.state for tr in transitions]
                next_states = [tr.next_state for tr in transitions]
                targets = np.zeros(shape=(len(states), env.action_space.n))
                q_val_pred = network.predict(states)
                next_q_val_pred = network.predict(next_states)
                for i, tran in enumerate(transitions):
                    target = q_val_pred[i]
                    if not tran.done:
                        target[tran.action] = tran.reward + \
                            args.gamma * np.max(next_q_val_pred[i])
                    else:
                        target[tran.action] = tran.reward
                    targets[i] = target

                network.train(states, targets)

            state = next_state

        if args.epsilon_final_at:
            epsilon = np.interp(env.episode + 1, [0, args.epsilon_final_at], [args.epsilon, args.epsilon_final])

        if env.episode % args.evaluation_freq == 0 and env.episode > 0:
            def evaluate():
                episode_returns = env._episode_returns
                report_each = env._report_each

                env._episode_returns = []
                env._report_each = 0
                for _ in range(args.evaluation_length):
                    state, done = env.reset(start_evaluation=False), False
                    while not done:
                        action = np.argmax(network.predict([state])[0])
                        state, reward, done, _ = env.step(action)
                evaluation_return = np.mean(env._episode_returns[-50:])
            
                env._episode_returns = episode_returns
                env._report_each = report_each
                print(f"Evaluation after episode {env.episode} returned {evaluation_return}")

                return evaluation_return

            evaluation_return = evaluate()
            if evaluation_return > 475:
                training = False

    # Final evaluation
    while True:
        state, done = env.reset(start_evaluation=True), False
        while not done:
            # TODO: Choose (greedy) action
            q_values = network.predict([state])[0]
            action = np.argmax(q_values)
            state, reward, done, _ = env.step(action)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    env = wrappers.EvaluationEnv(gym.make("CartPole-v1"), args.seed)

    main(env, args)
