#!/usr/bin/env python3
import argparse
import os

from gym import vector
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
parser.add_argument("--entropy_regularization", default=None, type=float, help="Entropy regularization weight.")
parser.add_argument("--env", default="CartPole-v1", type=str, help="Environment.")
parser.add_argument("--evaluate_each", default=200, type=int, help="Evaluate each number of batches.")
parser.add_argument("--evaluate_for", default=10, type=int, help="Evaluate the given number of episodes.")
parser.add_argument("--gamma", default=0.9, type=float, help="Discounting factor.")
parser.add_argument("--hidden_layer_size", default=100, type=int, help="Size of hidden layer.")
parser.add_argument("--learning_rate", default=0.001, type=float, help="Learning rate.")
parser.add_argument("--workers", default=32, type=int, help="Number of parallel workers.")
parser.add_argument("--min_return", default=480, type=int, help="Minimal return for training to stop.")

class Network:
    def __init__(self, env: wrappers.EvaluationEnv, args: argparse.Namespace) -> None:
        # TODO: Similarly to reinforce with baseline, define two components:
        # - actor, which predicts distribution over the actions
        # - critic, which predicts the value function
        #
        # Use independent networks for both of them, each with
        # `args.hidden_layer_size` neurons in one ReLU hidden layer,
        # and train them using Adam with given `args.learning_rate`.
        in_shape = env.observation_space.shape
        inputs = tf.keras.layers.Input(shape=in_shape)
        hidden = tf.keras.layers.Dense(args.hidden_layer_size, activation="relu")(inputs)
        outputs = tf.keras.layers.Dense(env.action_space.n, "softmax")(hidden)
        self._policy_network = tf.keras.Model(inputs, outputs)

        self._policy_network.compile(
            optimizer=tf.optimizers.Adam(learning_rate=args.learning_rate),
            loss=tf.losses.CategoricalCrossentropy(),
            metrics=tf.metrics.CategoricalCrossentropy()
        )
        
        inputs = tf.keras.layers.Input(shape=in_shape)
        hidden = tf.keras.layers.Dense(args.hidden_layer_size, activation="relu")(inputs)
        outputs = tf.keras.layers.Dense(1, activation=None)(hidden)
        self._value_network = tf.keras.Model(inputs, outputs)

        self._value_network.compile(
            optimizer=tf.optimizers.Adam(learning_rate=args.learning_rate),
            loss=tf.losses.MeanSquaredError(),
            metrics=tf.metrics.MeanSquaredError()
        )

    # The `wrappers.typed_np_function` automatically converts input arguments
    # to NumPy arrays of given type, and converts the result to a NumPy array.
    @wrappers.typed_np_function(np.float32, np.int32, np.float32)
    @tf.function
    def train(self, states: np.ndarray, actions: np.ndarray, returns: np.ndarray) -> None:
        # TODO: Train the policy network using policy gradient theorem
        # and the value network using MSE.
        #
        # The `args.entropy_regularization` might be used to include actor
        # entropy regularization -- however, the assignment can be solved
        # quite easily without it (my reference solution does not use it).
        # In any case, `tfp.distributions.Categorical` is the suitable distribution;
        # in PyTorch, it is `torch.distributions.categorical.Categorical`.
        with tf.GradientTape() as tape:
            pred_policy = self._policy_network(states)
            loss = self._policy_network.compiled_loss(y_true=actions, y_pred=pred_policy, sample_weight=returns)
        self._policy_network.optimizer.minimize(loss, self._policy_network.trainable_variables, tape=tape)
        self._policy_network.compiled_metrics.update_state(y_true=actions, y_pred=pred_policy, sample_weight=returns)

        with tf.GradientTape() as tape:
            value_pred = self._value_network(states)
            loss = self._value_network.compiled_loss(y_true=returns, y_pred=value_pred)
        self._value_network.optimizer.minimize(loss, self._value_network.trainable_variables, tape=tape)
        self._value_network.compiled_metrics.update_state(y_true=returns, y_pred=value_pred)
        
    @wrappers.typed_np_function(np.float32)
    @tf.function
    def predict_actions(self, states: np.ndarray) -> np.ndarray:
        # TODO: Return predicted action probabilities.
        return self._policy_network(states)

    @wrappers.typed_np_function(np.float32)
    @tf.function
    def predict_values(self, states: np.ndarray) -> np.ndarray:
        # TODO: Return estimates of value function.
        return self._value_network(states)

def main(env: wrappers.EvaluationEnv, args: argparse.Namespace) -> None:
    # Fix random seeds and number of threads
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Construct the network
    network = Network(env, args)

    def evaluate_episode(start_evaluation:bool = False) -> float:
        rewards, state, done = 0, env.reset(start_evaluation), False
        while not done:
            #print("evaluate_episode/state shape:", state.shape)
            if args.render_each and env.episode > 0 and env.episode % args.render_each == 0:
                env.render()

            # TODO: Predict the action using the greedy policy
            # My note: not vectorized
            action_dist = network.predict_actions([state])
            action = np.argmax(action_dist[0])
            state, reward, done, _ = env.step(action)
            rewards += reward
        return rewards

    # Create the vectorized environment
    vector_env = gym.vector.AsyncVectorEnv([lambda: gym.make(args.env)] * args.workers)
    vector_env.seed(args.seed)
    states = vector_env.reset()

    training = True
    rewards = np.zeros((states.shape[0]))
    while training:
        # Training
        for _ in range(args.evaluate_each):
            # TODO: Choose actions using network.predict_actions
            actions_dist = network.predict_actions(states)
            actions = [np.random.choice(env.action_space.n, p=dist) for dist in actions_dist]

            # TODO: Perform steps in the vectorized environment
            next_states, step_rewards, done, _ = vector_env.step(actions)
            rewards = rewards * (done == False) + step_rewards

            # TODO: Compute estimates of returns by one-step bootstrapping
            pred_next_value = network.predict_values(next_states)
            pred_next_value_reshaped = np.reshape(pred_next_value, (pred_next_value.shape[0]))
            returns = rewards + args.gamma * (done == False) * pred_next_value_reshaped

            # TODO: Train network using current states, chosen actions and estimated returns
            actions_oh = tf.one_hot(actions, env.action_space.n)
            network.train(states, actions_oh, returns)

            states = next_states

        # Periodic evaluation
        sum = 0
        for _ in range(args.evaluate_for):
            sum += evaluate_episode()
        mean = sum / args.evaluate_for
        if mean >= args.min_return:
            training = False

    # Final evaluation
    while True:
        evaluate_episode(start_evaluation=True)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    env = wrappers.EvaluationEnv(gym.make(args.env), args.seed)

    main(env, args)
